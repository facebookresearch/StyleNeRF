# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imp
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
import tqdm
import shutil
import legacy

from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from torch_utils.distributed_utils import gather_list_and_concat
from metrics import metric_main
from training.data_utils import save_image_grid, resize_image

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels, _ = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',      # Output directory.
    training_set_kwargs     = {},       # Options for training set.
    data_loader_kwargs      = {},       # Options for torch.utils.data.DataLoader.
    G_kwargs                = {},       # Options for generator network.
    D_kwargs                = {},       # Options for discriminator network.
    G_opt_kwargs            = {},       # Options for generator optimizer.
    D_opt_kwargs            = {},       # Options for discriminator optimizer.
    augment_kwargs          = None,     # Options for augmentation pipeline. None = disable.
    loss_kwargs             = {},       # Options for loss function.
    metrics                 = [],       # Metrics to evaluate during training.
    random_seed             = 0,        # Global random seed.
    world_size              = 1,        # Number of GPUs participating in the training.
    rank                    = 0,        # Rank of the current process.
    gpu                     = 0,        # Index of GPU used in training
    batch_gpu               = 4,        # Batch size for once GPU
    batch_size              = 4,        # Total batch size for one training iteration. Can be larger than batch_gpu * world_size.
    ema_kimg                = 10,       # Half-life of the exponential moving average (EMA) of generator weights.
    ema_rampup              = None,     # EMA ramp-up coefficient.
    G_reg_interval          = 4,        # How often to perform regularization for G? None = disable lazy regularization.
    D_reg_interval          = 16,       # How often to perform regularization for D? None = disable lazy regularization.
    augment_p               = 0,        # Initial value of augmentation probability.
    ada_target              = None,     # ADA target value. None = fixed p.
    ada_interval            = 4,        # How often to perform ADA adjustment?
    ada_kimg                = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
    total_kimg              = 25000,    # Total length of the training, measured in thousands of real images.
    kimg_per_tick           = 4,        # Progress snapshot interval.
    image_snapshot_ticks    = 50,       # How often to save image snapshots? None = disable.
    network_snapshot_ticks  = 50,       # How often to save network snapshots? None = disable.
    resume_pkl              = None,     # Network pickle to resume training from.
    resume_start            = 0,        # Resume from steps
    cudnn_benchmark         = True,     # Enable torch.backends.cudnn.benchmark?
    allow_tf32              = False,    # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
    abort_fn                = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
    progress_fn             = None,     # Callback function for updating training progress. Called for all ranks.
    update_cam_prior_ticks  = None,     # (optional) Non-parameteric updating camera poses of the dataset
    generation_with_image   = False,    # (optional) For each random z, you also sample an image associated with it.
    **unused,
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', gpu)
    np.random.seed(random_seed * world_size + rank)
    torch.manual_seed(random_seed * world_size + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark    # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32        # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    img_dir = run_dir + '/images'
    os.makedirs(img_dir, exist_ok=True)

    assert batch_gpu <= (batch_size // world_size)

    # Load training set.
    if rank == 0:
        print('Loading training set...')

    if world_size == 1:
        data_loader_kwargs.update({'num_workers': 1, 'prefetch_factor': 1})

    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    
    # Setup dataloader/sampler
    # if getattr(G.synthesis, 'sampler', None) is not None:
    #     raise NotImplementedError('ERROR NEED TO TAKE A LOOK')
    #     # training_set_sampler = G.synthesis.sampler[0](
    #     #     dataset=training_set, rank=rank, num_replicas=world_size, 
    #     #     seed=random_seed, device=device, **G.synthesis.sampler[1])
    # else:
    training_set_sampler = misc.InfiniteSampler(
            dataset=training_set, rank=rank, num_replicas=world_size, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(
        dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//world_size, **data_loader_kwargs))
    if generation_with_image:
        backup_data_iterator  = iter(torch.utils.data.DataLoader(
            dataset=copy.deepcopy(training_set), sampler=training_set_sampler, batch_size=batch_size//world_size, **data_loader_kwargs))


    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    if G_kwargs.get('img_channels', None) is not None:
        common_kwargs['img_channels'] = G_kwargs['img_channels']
        del G_kwargs['img_channels']
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()
    
    resize_real_img_early = D_kwargs.get('resize_real_early', False)
    disc_enable_ema = D_kwargs.get('enable_ema', False)
    if disc_enable_ema:
        D_ema = copy.deepcopy(D).eval()
    
    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        modules =  [('G', G), ('D', D), ('G_ema', G_ema)]
        if disc_enable_ema:
            modules += [('D_ema', D_ema)]
        for name, module in modules:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img = misc.print_module_summary(G, [z, c])
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {world_size} GPUs...')
    ddp_modules = dict()
    module_list = [('G_mapping', G.mapping), ('G_synthesis', G.synthesis), ('D', D), (None, G_ema), ('augment_pipe', augment_pipe)]
    if G.encoder is not None:
        module_list += [('G_encoder', G.encoder)]
    if disc_enable_ema:
        module_list += [('D_ema', D_ema)]
    for name, module in module_list:
        if (world_size > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(
                module, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)  # allows progressive
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules, **loss_kwargs) # subclass of training.loss.Loss

    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1, scaler=None)]
        else: # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs) # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1, scaler=None)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval, scaler=None)]
    
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
    grid_i = None
    if rank == 0:
        print(f'Exporting sample images... {batch_gpu}')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        grid_i = (torch.from_numpy(images).float() / 127.5 - 1).to(device).split(batch_gpu)

        if not os.path.exists(os.path.join(img_dir, 'reals.png')):
            save_image_grid(images, os.path.join(img_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        
        if not os.path.exists( os.path.join(img_dir, 'fakes_init.png')):
            with torch.no_grad():
                images = torch.cat([G_ema.get_final_output(z=z, c=c, noise_mode='const', img=img).cpu() for z, c, img in zip(grid_z, grid_c, grid_i)]).numpy()
            save_image_grid(images, os.path.join(img_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()

    cur_nimg = resume_start
    cur_tick = cur_nimg // (1000 * kimg_per_tick) 

    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)

    while True:
        # set number of images
        loss.set_alpha(cur_nimg)
        curr_res = loss.resolution
        
        # Estimating Cameras for the training set (optional)
        if hasattr(training_set_sampler, 'update_dataset_cameras') and \
         (cur_nimg == resume_start and resume_start > 0 and cur_tick > update_cam_prior_ticks):
            training_set_sampler.update_dataset_cameras(D.get_estimated_camera)
        
        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            def load_data(iterator):
                img, c, _ = next(iterator)
                if resize_real_img_early: 
                    img = resize_image(img, curr_res)
                img = [{'img': img} for img in (img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)]
                c = c.to(device).split(batch_gpu)
                return img, c
            
            phase_real_img, phase_real_c = load_data(training_set_iterator)
            all_gen_z   = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z   = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c   = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c   = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c   = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]
            all_gen_img = [[None for _ in range(len(phase_real_img))] for _ in range(len(phases))]
            
        # Execute training phases.
        # with torch.autograd.profiler.profile(with_stack=True, profile_memory=True) as prof:
        for phase, phase_gen_z, phase_gen_c, phase_gen_img in zip(phases, all_gen_z, all_gen_c, all_gen_img):
            if batch_idx % phase.interval != 0:
                continue
            
            if generation_with_image:
                phase_gen_img, phase_gen_c = load_data(backup_data_iterator)
                
            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            
            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_z, gen_c, fake_img) in enumerate(zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c, phase_gen_img)):
                sync = (round_idx == batch_size // (batch_gpu * world_size) - 1)
                gain = phase.interval

                losses = loss.accumulate_gradients(
                    phase=phase.name, 
                    real_img=real_img, 
                    real_c=real_c, gen_z=gen_z, 
                    gen_c=gen_c, fake_img=fake_img,
                    sync=sync, gain=gain, scaler=phase.scaler)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                if len(losses) > 0:
                    if phase.scaler is not None:
                        phase.scaler.unscale_(phase.opt)
                    all_grads = [] 
                    for param in phase.module.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                            all_grads += [torch.norm(param.grad.detach(), p=2)]
                    grad_norm = torch.stack(all_grads).norm(p=2)
                    if phase.scaler is not None:
                        phase.scaler.step(phase.opt)
                        phase.scaler.update()
                        training_stats.report(f'Scaler/{phase.name}', phase.scaler.get_scale())
                    else:
                        phase.opt.step()
                    training_stats.report(f'Gradient/{phase.name}', grad_norm)

            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))
        
        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)
            if disc_enable_ema:  # update EMA for discriminator
                for p_ema, p in zip(D_ema.parameters(), D.parameters()):
                    p_ema.copy_(p.lerp(p_ema, ema_beta))
                for b_ema, b in zip(D_ema.buffers(), D.buffers()):
                    b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))
 
        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        
        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = [f"[{run_dir}]:"]
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        if loss.alpha is not None:
            fields += [f"alpha {training_stats.report0('Progress/alpha', loss.alpha):<8.5f}"]
            fields += [f"res {training_stats.report0('Progress/res', loss.resolution):<5d}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintain_sec', maintenance_time):<5.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Update the dataset cameras?
        if hasattr(training_set_sampler, 'update_dataset_cameras') and \
          (cur_tick % update_cam_prior_ticks == 0 and cur_tick > 0):
            training_set_sampler.update_dataset_cameras(D.get_estimated_camera)

        # Save image snapshot.
        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            with torch.no_grad():
                images = torch.cat([G_ema.get_final_output(z=z, c=c, noise_mode='const', img=None).cpu() for z, c, img in zip(grid_z, grid_c, grid_i)]).numpy()
                save_image_grid(images, os.path.join(img_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

                images = torch.cat([G_ema.get_final_output(z=z, c=c, noise_mode='const', img=img, camera_mode=[0.5,0.5,0.5]).cpu() for z, c, img in zip(grid_z, grid_c, grid_i)]).numpy()
                save_image_grid(images, os.path.join(img_dir, f'fakes{cur_nimg//1000:06d}_000.png'), drange=[-1,1], grid_size=grid_size)

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            modules = [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]
            if disc_enable_ema:
                modules += [('D_ema', D_ema)]
            for name, module in modules:
                if module is not None:
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)
                # save the latest checkpoint
                shutil.copy(snapshot_pkl, os.path.join(run_dir, 'latest-network-snapshot.pkl'))
        
        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0) and (cur_tick > 1):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G_ema'],
                    dataset_kwargs=training_set_kwargs, num_gpus=world_size, rank=rank, device=device)
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
            if rank == 0:
                losses = [(key, fields[key]) for key in fields if 'Loss/' in key]
                losses = ["{}: {:.4f}".format(key[5:], loss['mean']) for key, loss in losses]
                
                print('\t'.join(losses))

        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
