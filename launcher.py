#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import random, shlex, datetime
import os, sys, subprocess, shutil
from glob import iglob


def copy_all_python_files(
    source, snapshot_main_dir, code_snapshot_hash, recurse_dirs="fairseq"
):
    """
    Copies following files from source to destination:
        a) all *.py files at direct source location.
        b) all fairseq/*.py recursively (default); recurse through comma-separated recurse_dirs
    """
    os.makedirs(snapshot_main_dir, exist_ok=True)
    destination = os.path.join(snapshot_main_dir, code_snapshot_hash)
    assert not os.path.exists(destination), "Code snapshot: {0} alredy exists".format(
        code_snapshot_hash
    )
    os.makedirs(destination)

    def all_pys(recurse_dirs):
        yield from iglob(os.path.join(source, "*.py"))
        for d in recurse_dirs.split(","):
            yield from iglob(os.path.join(source, d, "**/*.py"), recursive=True)
            yield from iglob(os.path.join(source, d, "**/*.so"), recursive=True)
            yield from iglob(os.path.join(source, d, "**/*.yaml"), recursive=True)

    for filepath in all_pys(recurse_dirs):
        directory, filename = os.path.split(filepath)
        if directory:
            os.makedirs(os.path.join(destination, directory), exist_ok=True)
        shutil.copy2(
            os.path.join(source, filepath), os.path.join(destination, filepath)
        )
    return destination

def launch_cluster(slurm_args, model_args):
    # prepare
    jobname = slurm_args.get('job-name', 'test')
    if slurm_args.get('workplace') is not None:
        os.makedirs(slurm_args.get('workplace'), exist_ok=True)
    if slurm_args.get('workplace') is not None:
        train_log = os.path.join(slurm_args['workplace'], 'train.%A.out')
        train_stderr = os.path.join(slurm_args['workplace'], 'train.%A.stderr.%j')
    else:
        train_log = train_stderr = None
    nodes, gpus = slurm_args.get('nodes', 1), slurm_args.get('gpus', 8)
    if not slurm_args.get('local', False):
        assert (train_log is not None) and (train_stderr is not None)
    # parse slurm

    destination = ""
    # if slurm_args.get('workplace', None) is not None:
    #     # Currently hash is just the current time in ISO format.
    #     # Remove colons since they cannot be escaped in POSIX PATH env vars.
    #     code_snapshot_hash = datetime.datetime.now().isoformat().replace(":", "_")
    #     destination = copy_all_python_files(
    #         ".",
    #         os.path.join(slurm_args['workplace'], "slurm_snapshot_code"),
    #         code_snapshot_hash,
    #         'fairseq',
    #     )
    #     os.environ["PYTHONPATH"] = destination + ":" + os.environ.get("PYTHONPATH", "")
    #     print('creat snapshot at {}'.format(destination))

    train_cmd = ['python', os.path.join(destination, 'run_train.py'), ]
    train_cmd.extend([f'gpus={nodes * gpus}'])
    train_cmd.extend([f'port={get_random_port()}'])
    train_cmd += model_args

    base_srun_cmd = [
            'srun',
            '--job-name', jobname,
            '--output', train_log,
            '--error', train_stderr,
            '--open-mode', 'append',
            '--unbuffered',
        ]
    srun_cmd = base_srun_cmd + train_cmd
    srun_cmd_str = ' '.join(map(shlex.quote, srun_cmd)) 
    srun_cmd_str = srun_cmd_str + ' &'

    sbatch_cmd = [
                'sbatch',
                '--job-name', jobname,
                '--partition', slurm_args.get('partition', 'learnfair'),
                '--gres', 'gpu:volta:{}'.format(gpus),
                '--nodes', str(nodes),
                '--ntasks-per-node', '1',
                '--cpus-per-task', '20',
                '--output', train_log,
                '--error', train_stderr,
                '--open-mode', 'append',
                '--signal', 'B:USR1@180',
                '--time', slurm_args.get('time', '4320'),
                '--mem', slurm_args.get('mem', '500gb'),
                '--exclusive',
                '--exclude', 'learnfair5035,learnfair5289,learnfair5088,learnfair5028,learnfair5032,learnfair5033,learnfair5056,learnfair5098,learnfair5122,learnfair5124,learnfair5156,learnfair5036,learnfair5258,learnfair5205,learnfair5201,learnfair5240,learnfair5087,learnfair5119,learnfair5246,learnfair7474,learnfair7585,learnfair5150,learnfair5166,learnfair5215,learnfair5142,learnfair5070,learnfair5236,learnfair7523'
            ]
    if 'constraint' in slurm_args:
        sbatch_cmd += ['-C', slurm_args.get('constraint')]
    if 'comment' in slurm_args:
        sbatch_cmd += ['--comment', slurm_args.get('comment')]
    
    wrapped_cmd = requeue_support() + '\n' + srun_cmd_str + ' \n wait $! \n sleep 610 & \n wait $!'
    sbatch_cmd += ['--wrap', wrapped_cmd]
    sbatch_cmd_str = ' '.join(map(shlex.quote, sbatch_cmd))
    
    # start training
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '2'
    env['NCCL_SOCKET_IFNAME'] = ''
    
    if env.get('SLURM_ARGS', None) is not None:
        del env['SLURM_ARGS']

    if nodes > 1:
        env['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
        env['NCCL_DEBUG'] = 'INFO'

    if slurm_args.get('dry-run', False):
        print(sbatch_cmd_str)
    
    elif slurm_args.get('local', False):
        assert nodes == 1, 'distributed training cannot be combined with local' 
        if 'CUDA_VISIBLE_DEVICES' not in env:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpus)))
        env['NCCL_DEBUG'] = 'INFO'
        
        if train_log is not None:
            train_proc = subprocess.Popen(train_cmd, env=env, stdout=subprocess.PIPE)
            tee_proc = subprocess.Popen(['tee', '-a', train_log], stdin=train_proc.stdout)
            train_proc.stdout.close()
            train_proc.wait()
            tee_proc.wait()
        else:
            train_proc = subprocess.Popen(train_cmd, env=env)
            train_proc.wait()
    else:
        with open(train_log, 'a') as train_log_h:
            print(f'running command: {sbatch_cmd_str}\n')
            with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
                stdout = train_proc.stdout.read().decode('utf-8')
                print(stdout, file=train_log_h)
                try:
                    job_id = int(stdout.rstrip().split()[-1])
                    return job_id
                except IndexError:
                    return None


def launch(slurm_args, model_args):
    job_id = launch_cluster(slurm_args, model_args)
    if job_id is not None:
        print('Launched {}'.format(job_id))
    else:
        print('Failed.')


def requeue_support():
    return """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """


def get_random_port():
    old_state = random.getstate()
    random.seed()
    port = random.randint(10000, 20000)
    random.setstate(old_state)
    return port
