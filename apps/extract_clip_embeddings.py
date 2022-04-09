# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import clip
import numpy as np
import click

from PIL import Image
from tqdm import tqdm
from apps.dataset_tool import open_dataset

@click.command()
@click.pass_context
@click.option('--source', help='Directory or archive name for input dataset', required=True, metavar='PATH')
def extract_clip_embeddings(
    ctx: click.Context,
    source: str,
):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)

    num_files, input_iter = open_dataset(source, max_images=None)
    extracted_features = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        img_orig = preprocess(Image.fromarray(image['img']))
        img_flip = preprocess(Image.fromarray(image['img'][:,::-1]))
        img = torch.stack([img_orig, img_flip], 0).to(device)
        with torch.no_grad():
            img_features = model.encode_image(img)
        extracted_features.append(img_features.cpu().numpy())
    extracted_features = np.asarray(extracted_features)
    np.savez(source.replace('.zip', '') + '.clip_feat.npz', extracted_features)
    print('done.')


if __name__ == "__main__":
    extract_clip_embeddings() # pylint: disable=no-value-for-parameter
