# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Modified in this work
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion

from utils import find_model, load_yaml_config

import argparse
from models import init_deploy, track_generator
import numpy as np

import pickle

from utils2.dataset import Track_dataset
from utils2.npy_to_pth import split_into_groups
from utils2.grid_map import get_nomorlised_y
from torch.utils.data import Dataset, DataLoader, ConcatDataset

def pad_and_group(result_array, group_size=12):
    n = result_array.shape[0]
    groups = []

    if n <= group_size:
        padding = np.full((group_size, 2), -1.0, dtype=np.float32) 
        padding[:n] = result_array  
        groups.append(padding)  

    else:
        for start in range(0, n - group_size, group_size):
            end = start + group_size
            group = result_array[start:end]
            groups.append(group)

        last_group = result_array[-group_size:] if n >= group_size else result_array
        groups.append(last_group)

    return groups  

def count_and_get_indices(array):

    mask = array != -1
    count = np.sum(mask)
    indices = np.argwhere(mask)
    
    return count, indices

def sample_tracks(config, condition, label):

    # Setup PyTorch:
    torch.manual_seed(config['sample']['seed'])
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load model:

    model = track_generator(**config['track_generator']).to(device)
    state_dict = find_model(config['sample']['track_generator_path'])
    model.load_state_dict(state_dict)
    model.eval()  # important!

    config['diffusion']['timestep_respacing'] = str(config['sample']['num_sampling_steps'])

    diffusion = create_diffusion(**config['diffusion'])

    # Labels to condition the model with (feel free to change):

    condition = condition.to(device)
    label = label.to(device, dtype=torch.int64)


    # print(fixed_start.shape, condition_.shape, label_.shape)

    # Create sampling noise:
    n = condition.shape[0]
    label_null = torch.tensor([0] * n).to(device)

    # print(condition.dtype, label.dtype, label_null.dtype)
    # assert 1==2
    z = torch.randn(n, config['track_generator']['in_channels'], config['track_generator']['input_size'], device=device)

    # z[:, :, 0] = fixed_start  

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    _condition = condition
    condition = torch.cat([condition, condition], 0)
    label = torch.cat([label, label_null], 0)

    y = (condition, label)


    model_kwargs = dict(y=y, cfg_scale=config['sample']['scale'])


    # Sample images:


    # samples = diffusion.p_sample_loop_with_fixed_points(
    #     model.forward_with_cfg, z.shape, z, fixed_start, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )


    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples_np = samples.detach().cpu().numpy()

    return samples_np, _condition.detach().cpu().numpy(), label.detach().cpu().numpy()

def samples_from_initdata(config):
    # Setup PyTorch:
    torch.manual_seed(config['sample']['seed'])

    # Load dataset:
    train_dataset = torch.load(config['train']["track_data_path"])
    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    for _, (condition_batch, pos_label_batch) in loader:
        return condition_batch, pos_label_batch


    return None
if __name__ == "__main__":



    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    args.config = f'config.yaml'

    #load config
    config = load_yaml_config(f"./{args.config}")

    # init_samples = sample_init(config)
    condition, label = samples_from_initdata(config) 

    tracks, _, _ = sample_tracks(config, condition, label)

    np.save(f'./generated_data/track_data', tracks)
    np.save(f'./generated_data/init_data', condition)
    # np.save(f'./generated_data/label_', label)