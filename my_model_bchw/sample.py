# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from utils import find_model, load_yaml_config

import argparse
from models import init_deploy, track_generator
import numpy as np
from utils2.grid_map import Grid, restore_original_data


def sample_init(config, n=10):
    # Setup PyTorch:
    torch.manual_seed(config['sample']['seed'])
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    # Load model:

    model = init_deploy(**config['init_deploy']).to(device)
    state_dict = find_model(config['sample']['init_deploy_path'])
    model.load_state_dict(state_dict)
    model.eval()  # important!

    config['diffusion']['timestep_respacing'] = str(config['sample']['num_sampling_steps'])

    diffusion = create_diffusion(**config['diffusion'])

    # Labels to condition the model with (feel free to change):
    # class_labels = [0] * int(n/2) + [1] * int(n/2)
    class_labels = [0] * 10000
    # class_labels = [0] * 15002


    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, config['init_deploy']['in_channels'], config['init_deploy']['input_size'][0], config['init_deploy']['input_size'][1], device=device)

    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([2] * n, device=device)
    y = torch.cat([y, y_null], 0)

    model_kwargs = dict(y=y, cfg_scale=config['sample']['scale'])


    # Sample images:

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples_np = samples.detach().cpu().numpy()
    # -----------------------data process --------------------------------------
    my_grid = Grid(1000, 1140, 5)
    data = np.where(samples_np < -0.2, -1, samples_np)
    data = np.where(data == -1, -1, np.clip(data, 0, 1))
    data = np.transpose(data, (0, 2, 3, 1))

    data = restore_original_data(data, my_grid)

    return data

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.yaml')
    args = parser.parse_args()

    #load config
    config = load_yaml_config(f"./{args.config}")


    data = sample_init(config)
    np.save(f'./generated_data/init_data', data)
    print("init_sates have been sampled")