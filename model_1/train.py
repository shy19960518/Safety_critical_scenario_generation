import os
import logging
from time import time, strftime, localtime
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from copy import deepcopy

# ------------------ Load local library ----------------------------------
from models import init_deploy

from diffusion import create_diffusion
from utils import create_logger, requires_grad, update_ema, find_model, draw_figure, show_figure, load_yaml_config
# ------------------------------------------------------------------------
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default='config.yaml')
args = parser.parse_args()


#------------------------------------ load setting ----------------------------------------------------
config = load_yaml_config(f"./{args.config}")

# ----------------------- create checkpoint and logger save path according to local time ---------------------
start_time = time()
formatted_time = strftime("%Y-%m-%d_%H-%M-%S", localtime(start_time))
os.makedirs(f'./results/{formatted_time}/log')

# ----------------------- create logger ----------------------------------------------------------------------
logger = create_logger(f'./results/{formatted_time}/log')
logger.info(f"load {args.config}")
logger.info(f"Experiment directory created at './results/f'{formatted_time}'/log'")

# ---------------------- Create model ---------------------------------------------------------------------

os.makedirs(f"./results/{formatted_time}/model")
device = torch.device("cuda")
model = init_deploy(**config['init_deploy']).to(device)
logger.info(f"Config: {config}")
ema = deepcopy(model).to(device)
requires_grad(ema, False)

# --------------------- load currenet model to continue learning ---------------------------------------------
# model_path = f"./results/2024-05-15_10-54-59/model/1000000.pt"
# state_dict = find_model(model_path)
# model.load_state_dict(state_dict)

# --------------------- create diffusion mdoel ---------------------------------------------------------------
diffusion = create_diffusion(**config['diffusion'])
opt = torch.optim.AdamW(model.parameters(), lr=float(config['train']['lr']), weight_decay=0)
logger.info(f"Model Parameters numbers: {sum(p.numel() for p in model.parameters()):,}")

# # ---------------------------------------- Setup data: -----------------------------------------------------
train_dataset = torch.load(config['train']["init_deploy_path"])

loader = DataLoader(
    train_dataset,
    batch_size=config['train']["batch_size"],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)

logger.info(f"Dataset contains {len(train_dataset):,} data")

# # ---------------------------------------- start training: -----------------------------------------------------------------

running_loss = 0
train_steps = 0
log_steps = 0
start_time = time()

num_epoch = config['train']['num_epoch']
logger.info(f"Training for {num_epoch} epochs...")

for epoch in range(num_epoch):

    logger.info(f"Beginning epoch {epoch}...")
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

        loss = loss_dict["loss"].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        update_ema(ema, model)

        # ------------------ log loss value per 50 steps --------------------------------

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1

        if train_steps % config['train']['log_per_steps'] == 0:
            # Measure training speed:
            
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)

            logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()

        # ------------------ save checkpoint per 4000 steps -------------------------------

        if train_steps % config['train']['save_checkpoint_per_steps'] == 0 and train_steps > 0:
            
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                # "opt": opt.state_dict(),
                # "args": args
            }
            checkpoint_path = f"./results/{formatted_time}/model/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)


logger.info("Done!")
