import os
import json
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from network import SCN
from utils.filters import Gaussian
from utils.dataset import SpineDataset
from utils.util import generate_patches, Aggregator, generate_train_mask
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(args):
    LANDMARK_NUM = args.num
    DATA_INFO = args.info
    MODEL_PATH = args.model
    SEED = args.seed
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    CKPT = args.ckpt
    DEVICE = args.device if torch.cuda.is_available() else 'cpu'
    BEST_TRAIN_LOSS = 1e5
    BEST_SCORE = 1e5

    os.makedirs("run", exist_ok=True)
    os.makedirs(f"run/{MODEL_PATH}", exist_ok=True)
    os.makedirs(f"tmp/{MODEL_PATH}", exist_ok=True)
    writer = SummaryWriter(f"tmp/{MODEL_PATH}")

    # Set Random State
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True  # make sure reproducibility
    random.seed(SEED)
    np.random.seed(SEED)

    with open(DATA_INFO, "r") as f:
        INFO = json.load(f)

    model = SCN(1, LANDMARK_NUM).to(DEVICE)
    criterion = nn.MSELoss()
    # Note that AdamW is different with Adam when weight_decay is not None
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=5e-4)
    gaussian = Gaussian(3, 7, 3, LANDMARK_NUM, norm=True).to(DEVICE)

    train_set = SpineDataset('data/2mm', INFO['train'], 'train', LANDMARK_NUM)
    val_set = SpineDataset('data/2mm', INFO['val'], 'val', LANDMARK_NUM)
    train_loader = DataLoader(train_set, BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, 1)

    NUM_BATCHES = len(train_loader)
    START = 0
    if CKPT:
        checkpoint = torch.load(CKPT)
        START = checkpoint['epoch']+1
        BEST_SCORE = checkpoint['score']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    global_step = START*NUM_BATCHES
    print("Start training")
    for ep in range(START, EPOCHS):
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.
        train_mask_loss = 0.
        for step, (ID, image, mask, landmark) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(DEVICE)
            train_mask = generate_train_mask(landmark).to(DEVICE)
            heatmap = gaussian(landmark.to(DEVICE))
            pred, HLA, HSC = model(image)
            mask_loss = criterion(pred*train_mask, heatmap)
            mask_loss.backward()
            writer.add_scalar("step/train_mask_loss", mask_loss.item(), global_step)
            optimizer.step()
            with torch.no_grad():
                loss = F.mse_loss(pred, heatmap)
            writer.add_scalar("step/train_loss", loss.item(), global_step)
            global_step += 1
            train_mask_loss += mask_loss.item()
            train_loss += loss.item()
            print(
                f"Iter:{global_step}\tID:{ID}\tmask loss:{mask_loss.item():.4E}\tloss:{loss.item():.4E}", end='\r')

        # save checkpoint every epoch
        checkpoint = {
            "epoch": ep,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "score": BEST_SCORE,
            "loss": train_loss,
        }
        torch.save(checkpoint, f"run/{MODEL_PATH}/checkpoint.pth")

        # save best checkpoint
        train_loss /= NUM_BATCHES
        train_mask_loss /= NUM_BATCHES
        writer.add_scalar("epoch/train_loss", train_loss, ep)
        writer.add_scalar("epoch/train_mask_loss", train_mask_loss, ep)
        if train_loss < BEST_TRAIN_LOSS:
            BEST_TRAIN_LOSS = train_loss
            torch.save(checkpoint, f"run/{MODEL_PATH}/best_checkpoint.pth")

        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            val_loss = 0.
            val_mask_loss = 0.
            for step, (ID, image, mask, landmark) in enumerate(val_loader):
                patches, location = generate_patches(image, mask, LANDMARK_NUM, 96)
                train_mask = generate_train_mask(landmark).to(DEVICE)
                heatmap = gaussian(landmark.to(DEVICE))
                patch_loader = DataLoader(patches, 1)
                agg = Aggregator(LANDMARK_NUM)
                for i, (new_image, new_mask, new_landmark) in enumerate(patch_loader):
                    new_image = new_image.to(DEVICE)
                    pred, _, _ = model(new_image)
                    agg.Add(pred, location[i])
                pred = agg.Execute().to(DEVICE)
                val_loss += F.mse_loss(pred, heatmap)
                val_mask_loss += F.mse_loss(pred*train_mask, heatmap)
            val_loss /= len(val_loader)
            val_mask_loss /= len(val_loader)
        writer.add_scalar("epoch/val_loss", val_loss, ep)
        writer.add_scalar("epoch/val_mask_loss", val_mask_loss)
        print(
            f"Epoch:{ep+1}/{EPOCHS}\ttrain_loss:{train_loss:.4E}\ttrain_mask_loss:{train_mask_loss:.4E}\tval_loss:{val_loss:.4E}\tval_mask_loss:{val_mask_loss:.4E}")

        if val_mask_loss < BEST_SCORE:
            BEST_SCORE = val_mask_loss
            torch.save({
                "weight": model.state_dict(),
                "epoch": ep,
                "score": BEST_SCORE
            }, f"run/{MODEL_PATH}/model.pth")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num", type=int, default=25)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--info", type=str, default="data/info.json")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--model", type=str, default='SCN_no_aug')
    args = parser.parse_args()

    main(args)
