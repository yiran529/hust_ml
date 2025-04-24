from glob import glob
import os
from torchvision import transforms, datasets
from models.resnet import ResNet18, ResNet34
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.nn as nn
import logging
from tqdm import tqdm
from utils import create_logger

def main(args):
    ##############################################################################
    #                                   Set up                                   #
    ############################################################################## 
    assert torch.cuda.is_available(), print("At least one GPU is required.")
    device = torch.device("cuda")

    # Create result-doocumenting folder
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-traning"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create logger and tensorboard loader
    writer = SummaryWriter(experiment_dir)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    ##############################################################################
    #                              Prepare datasets                              #
    ############################################################################## 
    train_transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),  # more data augmentation
        # transforms.ColorJitter(brightness=0.2, contrast=0.2), # more data augmentation
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.FashionMNIST(
        root='./data',          
        train=True,             
        download=True,          
        transform=train_transform     
    )

    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,             
        download=True,
        transform=test_transform
    )

    train_size = int(len(train_dataset) * args.split_rate)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )
    logger.info("Dataset is ready.")

    model = ResNet34().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epoch) # to squeeze performance
    criterion = nn.CrossEntropyLoss() # input: predicted results and targets

    ##############################################################################
    #                                 Training                                   #
    ############################################################################## 
    logger.info("Start training.")

    step = 0
    model.train()
    for epoch in range(args.num_epoch):
        logger.info(f"epoch {epoch}")
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)
            pred = model(img) 

            optim.zero_grad()
            loss = criterion(pred, label)
            loss.backward()
            optim.step()
            avg_loss = loss.item()

            step = step + 1

            if step % args.log_every == 0:
                logger.info(f"loss: {avg_loss:.4f}")
                writer.add_scalar('loss', avg_loss, step)

        hit = 0
        with torch.no_grad():
            for img, label in val_dataloader:
                img = img.to(device)
                label = label.to(device)
                _, pred = torch.max(model(img), dim=-1)
                hit = hit + (label == pred).sum()
        acc = hit / len(val_dataset)
        logger.info(f"validation accuracy: {acc:.4f}")
        writer.add_scalar('accuracy', acc, epoch)

        scheduler.step()
        
    logger.info("Training done!")

    ##############################################################################
    #                               Evaluation                                   #
    ############################################################################## 
    hit = 0
    model.eval() # IMPORTANT!
    for img, label in test_dataloader:
        img = img.to(device)
        label = label.to(device)
        _, pred = torch.max(model(img), dim=-1)
        hit = hit + (pred == label).sum()
    logger.info(f"Testing dataset accuracy: {hit / len(test_dataset):.4f}")

    checkpoint = {
        "model": model.state_dict(),
        "opt": optim.state_dict(),
        "args": args
    }
    checkpoint_path = f"{checkpoint_dir}/model.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--split-rate", type=float, default=0.8)
    parser.add_argument("--num-epoch", type=int, default=100)
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=28)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()
    main(args)
