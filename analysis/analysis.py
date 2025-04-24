import argparse
from glob import glob
import json
from PIL import Image
import numpy as np
import torch
from models.resnet import ResNet18
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

import os

from utils import create_logger

class dataset_w_path(Dataset):
    def __init__(self, path, transform):
        super().__init__()
        all_itmes = os.listdir(path)
        self.files = [
            {"name": name, 
             "label": int(name.split('.')[0].split('_')[0]), 
             "path": os.path.join(path, name)}
            for name in all_itmes
            if os.path.isfile(os.path.join(path, name))
        ]
        
        self.transform = transform

    def __len__(self): 
        return len(self.files)
    
    def __getitem__(self, index):
        return self.transform(Image.open(self.files[index]["path"])), \
                self.files[index]["label"], \
                self.files[index]["path"]
    
    
def main(args):
    ##############################################################################
    #                                   Set up                                   #
    ############################################################################## 
    assert torch.cuda.is_available(), print("At least one GPU is required.")
    device = torch.device("cuda")

    # Create result-doocumenting folder
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-predicting"  # Create an experiment folder
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create logger
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")
    
    model_path = args.model_path
    model = ResNet18()
    model.to(device).eval()
    model.load_state_dict(torch.load(model_path)["model"])

    ##############################################################################
    #                              Prepare datasets                              #
    ############################################################################## 
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,))
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = dataset_w_path(args.data_path, transform=transform)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

    ##############################################################################
    #                                 Predicting                                 #
    ############################################################################## 
    logger.info("Starting predicting.")
    hit = 0
    record = []
    pair = {} # 记录错误预测时的目标标签和预测标签
    model.eval() # IMPORTANT!
    for img, label, img_path in test_dataloader:
        img = img.to(device)
        label = torch.tensor(label).to(device)
        _, pred = torch.max(model(img), dim=-1)
        hit = hit + (pred == label).sum()

        for i, (x, y) in enumerate(zip(pred, label)):
            x = int(x.cpu())
            y = int(y.cpu())
            if x != y:
                record.append({"path": img_path[i], 
                               "target": y, 
                               "pred": x})
                if pair.get((x, y)) == None: pair[(x, y)] = 1
                else: pair[(x, y)] = pair[(x, y)] + 1

    result_path = f"{experiment_dir}/error_pred.json"
    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(record, file, indent=4)

    pair_path = f"{experiment_dir}/target_and_pred.txt"
    with open(pair_path, 'w') as f:
        for i in range(10):
            print(f'target:{i}', file=f)
            indent = "  "
            sum = 0
            for j in range(10):
                if pair.get((i, j)) == None:
                    print(f"{indent}{j}:{0}", file=f)
                else:
                    print(f"{indent}{j}:{pair[(i,j)]}", file=f)
                    sum = sum + pair[(i,j)]
            print(f"{indent}tot:{sum}", file=f)

    logger.info(f"accuracy: {hit / len(test_dataset):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
