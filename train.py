import argparse
import json
import os
import math

from contexttimer import Timer

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset as Dataset
from transform import transforms
from model_utils import UNet, DiceLoss
from utils import log_images, dsc, dsc_per_volume

import neptune.new as neptune
from neptune.new.types import File
import numpy as np

# Logger Class which holds the
# Neptune run.
class Logger(object):

    def __init__(self):
        self.run = neptune.init(
            project="common/Pytorch-ImageSegmentation-Unet",
            # Ideally set the Environment Variable!
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI4NTMwZGE1ZC02N2U5LTQxYjUtYTMxOC0zMGUyYTJkZTdhZDUifQ==",
        )

    def log_training_scalar(self, key, value):
        self.run["train/"+key].log(value)

    def log_finetuning_scalar(self, key, value):
        self.run["finetune/"+key].log(value)

    def log_scalar(self, key: str, value):
        self.run[key].log(value)

    def upload_image_list(self, tag, images, step, start_val=0):
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images, start=start_val):
            if img.max() > 1:
                img = img.astype(np.float32)/255

            self.run[f"{tag}_{step}/{i}.png"].upload(File.as_image(img))

def main(args):
    logger = Logger()
    logger.run["args"] = vars(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    loss_train = []
    loss_valid = []

    step = 0

    for epoch in tqdm(range(args.epochs), total=args.epochs, desc="epoch:"):
        for phase in ["train", "valid"]:
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in tqdm(enumerate(loaders[phase]),
                                desc=phase,
                                total=math.floor(len(loaders[phase].dataset)/args.batch_size)):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1) and False:
                            if i * args.batch_size < args.vis_images:
                                tag = "val_step"
                                num_images = args.vis_images - i * args.batch_size
                                logger.upload_image_list(
                                    tag,
                                    log_images(x, y_true, y_true)[:num_images],
                                    epoch,
                                    i * args.batch_size
                                )

                    if phase == "train":
                        logger.log_training_scalar("train_loss", loss.item())
                        loss.backward()
                        optimizer.step()

            if phase == "valid":
                logger.log_training_scalar("valid_loss", loss.item())
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                logger.log_training_scalar("val_dsc", mean_dsc)
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
        seed=args.seed
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
        seed=args.seed
    )
    return train, valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs to train (default: 5)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=1,
        help="frequency of saving images to log file (default: 1)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)
