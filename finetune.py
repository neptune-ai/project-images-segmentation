# File inspired from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/train.py
# Date accessed: 23rd June, 2022

import argparse
import math
import os

import neptune
import numpy as np
import torch
import torch.optim as optim
from neptune.types import File
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset import BrainSegmentationDataset
from model_utils import DiceLoss, UNet
from transform import transforms
from utils import dsc, dsc_per_volume, log_images


def datasets(args):
    train = BrainSegmentationDataset(
        images_dir=f"{args.images}train",
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=args.flip_prob),
        seed=args.seed,
    )
    valid = BrainSegmentationDataset(
        images_dir=f"{args.images}valid",
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
        seed=args.seed,
    )
    return train, valid


def data_loaders(dataset_train, dataset_valid, args):
    def worker_init(worker_id):
        np.random.seed(args.seed + worker_id)

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


def main(args):
    torch.manual_seed(args.seed)

    ##########################################
    # Fetch Previous Best Run for Finetuning #
    ##########################################

    # (neptune) fetch project
    project = neptune.init_project(project="common/project-images-segmentation")

    # (neptune) find best run
    best_run_df = project.fetch_runs_table(tag="best").to_pandas()
    best_run_id = best_run_df["sys/id"].values[0]

    # (neptune) re-init the chosen run
    base_namespace = "finetuning"
    ref_run = neptune.init_run(
        project="common/project-images-segmentation",
        tags=["finetuning"],
        source_files=None,
        monitoring_namespace=f"{base_namespace}/monitoring",
        with_id=best_run_id,
    )
    # (neptune) log cli args
    ref_run["finetuning/raw_cli_args"] = vars(args)

    # (neptune) Track Finetuning data
    ref_run["finetuning/data/version/train"].track_files(f"{args.s3_images_path}train")
    ref_run["finetuning/data/version/valid"].track_files(f"{args.s3_images_path}valid")

    ##########################################
    # Get Data for training and log samples  #
    ##########################################

    # Load Data
    dataset_train, dataset_valid = datasets(args)

    # Log Train images with mask overlay!
    for i in range(args.vis_train_images):
        image, mask, fname = dataset_train.get_original_image(i)
        # `log_images`` expects Shape for Image and Mask to be (N, C, H, W)
        mask = mask.unsqueeze(0)
        outline_image = log_images(image.unsqueeze(0), mask, torch.zeros_like(mask))[0]

        if outline_image.max() > 1:
            outline_image = outline_image.astype(np.float32) / 255
        # (neptune) Log sample images with mask overlay
        ref_run["finetune/data/samples/images"].append(File.as_image(outline_image), name=fname)

    # (neptune) Log Preprocessing Params
    ref_run["finetune/data/preprocessing_params"] = {
        "aug_angle": args.aug_angle,
        "aug_scale": args.aug_scale,
        "image_size": args.image_size,
        "flip_prob": args.flip_prob,
        "seed": args.seed,
    }

    loader_train, loader_valid = data_loaders(dataset_train, dataset_valid, args)

    ##########################
    # Get Model for training #
    ##########################

    # Choose device for training.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    unet = UNet(
        in_channels=BrainSegmentationDataset.in_channels,
        out_channels=BrainSegmentationDataset.out_channels,
    )
    unet.to(device)

    # (neptune) Download the weights from the `train` run
    ref_run["training/model/model_weight"].download("best_unet.pt")
    ref_run.wait()

    # Load the downloaded weights
    state_dict = torch.load("best_unet.pt", map_location=device)
    unet.load_state_dict(state_dict)

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    dsc_loss = DiceLoss()

    # (neptune) Log training hyper params
    ref_run["finetuning/hyper_params"] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    best_validation_dsc = None

    ##############
    # Train Loop #
    ##############

    for epoch in tqdm(range(args.epochs), total=args.epochs, desc="epoch:"):
        ###############
        # Train Phase #
        ###############
        unet.train()
        # Iterate over data in data-loaders
        for i, data in tqdm(
            enumerate(loader_train),
            desc="train",
            total=math.floor(len(loader_train.dataset) / args.batch_size),
        ):
            x, y_true, fnames = data
            x, y_true = x.to(device), y_true.to(device)
            assert x.max() <= 1.0 and y_true.max() <= 1.0

            optimizer.zero_grad()
            y_pred = unet(x)
            loss = dsc_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()

            # (neptune) Log train loss to finetune namespace
            ref_run["finetuning/metrics/train_dice_loss"].append(loss.item())

        ####################
        # Validation Phase #
        ####################
        unet.eval()
        validation_pred = []
        validation_true = []
        logged_images = 0
        for i, data in tqdm(
            enumerate(loader_valid),
            desc="valid",
            total=math.floor(len(loader_valid.dataset) / args.batch_size),
        ):
            x, y_true, fnames = data
            x, y_true = x.to(device), y_true.to(device)
            assert x.max() <= 1.0 and y_true.max() <= 1.0

            optimizer.zero_grad()

            with torch.no_grad():
                y_pred = unet(x)
                loss = dsc_loss(y_pred, y_true)

                # (neptune) Log validation lsos to finetune namespace
                ref_run["finetuning/metrics/validation_dice_loss"].append(loss.item())

                y_pred_np = y_pred.detach().cpu().numpy()
                validation_pred.extend([y_pred_np[s] for s in range(y_pred_np.shape[0])])

                y_true_np = y_true.detach().cpu().numpy()
                validation_true.extend([y_true_np[s] for s in range(y_true_np.shape[0])])

                if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                    # If current `epoch` is a multiple of `vis_freq`.
                    num_images = args.vis_images - logged_images
                    images = log_images(x, y_true, y_pred)[:num_images]

                    for i, img in enumerate(images):
                        if logged_images < args.vis_images:
                            # Log only the images which
                            # 1. Have false positives
                            # 2. Or have some mask in the ground truth.
                            true_sum = y_true[i].sum()
                            pred_sum = y_pred[i].round().sum()
                            if true_sum != 0 or pred_sum != 0:
                                dice_coeff = dsc(y_pred_np[i], y_true_np[i])

                                if img.max() > 1:
                                    img = img.astype(np.float32) / 255
                                fname = fnames[i]
                                fname = fname.replace(".tif", "")
                                img_no = fname[fname.rfind("_") + 1 :]
                                patient_name = fname[: fname.rfind("_")]
                                desc = (
                                    f"Epoch: {epoch}\nPatient: {patient_name}\nImage No: {img_no}"
                                )
                                # (neptune) Log prediction and ground-truth on original image
                                ref_run[
                                    f"finetuning/validation_prediction_progression/{fname}"
                                ].append(
                                    File.as_image(img),
                                    name=f"Dice: {dice_coeff}",
                                    description=desc,
                                )
                                logged_images += 1

        # Get mean dice segmentation coeff
        # per patient volume on validation set
        try:
            mean_dsc = np.mean(
                dsc_per_volume(
                    validation_pred,
                    validation_true,
                    loader_valid.dataset.patient_slice_index,
                )
            )
        except Exception as e:
            mean_dsc = 0.0
            print(e)

        ref_run["finetuning/metrics/validation_dice_coefficient"].append(mean_dsc)

        if best_validation_dsc is None or mean_dsc > best_validation_dsc:
            best_validation_dsc = mean_dsc
            # (neptune) log best_validation_dice_coefficient
            ref_run["finetuning/metrics/best_validation_dice_coefficient"] = best_validation_dsc
            torch.save(unet.state_dict(), os.path.join(args.weights, "finetune_unet.pt"))
            # (neptune) upload best fine-tuned weights
            ref_run["finetuning/model/model_weight"].upload(
                os.path.join(args.weights, "finetune_unet.pt")
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="number of epochs to train (default: 2)",
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
        default=7,
        help="number of visualization images to save in log (default: 7)",
    )
    parser.add_argument(
        "--vis-train-images",
        type=int,
        default=10,
        help="number of train visualization images to save in log (default: 10)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=1,
        help="frequency of saving images to log file (default: 1)",
    )
    parser.add_argument("--weights", type=str, default="./weights", help="folder to save weights")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--logs", type=str, default="./logs", help="folder to save logs")
    parser.add_argument("--images", type=str, default="./data/", help="folder to download images")
    parser.add_argument(
        "--s3-images-path",
        type=str,
        default="s3://neptune-examples/data/brain-mri-dataset/v3/",
        help="s3 folder path",
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
    parser.add_argument(
        "--flip-prob",
        type=int,
        default=0.5,
        help="probablilty of rotation of training image (default: 0.5)",
    )
    args = parser.parse_args()
    main(args)
