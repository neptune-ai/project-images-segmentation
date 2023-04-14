# File inspired from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/train.py
# Date accessed: 23rd June, 2022

import argparse
import json
import math
import os
import uuid
import boto3

import neptune
import numpy as np
import torch
import torch.optim as optim
from neptune.types import File
from torch.utils.data import DataLoader
from torchviz import make_dot
from tqdm import tqdm

from dataset import BrainSegmentationDataset
from model_utils import DiceLoss, UNet
from transform import transforms
from utils import dsc, dsc_per_volume, log_images

# Resource object for uploading to s3
s3 = boto3.resource('s3')

# Unique ID for the generated model
unique_model_id = str(uuid.uuid4())

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

    # (Neptune) init new run
    # Set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT as environment variables
    # or pass them as arguments to `init_run`.
    # Ref: https://docs.neptune.ai/usage/best_practices/#configuring-your-credentials
    #    : https://docs.neptune.ai/api/neptune/#init_run
    run = neptune.init_run(
        tags=["training"],
        source_files="*.py",  # Upload all `py` files.
    )

    # (Neptune) log the cli args
    run["raw_cli_args"] = vars(args)

    # (Neptune) track hash of the training data.
    run["data/version/train"].track_files(f"{args.s3_images_path}train")
    run["data/version/valid"].track_files(f"{args.s3_images_path}valid")

    ##########################################
    # Get Data for training and log samples  #
    ##########################################

    dataset_train, dataset_valid = datasets(args)

    # Log Train images with segments!
    for i in range(args.vis_train_images):
        image, mask, fname = dataset_train.get_original_image(i)
        # `log_images` expects Shape for Image and Mask to be (N, C, H, W)
        mask = mask.unsqueeze(0)
        outline_image = log_images(image.unsqueeze(0), mask, torch.zeros_like(mask))[0]

        if outline_image.max() > 1:
            outline_image = outline_image.astype(np.float32) / 255
        # (Neptune) Log sample images with mask overlay
        run["data/samples/images"].append(File.as_image(outline_image), name=fname)

    # (Neptune) Log Preprocessing Params
    run["data/preprocessing_params"] = {
        "aug_angle": args.aug_angle,
        "aug_scale": args.aug_scale,
        "image_size": args.image_size,
        "flip_prob": args.flip_prob,
        "seed": args.seed,
    }

    loader_train, loader_valid = data_loaders(dataset_train, dataset_valid, args)

    # (Neptune) Log preprocessed image
    preprocessed_image, _, _ = dataset_train[0]
    preprocessed_image = log_images(preprocessed_image.unsqueeze(0))[0]
    if preprocessed_image.max() > 1:
        preprocessed_image = preprocessed_image.astype(np.float32) / 255
    run["data/samples/preprocessed_image"] = File.as_image(preprocessed_image)

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
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    dsc_loss = DiceLoss()

    # (Neptune) Log model summary
    run["training/model/summary"] = str(unet)

    # (Neptune) Log training meta-data
    run["training/hyper_params"] = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    best_validation_dsc = None
    loss_train = []
    loss_valid = []

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

            # (Neptune) Log train loss after every step
            run["training/metrics/train_dice_loss"].append(loss.item())

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

                # (Neptune) Log valid loss after every step
                run["training/metrics/validation_dice_loss"].append(loss.item())

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
                                # (Neptune) Log prediction and ground-truth on original image
                                run[f"training/validation_prediction_progression/{fname}"].append(
                                    File.as_image(img),
                                    name=f"Dice: {dice_coeff}",
                                    description=desc,
                                )
                                logged_images += 1

        try:
            # Dice Segmentation Coeff
            # DSC per patient volume
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

        run["training/metrics/validation_dice_coefficient"].append(mean_dsc)
        if best_validation_dsc is None or mean_dsc > best_validation_dsc:
            # If we have the best_validation_dsc yet, then save the weights and
            # corresponding dice coefficient
            best_validation_dsc = mean_dsc
            torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
            # (Neptune) log best_validation_dice_coefficient
            run["training/metrics/best_validation_dice_coefficient"] = best_validation_dsc

            # upload best fine-tuned weights to S3
            s3.meta.client.upload_file('./weights/unet.pt', args.model_bucket, f'models/{unique_model_id}-unet.pt')

            # (Neptune) track the best fine-tuned weights
            run["training/model/model_weight"].track_files("s3://neptune-examples/"+f'models/{unique_model_id}-unet.pt')

        # Sync after every epoch
        run.sync()

    # Tag as the best if `best_validation_dsc` was better than previous best
    # (Neptune) fetch project
    project = neptune.init_project()

    # (Neptune) find best run for given data version
    best_run_df = project.fetch_runs_table(tag="best").to_pandas()
    best_run = neptune.init_run(
        with_id=best_run_df["sys/id"].values[0],
    )
    prev_best = best_run["training/metrics/best_validation_dice_coefficient"].fetch()

    # check if new model is new best
    if best_validation_dsc is not None and best_validation_dsc > prev_best:
        # (Neptune) If yes, add the best tag.
        run["sys/tags"].add("best")

        # (Neptune) Update prev best run.
        best_run["sys/tags"].remove("best")

        # (Neptune) add current model as a new version in model registry.
        model_version = neptune.init_model_version(model="IMG-MOD")
        model_version["model_weight"].track_files("s3://neptune-examples/"+f'models/{unique_model_id}-unet.pt')
        model_version["best_validation_dice_coefficient"] = best_validation_dsc
        model_version["valid/dataset"].track_files(f"{args.s3_images_path}valid")

        # (Neptune) associate run meta to model_version.
        run_meta = {
            "id": run.get_structure()["sys"]["id"].fetch(),
            "name": run.get_structure()["sys"]["name"].fetch(),
            "url": run.get_url(),
        }
        model_version["run"] = run_meta


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
        default=10,
        help="number of epochs to train (default: 10)",
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
        default=5,
        help="number of visualization images to save in log (default: 5)",
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
        default=5,
        help="frequency of saving images to log file (default: 5)",
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
        type=float,
        default=0.5,
        help="probablilty of rotation of training image (default: 0.5)",
    )
    parser.add_argument(
        "--model-bucket",
        type=str,
        default="neptune-examples",
        help="S3 bucket to upload model weights",
    )
    args = parser.parse_args()
    main(args)
