import argparse
import json
import os
import math

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import BrainSegmentationDataset
from transform import transforms
from model_utils import UNet, DiceLoss
from utils import log_images, dsc, dsc_per_volume

import neptune.new as neptune
from neptune.new.types import File
import numpy as np
from torchviz import make_dot


def datasets(args):
    train = BrainSegmentationDataset(
        images_dir=args.images + "train",
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=args.flip_prob),
        seed=args.seed
    )
    valid = BrainSegmentationDataset(
        images_dir=args.images + "valid",
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
        seed=args.seed
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
    run = neptune.init_run(
        project="common/Pytorch-ImageSegmentation-Unet",
        tags=["training"],
        source_files="*.py",  # Upload all `py` files.
    )
    run["cli_args"] = vars(args)

    run['data/version/train'].track_files(args.s3_images_path + "train")
    run['data/version/valid'].track_files(args.s3_images_path + "valid")

    dataset_train, dataset_valid = datasets(args)

    #
    # Log meta-data related to Data and Preprocessing.
    #

    # Log Train images with segments!
    for i in range(args.vis_train_images):
        image, mask, fname = dataset_train.get_original_image(i)
        # Log Images expects Shape for Image and Mask to be (N, C, H, W)
        mask = mask.unsqueeze(0)
        outline_image = log_images(image.unsqueeze(0), mask, torch.zeros_like(mask))[0]

        if outline_image.max() > 1:
            outline_image = outline_image.astype(np.float32) / 255
        run[f'data/samples/images'].log(File.as_image(outline_image), name=fname)

    # Log Preprocessing Params
    run['data/preprocessing_params'] = {'aug_angle': args.aug_angle,
                                        'aug_scale': args.aug_scale,
                                        'image_size': args.image_size,
                                        'flip_prob': args.flip_prob,
                                        'seed': args.seed}

    loader_train, loader_valid = data_loaders(dataset_train, dataset_valid, args)

    # Choose device for training.
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    # Get Model for training
    unet = UNet(in_channels=BrainSegmentationDataset.in_channels, out_channels=BrainSegmentationDataset.out_channels)
    unet.to(device)
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)
    dsc_loss = DiceLoss()

    y = unet(image.unsqueeze(0).to(device))
    model_vis = make_dot(y.mean(), params=dict(unet.named_parameters()))
    model_vis.format = "png"
    model_vis.render("model_vis")
    run["training/model/visualization"] = neptune.types.File("model_vis.png")

    # Log training meta-data
    run['training/hyper_params'] = {'lr': args.lr,
                                 'batch_size': args.batch_size, 'epochs': args.epochs}

    best_validation_dsc = None
    loss_train = []
    loss_valid = []
    # Train Loop
    for epoch in tqdm(range(args.epochs), total=args.epochs, desc="epoch:"):
        # Train
        unet.train()
        # Iterate over data in data-loaders
        for i, data in tqdm(enumerate(loader_train),
                            desc='train',
                            total=math.floor(len(loader_train.dataset)/args.batch_size)):
            x, y_true, fnames = data
            x, y_true = x.to(device), y_true.to(device)
            assert x.max() <= 1. and y_true.max() <= 1.

            optimizer.zero_grad()
            y_pred = unet(x)
            loss = dsc_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()

            run["training/metrics/train_dice_loss"].log(loss.item())

        # Validation
        unet.eval()
        validation_pred = []
        validation_true = []
        logged_images = 0
        for i, data in tqdm(enumerate(loader_valid),
                            desc='valid',
                            total=math.floor(len(loader_valid.dataset)/args.batch_size)):
            x, y_true, fnames = data
            x, y_true = x.to(device), y_true.to(device)
            assert x.max() <= 1. and y_true.max() <= 1.

            optimizer.zero_grad()

            with torch.no_grad():
                y_pred = unet(x)
                loss = dsc_loss(y_pred, y_true)

                run["training/metrics/validation_dice_loss"].log(loss.item())

                y_pred_np = y_pred.detach().cpu().numpy()
                validation_pred.extend([y_pred_np[s]
                                       for s in range(y_pred_np.shape[0])])

                y_true_np = y_true.detach().cpu().numpy()
                validation_true.extend([y_true_np[s]
                                       for s in range(y_true_np.shape[0])])

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

                            if not (true_sum == 0 and pred_sum == 0):
                                dice_coeff = dsc(y_pred_np[i], y_true_np[i])

                                if img.max() > 1:
                                    img = img.astype(np.float32)/255
                                fname = fnames[i]
                                fname = fname.replace(".tif", "")
                                run[f"training/validation_prediction_progression/{fname}"].log(
                                    File.as_image(img), name=f"Dice: {dice_coeff}")
                                logged_images += 1

        try:
            # DSC per patient volume
            mean_dsc = np.mean(
                dsc_per_volume(
                    validation_pred,
                    validation_true,
                    loader_valid.dataset.patient_slice_index,
                )
            )
        except Exception as e:
            mean_dsc = 0.
            print(e)

        run["training/metrics/validation_dice_coefficient"].log(mean_dsc)
        if best_validation_dsc is None or mean_dsc > best_validation_dsc:
            best_validation_dsc = mean_dsc
            torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
            run["training/metrics/best_validation_dice_coefficient"] = best_validation_dsc
            run['training/model/model_weight'].upload(os.path.join(
                args.weights, "unet.pt"))

        # Sync after every epoch
        run.sync()

    # Tag as the best if `best_validation_dsc` was better than previous best
    # (neptune) fetch project
    project = neptune.get_project(name="common/Pytorch-ImageSegmentation-Unet")

    # (neptune) find best run for given data version
    best_run_df = project.fetch_runs_table(tag="best").to_pandas()
    best_run = neptune.init_run(
        project="common/Pytorch-ImageSegmentation-Unet",
        run=best_run_df["sys/id"].values[0]
    )
    prev_best = best_run["training/metrics/best_validation_dice_coefficient"].fetch()

    # check if new model is new best
    if best_validation_dsc is not None and best_validation_dsc > prev_best:
        # If yes, add the best tag
        run["sys/tags"].add("best")

        # Update prev best run.
        best_run["sys/tags"].remove("best")

        # add current model as a new version in model registry.
        model_version = neptune.init_model_version(
            model="PYTOR3-MOD",
            project="common/Pytorch-ImageSegmentation-Unet"
        )
        model_version['model_weight'].upload(os.path.join(
                args.weights, "unet.pt"))
        model_version['best_validation_dice_coefficient'] = best_validation_dsc
        model_version["valid/dataset"].track_files(args.s3_images_path + "valid")

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
        default=25,
        help="number of epochs to train (default: 25)",
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
        default=5,
        help="frequency of saving images to log file (default: 5)",
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
        "--images", type=str, default="./data/", help="folder to download images"
    )
    parser.add_argument(
        "--s3-images-path", type=str, default="s3://neptune-examples/data/brain-mri-dataset/v3/", help="s3 folder path"
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
