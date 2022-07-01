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

# (neptune) fetch project
project = neptune.get_project(name="common/Pytorch-ImageSegmentation-Unet")

# (neptune) find best run
best_run_df = project.fetch_runs_table(tag="best").to_pandas()
best_run_id = best_run_df["sys/id"].values[0]

# (neptune) re-init the chosen run
base_namespace = "evaluate"
ref_run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    project="common/Pytorch-ImageSegmentation-Unet",
    tags=["evaluation"],
    source_files=None,
    monitoring_namespace=f"{base_namespace}/monitoring",
    run=best_run_id,
)

ref_run[f"{base_namespace}/validation_data_version"].track_files(
    "s3://neptune-examples/data/brain-mri-dataset/evaluation/TCGA_HT_7692_19960724/"
)
ref_run[f"{base_namespace}/validation_data_version"].download(
    destination="evaluation_data"
)

valid = BrainSegmentationDataset(
    images_dir="evaluation_data",
    subset="validation",
    random_sampling=False,
    seed=ref_run["data/preprocessing_params/seed"].fetch(),
)

device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

unet = UNet(
    in_channels=BrainSegmentationDataset.in_channels,
    out_channels=BrainSegmentationDataset.out_channels,
)
unet.to(device)

# (neptune) Download the weights from the `train` run
ref_run["training/model/model_weight"].download("evaluate_unet.pt")
ref_run.wait()

# Load the downloaded weights
state_dict = torch.load("evaluate_unet.pt", map_location=device)
unet.load_state_dict(state_dict)

loss_fn = DiceLoss()
loss = 0.0
for i in range(len(valid)):
    with torch.no_grad():
        x, y, fname = valid[i]
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        y_pred = unet(x)
        loss += loss_fn(y_pred, y).item()

# (neptune) log evaluated loss.
ref_run[f"{base_namespace}/mean_evaluation_loss"] = (loss) / len(valid)
