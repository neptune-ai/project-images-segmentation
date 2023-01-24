import os

import neptune.new as neptune
import torch

from dataset import BrainSegmentationDataset
from model_utils import DiceLoss, UNet

os.environ["NEPTUNE_PROJECT"] = "common/project-images-segmentation-update"

# (Neptune) fetch project
project = neptune.init_project()

# (Neptune) find best run
best_run_df = project.fetch_runs_table(tag="best").to_pandas()
best_run_id = best_run_df["sys/id"].values[0]

# (Neptune) re-init the chosen run
base_namespace = "evaluate"
ref_run = neptune.init_run(
    tags=["evaluation"],
    source_files=None,
    monitoring_namespace=f"{base_namespace}/monitoring",
    with_id=best_run_id,
)

ref_run[f"{base_namespace}/validation_data_version"].track_files(
    "s3://neptune-examples/data/brain-mri-dataset/evaluation/TCGA_HT_7692_19960724/"
)
ref_run[f"{base_namespace}/validation_data_version"].download(destination="evaluation_data")

valid = BrainSegmentationDataset(
    images_dir="evaluation_data",
    subset="validation",
    random_sampling=False,
    seed=ref_run["data/preprocessing_params/seed"].fetch(),
)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

unet = UNet(
    in_channels=BrainSegmentationDataset.in_channels,
    out_channels=BrainSegmentationDataset.out_channels,
)
unet.to(device)

# (Neptune) Download the weights from the `train` run
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

# (Neptune) log evaluated loss.
ref_run[f"{base_namespace}/mean_evaluation_loss"] = (loss) / len(valid)
