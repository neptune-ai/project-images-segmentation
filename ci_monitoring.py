import os
import random

import neptune.new as neptune

# Fetch project
project = neptune.init_project(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    name="common/project-images-segmentation",
)

# Find run with "in-prod" tag
runs_table_df = project.fetch_runs_table(tag="in-prod").to_pandas()
run_id = runs_table_df["sys/id"].values[0]

# Resume run
run = neptune.init_run(
    api_token=os.getenv("NEPTUNE_API_TOKEN"),
    project="common/project-images-segmentation",
    with_id=run_id,
)

# Run monitoring logic
# ... and log metadata to the run
run["production/monitoring/loss"].log(random.random() * 100)
