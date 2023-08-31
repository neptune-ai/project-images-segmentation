import random

import neptune

# Fetch project
project = neptune.init_project(
    project="common/project-images-segmentation",
)

# Find run with "in-prod" tag
runs_table_df = project.fetch_runs_table(tag="in-prod").to_pandas()
run_id = runs_table_df["sys/id"].values[0]

# Resume run
run = neptune.init_run(
    project="common/project-images-segmentation",
    with_id=run_id,
)

# Run monitoring logic
# ... and log metadata to the run
run["production/monitoring/loss"].append(random.random() * 100)
