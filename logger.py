import neptune.new as neptune
from neptune.new.types import File
import numpy as np


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

global_logger = Logger()
