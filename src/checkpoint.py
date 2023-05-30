from os.path import join as pjoin

import torch
from torch import nn


class CheckpointSaver:
    def __init__(self, model_name, save_dir: str = "", should_minimize: bool = True):
        self.should_minimize = should_minimize
        self.metric_val = 1e6 if should_minimize else -1
        self.model_name = model_name
        self.save_dir = save_dir
        
    def get_checkpoint(self, model,  metric_val, step: int):
        if ((self.should_minimize and metric_val < self.metric_val) or 
        (not self.should_minimize and metric_val > self.metric_val)):
            torch.save(model.state_dict(), 
                       pjoin(self.save_dir, f"{self.model_name}.pth"))
            self.metric_val = metric_val
            print(f'==> Saved checkpoint on {step}')
