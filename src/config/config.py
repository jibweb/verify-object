import os
# from typing import Any
# from enum import Enum
from .simple_configuration import BaseConfig
from .optimization_config import OptimizationConfig
from dataclasses import dataclass, field


PATH_DATASET_TRACEBOT = "/data/"


@dataclass
class GlobalConfig(BaseConfig):
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    resolution : int = 640
    path_repo: str = '/'.join(os.path.dirname(__file__).split('/')[:-1])
    objects_path: str = os.path.join(PATH_DATASET_TRACEBOT, 'models')
    debug_path: str = os.path.join(PATH_DATASET_TRACEBOT, 'debug')
    mesh_num_samples : int = 500
    mask_grabcut_refinement : bool = True
    iou_early_stopping : float = 0.9
    viz_blend : float = 0.4

