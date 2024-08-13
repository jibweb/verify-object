from .simple_configuration import BaseConfig
from dataclasses import dataclass, field
from enum import Enum

# class PoseRepresentation(Enum):
#     Q = 1 # choices=['so3', 'se3', 'q'], help='q for [q, t], so3 for [so3_log(R), t] or se3 for se3_log([R, t])'


@dataclass
class LossConfig(BaseConfig):
    active : bool = True
    early_stopping_loss : float = 1.
    weight : float = 1.


@dataclass
class LossesConfig(BaseConfig):
    silhouette_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=True, early_stopping_loss=0.002))
    contour_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=False, early_stopping_loss=0.7))
    collision_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=False, early_stopping_loss=0.03))
    plane_collision_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=False, early_stopping_loss=0.03))
    depth_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=False, early_stopping_loss=0.7))
    relative_pose_loss : LossConfig = field(
        default_factory=lambda: LossConfig(active=False, early_stopping_loss=0.7))

@dataclass
class OptimizationConfig(BaseConfig):
    learning_rate: float = 0.005
    optimizer_name: str = 'adam'
    # learning_rate: float = 0.01
    # optimizer_name: str = 'LBFGS'
    max_iter: int = 30
    pose_representation : str = 'q'  # choices=['so3', 'se3', 'q'], help='q for [q, t], so3 for [so3_log(R), t] or se3 for se3_log([R, t])'
    faces_per_pixel : int = 10
    plane_refinement : bool = True
    losses : LossesConfig = field(default_factory=LossesConfig)