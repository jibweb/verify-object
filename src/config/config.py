import os
# from typing import Any
# from enum import Enum
from .simple_configuration import BaseConfig
from .optimization_config import OptimizationConfig
from dataclasses import dataclass, field


PATHS_DATASET_TRACEBOT = [
    "/home/negar/Documents/Tracebot/Tracebot_Negar_2022_08_04",
    "/media/dominik/FastData/datasets/Tracebot_Negar_2022_08_04",
    "/home/jbweibel/dataset/Tracebot/Tracebot_Negar_2022_08_04"
]
try:
    PATH_DATASET_TRACEBOT = [path for path in PATHS_DATASET_TRACEBOT if os.path.exists(path)][0]
except IndexError:
    print("Tracebot dataset not available !")

PATHS_DATASET_BOP = [
    "/home/negar/Documents/Tracebot/Files/BOP_datasets/tless_test_primesense_bop19/test_primesense",
]
try:
    PATHS_DATASET_BOP = [path for path in PATHS_DATASET_BOP if os.path.exists(path)][0]
except IndexError:
    print("BOP dataset not available !")

PATH_REPO = '/'.join(os.path.dirname(__file__).split('/')[:-1])


@dataclass
class GlobalConfig(BaseConfig):
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    path_repo: str = '/'.join(os.path.dirname(__file__).split('/')[:-1])
    objects_path: str = os.path.join(PATH_DATASET_TRACEBOT, 'objects')
    mesh_num_samples : int = 500


if __name__ == "__main__":
    import yaml
    cfg = GlobalConfig()
    # import pdb; pdb.set_trace()
    cfg.optim_cfg.learning_rate = 0.01
    print(cfg.optim_cfg.learning_rate, "should be 0.01")

    cfg.from_dict({"optim_cfg": {"learning_rate": 0.02}})
    print(cfg.optim_cfg.learning_rate, "should be 0.02")
    cfg.from_dict({"optim_cfg.learning_rate": 0.03}, flat=True)
    print(cfg.optim_cfg.learning_rate, "should be 0.03")

    try:
        cfg.other_attr = 2
        print("!!! Uncaught attr creation error")
    except Exception as e:
        print("Expected error about attribute creation //", e)

    try:
        cfg.optim_cfg.mask_color = 2
        print("!!! Uncaught attr type error")
    except Exception as e:
        print("Expected error about Enum value type //", e)

    try:
        cfg.optim_cfg.learning_rate = "s"
        print("!!! Uncaught attr type error")
    except Exception as e:
        print("Expected error about value type //", e)



    print("\n", "-"*10)
    dumped_str = yaml.dump(cfg)
    print(dumped_str)
    print("\n", "-"*10)
    print(yaml.load(dumped_str, Loader=yaml.Loader))