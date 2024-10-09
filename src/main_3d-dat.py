import argparse
import cv2
import numpy as np
import os
import sys
from tqdm import tqdm
import yaml

sys.path.append('/home/jbweibel/code/3d-dat/')  # Adapt to your 3d-dat repository location
import v4r_dataset_toolkit as v4r

from config import config
from pose_refiner import RefinePose

INTERNAL_TO_PROJECT_NAMES = {
    1: "MediumBottle",
    2: "SmallBottle",
    3: "Needle",
    4: "NeedleCap",
    5: "RedPlug",
    6: "Canister",
    7: "BigBottle",
    8: "YellowPlug",
    9: "WhiteClamp",
    10: "RedClamp",
    # 'container': 'Canister',
}

OBJECTS_TO_OPTIMIZE = {
    "MediumBottle": True,
    "SmallBottle": True,
    "Needle": True,
    "NeedleCap": True,
    "RedPlug": True,
    "Canister": True,
    "BigBottle": True,
    "YellowPlug": True,
    "WhiteClamp": True,
    "RedClamp": True,
}

PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}

OBJECTS_TO_OPTIMIZE_INTERNAL = {
    PROJECT_TO_INTERNAL_NAMES[name]: val
    for name, val in OBJECTS_TO_OPTIMIZE.items()
}

SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--cfg', type=str, default='/home/jbweibel/code/inverse_rendering/cea_data_collection/dataset/silhouette_contact.yml')
    parser.add_argument('--dataset_cfg', type=str, default='/home/jbweibel/code/inverse_rendering/cea_data_collection/dataset/config.cfg')
    args = parser.parse_args()

    cfg = config.GlobalConfig()

    if args.cfg:
        with open(args.cfg) as fp:
            params = yaml.safe_load(fp)
        cfg.from_dict(params)


    # 3D-DAT scene setup
    scene_file_reader = v4r.io.SceneFileReader.create(args.dataset_cfg)
    scenes_id = scene_file_reader.get_scene_ids()
    print(scenes_id)

    scene_id = scenes_id[0]
    # scene_id = 'medium_bottle_sanded_1'
    # scene_id = 'large_bottle_4fingers_2'
    scene_id = 'large_bottle_2fingers_1'

    intrinsics = scene_file_reader.get_camera_info_scene(scene_id)
    tool_cam_poses = [
        pose.tf for pose in scene_file_reader.get_camera_poses(scene_id)
    ]
    intrinsics = scene_file_reader.get_camera_info_scene(scene_id).as_numpy3x3()
    objects = scene_file_reader.get_object_poses(scene_id)
    sensor_depth_paths = scene_file_reader.get_images_depth_path(scene_id)
    rgb_paths = scene_file_reader.get_images_rgb_path(scene_id)
    contact_pts_paths = v4r.io.get_file_list(
        os.path.join(
            scene_file_reader.root_dir,
            scene_file_reader.scenes_dir,
            scene_id,
            'contact_pts'),
        ('.txt'))
    contact_pts_paths.sort()
    flat_mask_paths = v4r.io.get_file_list(
        os.path.join(
            scene_file_reader.root_dir,
            scene_file_reader.scenes_dir,
            scene_id,
            'masks'),
        ('.png'))
    mask_paths = [
        [mask_path
         for mask_path in flat_mask_paths
         if mask_path.split('.')[-2].split('/')[-1].split('_')[-1] == rgb_path.split('.')[-2].split('/')[-1]]
        for rgb_path in rgb_paths
    ]

    object_poses = scene_file_reader.get_object_poses(scene_id)
    init_poses_tool = [np.array(pose).reshape((4,4)) for (obj,pose) in object_poses]
    scene_objects = [PROJECT_TO_INTERNAL_NAMES[obj.id] for (obj,pose) in object_poses]

    img = cv2.imread(rgb_paths[0])
    height, width, channels = img.shape

    pose_refiner = RefinePose(
        cfg, intrinsics.copy(),
        objects_names=SUPPORTED_OBJECTS,
        objects_to_optimize=OBJECTS_TO_OPTIMIZE_INTERNAL,
        width=width,
        height=height,
        plane_normal=None,
        plane_pt=None,
        debug_flag=args.debug)

    for rgb_idx, rgb_path in enumerate(rgb_paths):
        depth_path = sensor_depth_paths[rgb_idx]
        cam_tf = tool_cam_poses[rgb_idx]
        img_mask_paths = mask_paths[rgb_idx]
        if len(contact_pts_paths):
            contact_path = contact_pts_paths[rgb_idx]
        img_nb = int(rgb_path.split('.')[-2].split('/')[-1])

        # if img_nb not in [2]:
        #     continue

        init_poses = [np.dot(np.linalg.inv(cam_tf), pose).astype(np.float32) for pose in init_poses_tool]
        rgb = cv2.imread(rgb_path)
        depth = (cv2.imread(depth_path, -1))
        masks = [cv2.imread(mask_path, -1).astype(bool)
                 for mask_path in img_mask_paths]

        if len(contact_pts_paths):
            with open(contact_path) as fp:
                lines = fp.readlines()
                contact_pts = [list(map(float, line.split())) + [1] for line in lines]
            contact_pts_cam = np.array([
                np.dot(np.linalg.inv(cam_tf), pt)[:-1]
                for pt in contact_pts]).astype(np.float32)
        else:
            contact_pts_cam = None

        # Add noise
        for pose_idx in range(len(init_poses)):
            init_poses[pose_idx][:3, 3] += [0.1, 0.1, 0.1]

        pose_refiner.optimize(rgb, depth, scene_objects,
            init_poses, masks, point_contacts=contact_pts_cam)
