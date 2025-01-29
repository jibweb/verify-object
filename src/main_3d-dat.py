import argparse
import cv2
from itertools import product
import json
import numpy as np
import os
from scipy import spatial
import sys
from tqdm import tqdm
import yaml

sys.path.append('/home/jbweibel/code/3d-dat/')  # Adapt to your 3d-dat repository location
import v4r_dataset_toolkit as v4r

from config import config
from pose.pose_refiner import RefinePose
from pose.icp import ICPRefiner
from pose.megapose import MegaposeRefiner
from pose.pose_noise_sampling import sample_noise_pose


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
    "distal_phalanx": "distal_phalanx",
    "intermediate_phalanx": "intermediate_phalanx",
    "proximal_phalanx": "proximal_phalanx",
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
    "distal_phalanx": False,
    "intermediate_phalanx": False,
    "proximal_phalanx": False,
}


NON_SYM_AXIS = {
    "MediumBottle": [0,1],
    "SmallBottle": [0,1],
    "Needle": [0,1,2],
    "NeedleCap": [0,1],
    "RedPlug": [0,1,2],
    "Canister": [0,1,2],
    "BigBottle": [0,1],
    "YellowPlug": [0,1,2],
    "WhiteClamp": [0,1,2],
    "RedClamp": [0,1,2],
}


PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}

OBJECTS_TO_OPTIMIZE_INTERNAL = {
    PROJECT_TO_INTERNAL_NAMES[name]: val
    for name, val in OBJECTS_TO_OPTIMIZE.items()
}

SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()


def transform_pts_Rt(pts, R, t):
    """Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 ndarray with a rotation matrix.
    :param t: 3x1 ndarray with a translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert pts.shape[1] == 3
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def adi(T_est, T_gt, pts):
    """Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV'12).

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :param pts: nx3 ndarray with 3D model points.
    :return: The calculated error.
    """
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from vertices in the
    # ground-truth pose to vertices in the estimated pose.
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def get_masks(scene_path, objects, rgb_paths, mask_method=None):
    mask_gts, mask_paths = [], []

    for rgb_path in rgb_paths:
        mask_gts.append([])
        mask_paths.append([])

        for obj_idx, (obj, obj_pose) in enumerate(objects):
            mask_path = f"{obj.id}__{obj_idx:03d}__{os.path.basename(rgb_path)}"
            gt_path = os.path.join(scene_path, 'masks', mask_path)
            if os.path.isfile(gt_path):
                mask_gts[-1].append(gt_path)
            else:
                raise Exception('Missing groundtruth mask {}'.format(gt_path))

            if mask_method:
                method_path = os.path.join(
                    scene_path,
                    'masks_{}'.format(mask_method),
                    mask_path)
                if os.path.isfile(method_path):
                    mask_paths[-1].append(method_path)
                else:
                    mask_paths[-1].append(None)
            else:
                mask_paths[-1].append(gt_path)

    return mask_gts, mask_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--redo', dest='redo', default=False, action='store_true')
    parser.add_argument('--cfg', type=str, default='/home/jbweibel/code/inverse_rendering/cea_data_collection/dataset/silhouette_contact.yml')
    parser.add_argument('--dataset_cfg', type=str, default='/home/jbweibel/code/inverse_rendering/cea_data_collection/dataset/config.cfg')
    parser.add_argument('--scene', type=str)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--depth_method', type=str, default='depth')
    parser.add_argument('--mask_method', type=str)
    parser.add_argument('--pose_method', type=str, default='diffrend')
    args = parser.parse_args()

    noise_rotation_degrees = [5, 20, 40]
    noise_translation_obj_pct = [0.25, 0.5, 0.75]

    cfg = config.GlobalConfig()

    if args.cfg:
        with open(args.cfg) as fp:
            params = yaml.safe_load(fp)
        cfg.from_dict(params)

    # 3D-DAT scene setup
    scene_file_reader = v4r.io.SceneFileReader.create(args.dataset_cfg)
    scenes_ids = scene_file_reader.get_scene_ids()
    print(scenes_ids)
    if args.scene is not None:
        scenes_ids = [args.scene]

    for scene_id in scenes_ids:
        scene_path = os.path.join(
            scene_file_reader.root_dir,
            scene_file_reader.scenes_dir,
            scene_id)

        if args.exp_name:
            exp_name = args.exp_name
        else:
            exp_name = "exp_{}".format(
                len(os.listdir(os.path.join(
                    scene_path,
                    'refinement_res'))))

        exp_path = os.path.join(
            scene_path,
            'refinement_res',
            exp_name)
        if not os.path.exists(exp_path):
            os.mkdir(exp_path)

        intrinsics = scene_file_reader.get_camera_info_scene(scene_id)
        tool_cam_poses = [
            pose.tf for pose in scene_file_reader.get_camera_poses(scene_id)
        ]
        cam_info = scene_file_reader.get_camera_info_scene(scene_id)
        intrinsics = cam_info.as_numpy3x3()
        objects = scene_file_reader.get_object_poses(scene_id)
        # rgb_paths = scene_file_reader.get_images_rgb_path(scene_id)
        rgb_paths = v4r.io.get_file_list(
            os.path.join(scene_path, scene_file_reader.rgb_dir), ('.png',))
        rgb_paths.sort()
        sensor_depth_paths = scene_file_reader.get_images_depth_path(scene_id)
        contact_pts_paths = v4r.io.get_file_list(
            os.path.join(scene_path, 'contact_pts'), ('.txt',))
        contact_pts_paths.sort()

        mask_gts, mask_paths = get_masks(
            scene_path,
            objects,
            rgb_paths,
            mask_method=args.mask_method)

        gt_poses_tool = [np.array(pose).reshape((4,4)) for (obj,pose) in objects]
        scene_objects = [obj.id for (obj,pose) in objects]

        height, width = cam_info.height, cam_info.width
        objects_sizes = {}

        if args.pose_method == 'diffrend':
            pose_refiner = RefinePose(
                cfg, intrinsics.copy(),
                objects_names=None,  # Load all models of the object_library in 3d-dat mode
                objects_to_optimize=OBJECTS_TO_OPTIMIZE,  # Optimize all models of the object_library in 3d-dat mode
                width=width,
                height=height,
                plane_normal=None,
                plane_pt=None,
                debug_flag=args.debug)

            # Compute object diagonal length to weigh the translation noise
            for obj_internal, obj_mesh in pose_refiner.model.renderer.meshes.items():
                print("Object sizes processing:", obj_internal)
                objects_sizes[obj_internal] = (obj_mesh.verts_packed().max(dim=0).values - obj_mesh.verts_packed().min(dim=0).values).pow(2).sum().sqrt().cpu().numpy()
        elif args.pose_method == 'icp':
            pose_refiner = ICPRefiner(
                cfg,
                intrinsics.copy(),
                width, height,
                objects_names=None,  # Load all models of the object_library in 3d-dat mode
                objects_to_optimize=OBJECTS_TO_OPTIMIZE
            )

            # Compute object diagonal length to weigh the translation noise
            for obj_internal, obj_mesh in pose_refiner.obj_pcds.items():
                bbox = obj_mesh.get_axis_aligned_bounding_box()
                objects_sizes[obj_internal] = np.sqrt(np.power(bbox.max_bound - bbox.min_bound, 2).sum())
        elif args.pose_method == 'megapose':
            pose_refiner = MegaposeRefiner(
                cfg,
                intrinsics.copy(),
                width, height,
                objects_names=None,  # Load all models of the object_library in 3d-dat mode
                objects_to_optimize=OBJECTS_TO_OPTIMIZE
            )

            # Compute object diagonal length to weigh the translation noise
            for obj_internal, obj_mesh in pose_refiner.obj_pcds.items():
                bbox = obj_mesh.get_axis_aligned_bounding_box()
                objects_sizes[obj_internal] = np.sqrt(np.power(bbox.max_bound - bbox.min_bound, 2).sum())

        for rgb_idx, rgb_path in enumerate(rgb_paths):
            if '00012.png' in rgb_path:
                continue

            depth_path = sensor_depth_paths[rgb_idx]
            depth_path = depth_path.replace("depth", args.depth_method)
            cam_tf = tool_cam_poses[rgb_idx]
            img_mask_paths = mask_paths[rgb_idx]
            img_nb = int(rgb_path.split('.')[-2].split('/')[-1])

            gt_poses = [np.dot(np.linalg.inv(cam_tf), pose).astype(np.float32) for pose in gt_poses_tool]
            rgb = cv2.imread(rgb_path)
            depth = cv2.imread(depth_path, -1)
            empty_mask = np.zeros((height, width), dtype=bool)
            ref_masks = []
            mask_validity = []
            for mask_path in img_mask_paths:
                if mask_path:
                    ref_masks.append(cv2.imread(mask_path, -1).astype(bool))
                    mask_validity.append(True)
                else:
                    ref_masks.append(empty_mask)
                    mask_validity.append(False)

            if len(contact_pts_paths):
                contact_path = contact_pts_paths[rgb_idx]
                with open(contact_path) as fp:
                    lines = fp.readlines()
                    contact_pts = [list(map(float, line.split())) + [1] for line in lines]
                contact_pts_cam = np.array([
                    np.dot(np.linalg.inv(cam_tf), pt)[:-1]
                    for pt in contact_pts]).astype(np.float32)
            else:
                contact_pts_cam = None

            #TODO: save noise profile properly
            noise_sample_path = os.path.join(
                scene_path, 'noise_samples', "{:06d}.json".format(img_nb))
            if not os.path.isfile(noise_sample_path):
                noise_samples = []
                for noise_deg, noise_trans_obj_pct in product(noise_rotation_degrees, noise_translation_obj_pct):
                    noise_sample_per_pose = []
                    for pose_idx in range(len(gt_poses)):
                        noise_rot, noise_trans = sample_noise_pose(
                            angle=noise_deg,
                            dist=noise_trans_obj_pct*objects_sizes[scene_objects[pose_idx]],
                            degrees=True)
                        noise_sample_per_pose.append((noise_rot.tolist(), noise_trans.tolist()))
                    noise_samples.append(noise_sample_per_pose)

                if not os.path.isdir(os.path.join(scene_path, 'noise_samples')):
                    os.mkdir(os.path.join(scene_path, 'noise_samples'))

                noise_sample_dict = {
                    'samples': noise_samples,
                    'noise_rotation': noise_rotation_degrees,
                    'noise_translation': (noise_trans_obj_pct*objects_sizes[scene_objects[pose_idx]]).tolist()
                }

                with open(noise_sample_path, 'w') as fp:
                    json.dump(noise_sample_dict, fp)
            else:
                with open(noise_sample_path) as fp:
                    noise_sample_dict = json.load(fp)
                    noise_samples = noise_sample_dict['samples']

            img_results = {
                'gt_poses': [p.tolist() for p in gt_poses],
                'cfg': cfg.to_dict(),
                'depth_method': args.depth_method,
                'iter_nb': [],
                'mask_validity': mask_validity,
                'mask_method': args.mask_method,
                'noise_samples': noise_samples,
                'noise_rotation': noise_sample_dict['noise_rotation'],
                'noise_translation': noise_sample_dict['noise_translation'],
                'object_sizes': {k:v.tolist() for k,v in objects_sizes.items()},
                'pose_method': args.pose_method,
                'predicted_poses': [],
                'pre_post_adi': [],
            }

            # Noise sampling and testing ------------------------------------------
            img_results_path = os.path.join(
                scene_path,
                'refinement_res',
                exp_name,
                "{:06d}__{}__{}__{}.json".format(
                    img_nb,
                    args.depth_method,
                    args.mask_method,
                    args.pose_method)
            )

            if os.path.exists(img_results_path) and not args.redo:
                print('Skipping', img_results_path)
                continue

            for noise_per_pose in noise_samples:
                # Add noise
                init_poses = []
                for pose_idx in range(len(gt_poses)):
                    noise_rot, noise_trans = noise_per_pose[pose_idx]
                    if not OBJECTS_TO_OPTIMIZE[scene_objects[pose_idx]]:
                        init_poses.append(gt_poses[pose_idx].astype(np.float32))
                    else:
                        init_pose = np.eye(4)
                        init_pose[:3,:3] = np.dot(gt_poses[pose_idx][:3,:3], np.array(noise_rot))
                        init_pose[:3,3] = gt_poses[pose_idx][:3,3] + np.array(noise_trans)
                        print(init_pose)
                        init_poses.append(init_pose.astype(np.float32))

                if args.pose_method == 'diffrend':
                    predicted_poses, ref_rgb, ref_depth, rend_masks, viz_img = pose_refiner.optimize(
                        rgb, depth, scene_objects, init_poses, ref_masks, point_contacts=contact_pts_cam)
                    img_results['iter_nb'].append(pose_refiner.last_iter)
                elif args.pose_method in ['icp', 'megapose']:
                    predicted_poses = pose_refiner.optimize(
                        rgb, depth, scene_objects, init_poses, ref_masks)

                img_results['predicted_poses'].append([p.tolist() for p in predicted_poses])
                img_results['pre_post_adi'].append([])
                #TODO save preds
                #TODO run it on the desktop

                for pose_idx in range(len(gt_poses)):
                    if not OBJECTS_TO_OPTIMIZE[scene_objects[pose_idx]]:
                        continue

                    if args.pose_method == 'diffrend':
                        pts = pose_refiner.model.scene_sampled_meshes[pose_idx].cpu().numpy()[0,:,:3]
                    elif args.pose_method in ['icp', 'megapose']:
                        pts = np.asarray(pose_refiner.obj_pcds[scene_objects[pose_idx]].points)

                    pre_adi = adi(init_poses[pose_idx], gt_poses[pose_idx], pts)
                    post_adi = adi(predicted_poses[pose_idx], gt_poses[pose_idx], pts)
                    img_results['pre_post_adi'][-1].append((pre_adi, post_adi))

                    if args.debug:
                        print("Translation error in {:.4f}".format(np.linalg.norm(init_poses[pose_idx][:3,3] - gt_poses[pose_idx][:3,3])),
                            "Rotation error in {:.1f}".format(180./3.14*np.arccos((np.trace(init_poses[pose_idx][:3,:3].T @ gt_poses[pose_idx][:3,:3]) - 1.) / 2.)))
                        print(
                            "Translation error out {:.4f}".format(np.linalg.norm(predicted_poses[pose_idx][:3,3] - gt_poses[pose_idx][:3,3])),
                            "Rotation error out {:.1f}".format(180./3.14*np.arccos((np.trace(predicted_poses[pose_idx][:3,:3].T @ gt_poses[pose_idx][:3,:3]) - 1.) / 2.)))
                        print('Pre ', pre_adi)
                        print('Post', post_adi)

                if args.debug:
                    input("Press ENTER to continue")

            if not args.debug:
                with open(img_results_path, 'w') as fp:
                    json.dump(img_results, fp)
