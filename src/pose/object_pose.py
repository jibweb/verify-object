import json
import os
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
import numpy as np
from tqdm import tqdm
import trimesh
import yaml
from scipy.spatial.transform.rotation import Rotation
import imageio
from PIL import Image
# from src.contour.contour import single_image_edge, img_contour_sdf, imshow
from contour.contour import compute_sdf_image
from utility.visualization import visualization_progress
# from torchvision.transforms import GaussianBlur
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
import matplotlib.pyplot as plt

import torchvision
import kornia as K


# --------- PREPARE OUR OWN DATA
def tq_to_m(tq):
    # tq = [tx, ty, tz, qx, qy, qz, qw]
    m = np.eye(4, dtype=np.float32)
    m[:3, 3] = tq[:3]
    m[:3, :3] = Rotation.from_quat(tq[3:]).as_matrix()
    return m


def add_noise(t_mag, gt_pose_list, r_error_deg, t_error_list, axes_rotation):
    """
    Adding noise to the ground truth poses
    :param axes_rotation: Specifies sequence of axes for rotations. Up to 3 characters
                    belonging to the set {'X', 'Y', 'Z'} for intrinsic rotations, or
                    {'x', 'y', 'z'} for extrinsic rotations.
    :param gt_pose_list: list of ground truth pose of the objects' single scene
    :param r_error_deg: float or array_like, shape (N,) or (N, [1 or 2 or 3])
                    Euler angles
    :param t_error_list: transition error list
    :return: Noisy poses list
    """
    # initialization error: added on top of GT pose
    r_error = Rotation.from_euler(axes_rotation, r_error_deg, degrees=True).as_matrix().astype(np.float32)
    t_error = np.asarray(t_error_list) * float(t_mag / 10)

    new_gt_poses_list = []

    for obj_num in range(len(gt_pose_list)):
        gt_pose = gt_pose_list[obj_num]
        gt_pose[0, :3, :3] = torch.from_numpy(r_error).to(device) @ gt_pose[0, :3, :3]
        gt_pose[0, :3, 3] += torch.from_numpy(t_error).to(device)
        new_gt_poses_list.append(gt_pose)

    return new_gt_poses_list


def get_gt_pose_camera_frame(object_pose_file_path, camera_pose_file_path):
    """
    Convert poses from world to camera coordinate system for multiple objects
    :param object_pose_file_path: path to the object pose yaml fil
    :param camera_pose_file_path: path to the camera pose yaml file
    :return:
    """

    # TODO : How to make it computationally better?
    # object pose (world space): model -> world
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    # List to store the object poses
    objects_gt_poses = []
    # List of objects ids
    objects_id = []
    # Reading the id of each object and calculating the pose of individual objecst
    for object_dic in objects_poses_dic:
        objects_id.append(object_dic['id'])
        obj_pose = np.asarray(object_dic['pose']).reshape(4, 4).astype(np.float32)
        # camera pose: camera -> world
        cam_poses = [tq_to_m([float(v) for v in line.split(" ")[1:]])
                     for line in open(camera_pose_file_path).readlines()]
        # object pose (camera space): model -> world -> camera
        gt_poses = [np.linalg.inv(cam_pose) @ obj_pose for cam_pose in cam_poses]
        objects_gt_poses.append(gt_poses)

    return objects_gt_poses, objects_id


def get_gt_pose_camera_frame_bop(object_pose_file_path, camera_pose_file_path):
    """
    Convert poses from world to camera coordinate system for multiple objects
    :param object_pose_file_path: path to the object pose yaml fil
    :param camera_pose_file_path: path to the camera pose yaml file
    :return:
    """

    # TODO : How to make it computationally better?
    # object pose (world space): model -> world
    objects_poses_dic = yaml.load(open(object_pose_file_path, 'r'), Loader=yaml.FullLoader)
    # List to store the object poses
    objects_gt_poses_list = []  # shape: (num_images, num_obj_per_image)
    # List of objects ids
    objects_id = []
    image_ids = []
    f = open(camera_pose_file_path)
    camera_poses_dic = json.load(f)
    # Reading the id of each object and calculating the pose of individual objecst
    for image_id in objects_poses_dic:
        objects_gt_poses_image = []  # to store objects per image
        image_ids.append(image_id)
        # camera pose: camera -> world
        camera = camera_poses_dic[image_id]
        camera_pose = np.zeros((4, 4)).astype(np.float32)
        camera_pose[:3, :3] = np.asarray(camera['cam_R_w2c']).reshape((3, 3))
        camera_pose[:3, 3] = np.asarray(camera['cam_t_w2c'])
        camera_pose[-1, -1] = 1
        for obj in objects_poses_dic[image_id]:
            objects_id.append(obj['obj_id'])
            obj_pose = np.zeros((4, 4)).astype(np.float32)
            obj_pose[:3, :3] = np.asarray(obj['cam_R_m2c']).reshape((3, 3))
            obj_pose[:3, 3] = np.asarray(obj['cam_t_m2c'])
            obj_pose[-1, -1] = 1
            # object pose (camera space): model -> world -> camera
            objects_gt_poses_image.append(np.linalg.inv(camera_pose) @ obj_pose)
            # objects_gt_poses_image.append(obj_pose)

        objects_gt_poses_list.append(objects_gt_poses_image)

    return objects_gt_poses_list, np.asarray(objects_id), image_ids


def load_objects_models(object_names, objects_path, cmap=plt.cm.tab20(range(20)), mesh_num_samples=500, scale=1000):
    meshes = {}
    sampled_down_meshes = {}

    for oi, object_name in enumerate(object_names):
        # Load mesh
        verts, faces_idx, _ = load_obj(os.path.join(objects_path, f'{object_name}.obj'))
        textures = TexturesVertex(
            verts_features=torch.from_numpy(cmap[oi][:3])[None, None, :]
                                    .expand(-1, verts.shape[0], -1).type_as(verts)
        )
        mesh = Meshes(
            verts=[verts/scale],
            faces=[faces_idx.verts_idx],
            textures=textures
        )

        meshes[oi+1] = mesh

        # Create a randomly point-normal set
        # Same number of points for each individual object
        mesh_sampled_down = trimesh.load(os.path.join(objects_path, f'{object_name}.obj'))
        norms = mesh_sampled_down.face_normals
        samples = trimesh.sample.sample_surface_even(mesh_sampled_down, mesh_num_samples) # either exactly NUM_samples, or <= NUM_SAMPLES --> pad by random.choice
        samples_norms = norms[samples[1]] # Norms pointing out of the object
        samples_point_norm = np.concatenate((np.asarray(samples[0]/scale), np.asarray(0-samples_norms)), axis=1)
        if samples_point_norm.shape[0] < mesh_num_samples:  # NUM_SAMPLES not equal to mesh_num_samples -> padding
            idx = np.random.choice(samples_point_norm.shape[0], mesh_num_samples - samples_point_norm.shape[0])
            samples_point_norm = np.concatenate((samples_point_norm, samples_point_norm[idx]), axis=0)

        sampled_down_meshes[oi+1] = torch.from_numpy(samples_point_norm.astype(np.float32))[None, ...]

    return meshes, sampled_down_meshes


def scene_optimization(logger, t_mag, isbop, scene_path, mask_path, scene_number, im_id, model, T_gt_list, T_igt_list,
                       rotation_noise_degree_list, rotation_axes_list, trans_noise_list, scale, max_num_iterations,
                       early_stopping_loss, lr, optimizer_type, img_debug_name, debug_flag):
    # For visualization
    iter_values = {"r": [], "t": [], "loss": [], "image_id": im_id}
    # Get reference images
    reference_rgb = imageio.imread(os.path.join(scene_path, f"rgb/{im_id:06d}.png"))
    # Depending on how many objects in the scene, sum all of them up together
    reference_mask = np.zeros((reference_rgb.shape[0], reference_rgb.shape[1]), dtype=np.float32)
    for num_obj in range(len(T_gt_list)):
        # TODO make sure that mask for num_obj corresponds to pose in T_gt_list and the ith model in the scene
        if isbop:
            obj_mask = imageio.imread(os.path.join(mask_path, f"{scene_number}/mask/{im_id:06d}_{num_obj:06d}.png"))
        else:
            obj_mask = imageio.imread(os.path.join(mask_path,
                                                   f"{scene_number}/masks/003_{model.meshes_name[num_obj]}_00{num_obj}_{im_id:06d}.png"))
        # Different objects have different pixel values
        reference_mask[obj_mask > 0] = num_obj+1
    if scale != 1:
        reference_height, reference_width = reference_rgb.shape[:2]
        # Create colored instance mask
        reference_mask = reference_mask[..., None].repeat(3, axis=-1)

        # Giving each individual object, different color in the mask image
        for oi in range(1, 3):
            cmap = plt.cm.tab20(range(20))
            color = np.array(cmap[oi-1][:3])[None, None, :].repeat(reference_height, axis=0).repeat(reference_width, axis=1)
            reference_mask[reference_mask == oi] = color[reference_mask == oi]

        from skimage.transform import resize
        reference_width //= scale
        reference_height //= scale
        reference_rgb = resize(reference_rgb[..., :3], (reference_height, reference_width))
        reference_mask = resize(reference_mask, (reference_height, reference_width))

    # If the loss num = 6, calculate the 2D SDF
    if model.loss_func_num == 6:
        # Calculating the sdf image for the contour based loss
        if os.path.exists(f"result/sdf_images/{scene_number}/sdf_image_{im_id}.pt"):
            sdf_image = torch.load(f"result/sdf_images/{scene_number}/sdf_image_{im_id}.pt")
        else:
            sdf_image = compute_sdf_image(reference_rgb, reference_mask)
        ref_gray_tensor = sdf_image


    # Add artificial error to get initial pose estimate
    # model.trash = T_gt_list
    T_init_list = [T_gt.clone() for T_gt in T_gt_list]
    T_init_list = add_noise(t_mag, T_init_list, rotation_noise_degree_list, trans_noise_list, rotation_axes_list) # Return the list of T_init with noises


    best_metrics, iter_values = optimization_step(
        model, reference_rgb, reference_mask, T_init_list, T_igt_list, ref_gray_tensor,
        optimizer_type, max_num_iterations, early_stopping_loss, lr,
        logger, im_id, debug_flag, img_debug_name,
        isbop)


    return best_metrics, iter_values


def optimization_step(model, reference_rgb, reference_depth, reference_masks,
                      optim_cfg, T_igt_list, debug_flag, debug_path, isbop, viz_blend=0.4):
    # For visualization
    # iter_values = {"r": [], "t": [], "loss": [], "image_id": im_id}
    iter_values = {"r": [], "t": [], "loss": []}

    # If the loss num = 6, calculate the 2D SDF
    if optim_cfg.losses.contour_loss.active:
        # Calculating the sdf image for the contour based loss
        sdf_image = compute_sdf_image(reference_rgb, reference_masks)
    else:
        sdf_image = None

    if T_igt_list:
        metrics, metrics_str = model.evaluate_progress(
            [T_igt.transpose(-2, -1) for T_igt in T_igt_list],
            isbop=isbop)
    else:
        metrics, metrics_str = {"ADI": 1, "ADD": 1}, "No grountruth available for metrics"

    # Optimization
    scene_early_stopping_loss = len(reference_masks) * sum(
        [v.weight * v.early_stopping_loss for k, v in optim_cfg.losses.__dict__.items() if v.active]
    )

    # Prepare events, recording time
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Record start
    start.record()

    optimizer_type = {
            'adam': torch.optim.Adam,
            #'adagrad': torch.optim.Adagrad,
            #'RMSprop': torch.optim.RMSprop,
            #'SGD': torch.optim.SGD,
            #'LBFGS': torch.optim.LBFGS
        }[optim_cfg.optimizer_name]

    optimizer = optimizer_type(model.parameters(), lr=optim_cfg.learning_rate)  # TODO try different optimizers
    best_metrics, best_metrics_str = metrics, metrics_str
    best_R_list, best_t_list = model.get_R_t()

    ref_masks = [
        torch.from_numpy(mask.astype(np.float32))
        for mask in reference_masks]

    reference_rays = []
    for mask in ref_masks:
        indices_v, indices_u = torch.nonzero(mask, as_tuple=True)
        x = (indices_u - model.renderer.intrinsics[0 ,2]) / model.renderer.intrinsics[0,0]
        y = (indices_v - model.renderer.intrinsics[1,2]) / model.renderer.intrinsics[1,1]
        z = torch.ones(x.shape)

        l2_norm = torch.sqrt(x**2 + y**2 + 1.)
        x /= l2_norm
        y /= l2_norm
        z /= l2_norm
        reference_rays.append(
            torch.cat([x[:, None],y[:, None],z[:, None]], dim=1).to(device))

    ref_masks = [mask.to(device) for mask in ref_masks]

    optim_images = []
    mask_images = [[] for _ in reference_masks]
    pbar = tqdm(range(optim_cfg.max_iter))
    for i in pbar:

        optimizer.zero_grad()

        loss, losses_values, image_est, depth_est, obj_masks, fragments_est = model(
            reference_rgb,
            reference_depth,
            ref_masks,
            reference_rays)  # Calling forward function

        loss.backward()
        optimizer.step()

        if debug_flag:
            out_np = K.utils.tensor_to_image(torch.movedim(image_est[..., :3], 3, 1))
            blended = viz_blend * out_np + (1-viz_blend) * reference_rgb[...,::-1] / 255.
            optim_images.append((255*blended).astype(np.uint8))

            for mask_idx, (obj_mask, mask) in enumerate(zip(obj_masks, reference_masks)):
                tmp_mask = obj_mask.detach().cpu().numpy().astype(float) - mask
                mask_images[mask_idx].append(
                    (255*(tmp_mask - tmp_mask.min()) / (tmp_mask.max() - tmp_mask.min()))[0].astype(np.uint8))

        pbar.set_description("LOSSES: {}".format(
            " | ".join([f"{name}: {np.array(loss_val).sum():.3f}"
                        for name, loss_val in losses_values.items()])))

        # early stopping
        if loss.item() < scene_early_stopping_loss:
            break

    if debug_flag:
        imageio.mimsave(os.path.join(debug_path, "optim.gif"), optim_images, fps=5)
        for obj_idx, obj_mask_images in enumerate(mask_images):
            imageio.mimsave(os.path.join(debug_path, "mask-{}.gif".format(obj_idx)), obj_mask_images, fps=5)

    # record end and synchronize
    end.record()
    torch.cuda.synchronize()

    # get time between events (in ms)
    print("______timing_________: ", start.elapsed_time(end))

    iter_values.setdefault("best_T")
    iter_values["r"].append([(best_R.to('cpu').detach().numpy()).tolist() for best_R in best_R_list])
    iter_values["t"].append([(best_t.to('cpu').detach().numpy()).tolist() for best_t in best_t_list])
    iter_values["loss"].append(0)

    return best_metrics, iter_values
