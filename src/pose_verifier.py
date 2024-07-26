import argparse
import cv2
import imageio
from matplotlib import pyplot as plt
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

from pose.model import OptimizationModel
import pose.object_pose as object_pose
from collision.plane_detector import PlaneDetector
from collision.point_clouds_utils import plane_pt_intersection_along_ray
# from contour.contour import compute_sdf_image
# from utility.logger import Logger
from config import config
from collision.point_clouds_utils import project_point_cloud


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)  # To check whether we have nan or inf in our gradient calculation


INTERNAL_TO_PROJECT_NAMES = {
    1: "MediumBottle",
    2: "SmallBottle",
    3: "Needle",
    4: "NeedleCap",
    5: "RedPlug",
    6: "Canister",
    7: "LargeBottle",
    8: "YellowPlug",
    9: "WhiteClamp",
    10: "RedClamp",
    # 'container': 'Canister',
}


PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}


SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()


class VerifyPose:
    def __init__(self, cfg, intrinsics, objects_names, width, height,
                 plane_normal=None, plane_pt=None, debug_flag=False):
        self.debug_flag = debug_flag
        self.plane_normal, self.plane_pt = plane_normal, plane_pt

        self.scale = width // 320
        self.intrinsics = intrinsics
        self.intrinsics[:2, :] /= self.scale
        print("Intrisics", self.intrinsics)

        self.cfg = cfg

        # load all meshes that will be needed
        self.cmap = plt.cm.tab20(range(20)) # 20 different colors, two consecutive ones are similar (for two instances)
        meshes, sampled_down_meshes = object_pose.load_objects_models(
            ["obj_{:06d}".format(obj_id) for obj_id in objects_names],
            cfg.objects_path,
            cmap=self.cmap,
            mesh_num_samples=cfg.mesh_num_samples)

        # Optimization model
        self.model = OptimizationModel(
            meshes,
            sampled_down_meshes,
            self.intrinsics,
            width // self.scale,
            height // self.scale,
            cfg.optim,
            debug=self.debug_flag,
            debug_path=self.cfg.debug_path)

    def refine_masks_with_scene_depth(self, masks, depth):
        return [
            np.logical_and(mask.astype(bool), depth != 0)
            for mask in masks
        ]

    def refine_masks_with_grabcut(self, ref_rgb, masks):
        kernel = np.ones((3,3),np.uint8)
        refined_masks = []
        for mask in tqdm(masks, desc='Mask refinement with GrabCut'):
            obj_mask = mask.astype(np.uint8)
            erosion = cv2.erode(obj_mask, kernel, iterations=1)
            dilation = cv2.dilate(obj_mask, kernel, iterations=1)
            new_mask = np.zeros(obj_mask.shape, dtype=np.uint8)
            new_mask[dilation.astype(bool)] = 2
            new_mask[erosion.astype(bool)] = 1

            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            gb_mask, bgdModel, fgdModel = cv2.grabCut(
                ref_rgb, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            refined_masks.append(
                np.where((gb_mask==2)|(gb_mask==0),False,True)
            )

        return refined_masks

    def optimize(self, rgb, depth, scene_objects, init_poses, masks):
        intrinsics = self.intrinsics.copy()

        if self.debug_flag:
            cv2.imwrite(os.path.join(self.cfg.debug_path, "input_depth.png"), depth)
            cv2.imwrite(os.path.join(self.cfg.debug_path, "input_color.png"), rgb)

        # Scale images ========================================================
        if self.scale != 1:
            rgb = cv2.resize(rgb, (rgb.shape[1] // self.scale, rgb.shape[0] // self.scale))
            depth = cv2.resize(depth, (depth.shape[1] // self.scale, depth.shape[0] // self.scale)) # TODO: check size to see if scaling is necessary
            masks = [
                cv2.resize(
                    mask.astype(np.float32),
                    (mask.shape[1] // self.scale, mask.shape[0] // self.scale)
                ).astype(bool)
                for mask in masks]

        reference_height, reference_width = rgb.shape[:2]

        # Extract plane =======================================================
        plane_det = PlaneDetector(
            reference_width, reference_height,
            to_meters=1e-3,
            distance_threshold=0.01)
        cloud = plane_det.create_point_cloud(
            rgb, depth, intrinsics, max_dist=1.5)
        if self.plane_normal is None or self.plane_pt is None:
            plane_T, plane, scene, indices = plane_det.detect(cloud)

            self.plane_normal = plane_T[:,2][:3].astype(np.float32)
            self.plane_pt = np.asarray(plane.points[0].astype(np.float32))
        else:
            plane, scene = plane_det.filter_plane(
                cloud, self.plane_normal, self.plane_pt)

        # Create the filtered scene depth image
        scene_depth = project_point_cloud(
            scene, intrinsics, reference_height, reference_width)

        if self.debug_flag:
            plt.imshow(scene_depth); plt.savefig(os.path.join(self.cfg.debug_path, "ref_depth.png"))

        # Filter detection based on depth =====================================
        # perc_valid_depth_per_det = []
        # det_mean_dist = []
        # for bbox in bounding_boxes:
        #     bbox_depth = scene_depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        #     # print("Bboxes & depth", bbox, bbox_depth)
        #     perc_valid_depth_per_det.append(
        #         np.mean(bbox_depth != 0))
        #     det_mean_dist.append(np.mean(bbox_depth[bbox_depth != 0]))

        # valid_det_mask = np.array(perc_valid_depth_per_det) > 0.1

        # print("valid_det_mask", valid_det_mask)

        # scene_objects = list(np.array(scene_objects)[valid_det_mask])
        # init_poses = list(np.array(init_poses)[valid_det_mask])
        # bounding_boxes = list(np.array(bounding_boxes)[valid_det_mask])
        # if len(goal.object_masks) != 0:
        #     masks = list(np.array(masks)[valid_det_mask])

        # Prepare target silhouettes ==========================================
        if self.cfg.mask_grabcut_refinement:
            masks = self.refine_masks_with_grabcut(rgb, masks)

        # Create colored instance mask
        reference_mask = np.zeros((reference_height, reference_width, 3))
        for mask_idx, mask in enumerate(masks):
            reference_mask[mask] = self.cmap[mask_idx, :3]

        if self.debug_flag:
            cv2.imwrite(os.path.join(self.cfg.debug_path, "ref_mask.png"),
                        (0.7*255*reference_mask + 0.3*rgb).astype(np.uint8))
            cv2.imwrite(os.path.join(self.cfg.debug_path, "ref_rgb.png"), rgb)


        # Set up optimization =================================================
        T_init_list = [torch.from_numpy(pose[None, ...]).to(device)
                     for pose in init_poses]

        reference_rays = self.model.init(
            scene_objects,
            T_init_list,
            masks,
            plane_normal=self.plane_normal,
            plane_pt=self.plane_pt)

        # Perform optimization ================================================
        scene_early_stopping_loss = len(masks) * sum(
            [v.weight * v.early_stopping_loss
             for k, v in self.cfg.optim.losses.__dict__.items()
             if v.active]
        )

        optimizer_type = {
            'adam': torch.optim.Adam,
            #'adagrad': torch.optim.Adagrad,
            #'RMSprop': torch.optim.RMSprop,
            #'SGD': torch.optim.SGD,
            #'LBFGS': torch.optim.LBFGS
        }[self.cfg.optim.optimizer_name]

        optimizer = optimizer_type(
            self.model.parameters(), lr=self.cfg.optim.learning_rate)  # TODO try different optimizers
        best_R_list, best_t_list = self.model.get_R_t()

        ref_masks = [
            torch.from_numpy(mask.astype(np.float32))
            for mask in masks]

        ref_masks = [mask.to(device) for mask in ref_masks]

        optim_images = []
        mask_images = [[] for _ in masks]
        pbar = tqdm(range(self.cfg.optim.max_iter))
        for i in pbar:
            optimizer.zero_grad()
            loss, losses_values, image_est, depth_est, obj_masks, fragments_est = self.model(
                rgb,
                scene_depth,
                ref_masks,
                reference_rays)  # Calling forward function

            loss.backward()
            optimizer.step()

            if self.debug_flag:
                viz_blend = 0.4
                out_np = image_est[..., :3].cpu().detach().squeeze().numpy()
                blended = viz_blend * out_np + (1-viz_blend) * rgb[...,::-1] / 255.
                optim_images.append((255*blended).astype(np.uint8))

                for mask_idx, (obj_mask, mask) in enumerate(zip(obj_masks, masks)):
                    tmp_mask = obj_mask.detach().cpu().numpy().astype(float) - mask
                    mask_images[mask_idx].append(
                        (255*(tmp_mask - tmp_mask.min()) / (tmp_mask.max() - tmp_mask.min()))[0].astype(np.uint8))

            pbar.set_description("LOSSES: {}".format(
                " | ".join([f"{name}: {np.array(loss_val).sum():.3f}"
                            for name, loss_val in losses_values.items()])))

            # early stopping
            # if loss.item() < scene_early_stopping_loss:
            #     break

        if self.debug_flag:
            imageio.mimsave(
                os.path.join(self.cfg.debug_path, "optim.gif"),
                optim_images, fps=5)
            for obj_idx, obj_mask_images in enumerate(mask_images):
                imageio.mimsave(
                    os.path.join(
                        self.cfg.debug_path,
                        "mask-{}.gif".format(obj_idx)),
                    obj_mask_images, fps=5)

        best_R_list, best_t_list = self.model.get_R_t()
        predicted_poses = []
        for best_R, best_t in zip(best_R_list, best_t_list):
            pose = np.eye(4)
            pose[:3, :3] = best_R.to('cpu').detach().numpy()[0]
            pose[:3, 3] = best_t.to('cpu').detach().numpy()[0]
            predicted_poses.append(pose)

        import open3d as o3d
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(self.model.renderer.scene_transformed.verts_packed().detach().cpu().numpy())
        o3d.visualization.draw_geometries([cloud, pcd2])

        return predicted_poses, rgb, depth, masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    args = parser.parse_args()

    cfg = config.GlobalConfig()

    # TODO
