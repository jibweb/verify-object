import numpy as np
import open3d as o3d
import os
import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from collision import transformations as tra
from collision.point_clouds_utils import plane_contact_loss, plane_pt_intersection_along_ray, point_ray_loss
from contour import Sobel
from pose.renderer import Renderer
from pytorch3d.ops import interpolate_face_attributes

import matplotlib.pyplot as plt


class OptimizationModel(nn.Module):
    def __init__(self, meshes, sampled_meshes, intrinsics, width, height,
                 objects_to_optimize, cfg, debug=False, debug_path="/debug"):
        super().__init__()
        # self.meshes = meshes
        self.sampled_meshes = sampled_meshes
        self.device = device  # TODO : should I check the device for all the objects ? Or assume that they are all set for cuda?
        self.cfg = cfg
        self.debug = debug
        self.debug_path = debug_path

        self.meshes_diameter = None
        self.mask_sigma = 1e-4

        self.sobel_filt = Sobel()

        # Set up renderer
        self.renderer = Renderer(
            meshes, intrinsics, width, height, objects_to_optimize,
            representation=cfg.pose_representation,
            faces_per_pixel=cfg.faces_per_pixel)

    def init(self, scene_objects, T_init_list, masks,
             relative_pose=None, point_contacts=None,
             plane_normal=None, plane_pt=None):
        # Init for rendering
        self.renderer.init(scene_objects, T_init_list)

        # Init for collision
        self.scene_sampled_meshes = [
            self.sampled_meshes[object_name].clone().to(device)
            for object_name in scene_objects
        ]

        if relative_pose is not None:
            self.relative_pose = relative_pose

        if point_contacts is not None:
            self.point_contacts = torch.from_numpy(point_contacts).to(device)

        if plane_normal is not None and plane_pt is not None:
            self.plane_normal = torch.from_numpy(plane_normal).to(device)
            self.plane_pt = torch.from_numpy(plane_pt).to(device)

        self.ref_masks = masks
        self.ref_rays = []
        self.ref_contour_masks = []
        self.ref_contour_rays = []
        for obj_idx, mask in enumerate(masks):
            # Convert mask pixels to rays
            x,y,z = self.mask_to_rays(mask)

            self.ref_rays.append(
                torch.cat([x[:, None],y[:, None],z[:, None]], dim=1).to(device))

            if self.cfg.losses.contour_loss.active:
                # Convert contour mask to rays
                contour = self.sobel_filt(mask[None, None, ...].float())[0,0] > 0
                self.ref_contour_masks.append(contour)
                x_cont, y_cont, z_cont = self.mask_to_rays(contour)
                self.ref_contour_rays.append(
                    torch.cat([x_cont[:, None],y_cont[:, None],z_cont[:, None]], dim=1).to(device))

            if plane_normal is not None and plane_pt is not None and self.cfg.plane_refinement:
                # --- Align with supporting plane along mask rays -------------
                # Create average ray
                ray = torch.Tensor([
                    x[:, None].mean(),
                    y[:, None].mean(),
                    z[:, None].mean()]).to(device)

                # Prepare meshes
                rot, trans = T_init_list[obj_idx][:, :3, :3], T_init_list[obj_idx][:, :3, 3]
                mesh = self.renderer.scene_meshes[obj_idx]
                verts_trans = \
                    ((rot @ mesh.verts_padded()[..., None]) + trans[..., None])[..., 0]

                # Compute the difference to apply along ray to ensure plane contact
                vec_to_plane = plane_pt_intersection_along_ray(
                    ray,
                    verts_trans[0],
                    self.plane_normal,
                    self.plane_pt)

                # Adapt the initial pose
                T_init_list[obj_idx][:, :3, 3] += vec_to_plane

        if self.cfg.plane_refinement:
            # Re-init the model with the corrected filtered poses
            self.renderer.init_repr(T_init_list)

    def mask_to_rays(self, mask, normalize=True):
        indices_v, indices_u = torch.nonzero(mask, as_tuple=True)
        x = (indices_u - self.renderer.intrinsics[0 ,2]) / self.renderer.intrinsics[0,0]
        y = (indices_v - self.renderer.intrinsics[1,2]) / self.renderer.intrinsics[1,1]
        z = torch.ones(x.shape, device=mask.device)

        if normalize:
            l2_norm = torch.sqrt(x**2 + y**2 + 1.)
            x /= l2_norm
            y /= l2_norm
            z /= l2_norm

        return x, y, z

    def get_R_t(self):
        return self.renderer.get_R_t()

    def forward(self, ref_rgb, ref_depth):
        # Render the silhouette using the estimated pose
        image_est, depth_est, obj_masks, fragments_est = self.renderer()

        loss = torch.zeros(1).to(device)
        losses_values = {}

        # === SETUP ===========================================================
        MAX_FACE = min(5,self.cfg.faces_per_pixel)
        if self.cfg.losses.silhouette_loss.active or self.cfg.losses.contour_loss.active:
            # Mask for padded pixels.
            valid_mask = fragments_est.pix_to_face >= 0
            pix_position = interpolate_face_attributes(
                fragments_est.pix_to_face,
                torch.ones(fragments_est.bary_coords.shape, device=device),
                self.renderer.scene_transformed.verts_packed()[
                    self.renderer.scene_transformed.faces_packed()]
            )[..., :MAX_FACE, :]

        if (self.cfg.losses.point_contact_loss and hasattr(self, 'point_contacts')) or \
           (self.cfg.losses.plane_collision_loss.active and hasattr(self, 'plane_normal') and hasattr(self, 'plane_pt')):
            # Create scene point cloud from models and poses
            R, t = self.get_R_t()  # (N, 1, 3, 3), (N, 1, 3)
            points = torch.cat(self.scene_sampled_meshes, dim=0)  # (N, N_pts , 6 (coordinates and norms))
            points_in_cam = torch.stack([
                (rot_mat @ points[idx, :, :3, None])[..., 0] + t[idx] for idx, rot_mat in enumerate(R)])  # (N, N_pts, 3)

        # === LOSSES ==========================================================
        # Silhouette loss -----------------------------------------------------
        if self.cfg.losses.silhouette_loss.active:
            diff_rend_loss = torch.zeros(len(self.ref_masks)).to(device)
            for ray_idx, rays in enumerate(self.ref_rays):
                if not self.renderer.active_objects[ray_idx]:
                    continue
                obj_face_mask = torch.logical_and(
                    valid_mask[..., :MAX_FACE],  # Only consider the MAX_FACE closest faces
                    obj_masks[ray_idx][..., None])
                obj_pix_position = pix_position[obj_face_mask]

                if len(obj_pix_position) == 0:
                    print("WARNING: object mesh {} outside of camera frustum".format(
                        ray_idx))
                    zero_loss = torch.tensor(0.)
                    zero_loss.requires_grad_()
                    diff_rend_loss[ray_idx] = zero_loss
                else:
                    diff_rend_loss[ray_idx] = point_ray_loss(rays, obj_pix_position)

            # diff_rend_loss = torch.sum(diff_rend_loss)
            loss += torch.sum(diff_rend_loss) * self.cfg.losses.silhouette_loss.weight
            losses_values['silhouette'] = torch.sum(diff_rend_loss).item()

        # Contour loss --------------------------------------------------------
        if self.cfg.losses.contour_loss.active:
            contour_loss = torch.zeros(len(self.ref_masks)).to(device)

            contour_masks = []
            for ray_idx, rays in enumerate(self.ref_contour_rays):
                if not self.renderer.active_objects[ray_idx]:
                    continue
                contour_mask = self.sobel_filt(obj_masks[ray_idx][None, ...].float())[0,0] > 0
                contour_masks.append(contour_mask[None, ...])

                obj_cont_face_mask = torch.logical_and(
                    valid_mask[..., :MAX_FACE],  # Only consider the MAX_FACE closest faces
                    contour_mask[..., None])
                obj_cont_position = pix_position[obj_cont_face_mask]

                contour_loss[ray_idx] = point_ray_loss(rays, obj_cont_position)

            loss += torch.sum(contour_loss) * self.cfg.losses.contour_loss.weight
            losses_values['contour'] = torch.sum(contour_loss).item()

        # Collision loss ------------------------------------------------------
        if self.cfg.losses.collision_loss.active:
            # TODO: collision_loss finish implementation
            print('Object collision loss not implemented')
            # losses_values['collision'] = .item()

        # Point contact loss ------------------------------------------------
        if (self.cfg.losses.point_contact_loss.active and hasattr(self, 'point_contacts')):
            # TODO: point_contact_loss test implentation
            distances = torch.cdist(points_in_cam, self.point_contacts) # (N, N_pts, 3) and (N, N_objcontact, 3) -> (N, N_pts, N_objcontact)
            dist_to_closest_obj_pt = distances.min(dim=1).values  # (N, N_objcontact)
            loss += torch.mean(dist_to_closest_obj_pt) * self.cfg.losses.point_contact_loss.weight
            losses_values['point_contact'] = torch.mean(dist_to_closest_obj_pt).item()

        # Plane collision loss ------------------------------------------------
        if self.cfg.losses.plane_collision_loss.active and hasattr(self, 'plane_normal') and hasattr(self, 'plane_pt'):
            plane_col_loss = plane_contact_loss(
                points_in_cam,
                self.plane_normal,
                self.plane_pt)
            loss += plane_col_loss * self.cfg.losses.plane_collision_loss.weight
            losses_values['plane_loss'] = plane_col_loss.item()

        # Depth loss ----------------------------------------------------------
        if self.cfg.losses.depth_loss.active:
            # TODO: depth_loss finish implementation
            ref_depth_tensor = torch.from_numpy(
                ref_depth.astype(np.float32)).to(device)
            # ref_depth_tensor *= (self.image_ref[..., 0] > 0).float()
            # d_depth = (ref_depth_tensor - depth_est)
            # depth = torch.gather(zbuf, 0, depth_indices[..., None])
            d_depth = (depth_est - ref_depth_tensor[None, ..., None])[depth_est > 0]

            # depth_loss = torch.sum(d_depth**2) / torch.sum((zbuf > -1).float())
            depth_loss = torch.sum(d_depth**2) / (d_depth.shape[0])
            loss += depth_loss * self.cfg.losses.depth_loss.weight
            losses_values['depth'] = depth_loss.item()

        # Relative pose loss --------------------------------------------------
        if self.cfg.losses.relative_pose_loss.active:
            # TODO: relative_pose_loss finish implementation
            relative_pose_loss = torch.zeros(len(self.relative_pose)).to(device)
            # Get pose of each object
            R, t = self.get_R_t()
            for idx, ((o1, o2), (ref_rot, ref_t)) in enumerate(self.relative_pose.items()):
                relative_pose_loss[idx] +=  torch.acos((torch.trace((R[o2] @ R[o1].T) @ ref_rot) - 1.) / 2.)
                relative_pose_loss[idx] +=  torch.sqrt(torch.pow((t[o1] - t[o2]) - ref_t, 2).sum())


            loss += torch.sum(relative_pose_loss) * self.cfg.losses.relative_pose_loss.weight
            losses_values['relative_pose'] = torch.sum(relative_pose_loss).item()


        if self.cfg.losses.contour_loss.active:
            return loss, losses_values, image_est, depth_est, contour_masks, fragments_est
        else:
            return loss, losses_values, image_est, depth_est, obj_masks, fragments_est


def square_distance(pcd1, pcd2):
    # via https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/13
    r_xyz1 = torch.sum(pcd1 * pcd1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(pcd2 * pcd2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(pcd1, pcd2.permute(0, 2, 1))  # (B,M,N)
    return r_xyz1 - 2 * mul + r_xyz2.permute(0, 2, 1)
