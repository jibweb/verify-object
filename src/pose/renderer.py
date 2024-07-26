
import torch
import torch.nn as nn

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from pytorch3d.transforms import (
    matrix_to_quaternion, quaternion_to_matrix, so3_log_map, so3_exp_map, se3_log_map, se3_exp_map,
    matrix_to_euler_angles, euler_angles_to_matrix
)
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, BlendParams, SoftSilhouetteShader, HardGouraudShader, SoftGouraudShader, HardPhongShader,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from collision import transformations as tra
import numpy as np
import open3d as o3d

import matplotlib.pyplot as plt
import kornia as K


class Renderer(nn.Module):
    def __init__(self, meshes, intrinsics, width, height, representation='q'):
        super().__init__()
        self.meshes = meshes
        self.device = device  # TODO : should I check the device for all the objects ? Or assume that they are all set for cuda?

        # Plane (Table in this dataset) point clouds and transformation matrix
        self.plane_pcd = None
        self.plane_T_matrix = None

        # Camera intrinsic
        self.intrinsics = intrinsics
        cam = o3d.camera.PinholeCameraIntrinsic()
        cam.intrinsic_matrix = intrinsics.tolist()
        cam.height = height
        cam.width = width

        # Camera
        # - "assume that +X points left, and +Y points up and +Z points out from the image plane"
        # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/renderer_getting_started.md
        # - this is different from, e.g., OpenCV -> inverting focal length achieves the coordinate flip (hacky solution)
        # - see https://github.com/facebookresearch/pytorch3d/issues/522#issuecomment-762793832
        ren_intrinsics = intrinsics.copy()
        ren_intrinsics[0, 0] *= -1  # Based on the differentiation between the coordinate systems: negative focal length
        ren_intrinsics[1, 1] *= -1
        ren_intrinsics = ren_intrinsics.astype(np.float32)
        cameras = cameras_from_opencv_projection(R=torch.from_numpy(np.eye(4, dtype=np.float32)[None, ...]),
                                                 tvec=torch.from_numpy(np.asarray([[0, 0, 0]]).astype(np.float32)),
                                                 camera_matrix=torch.from_numpy(ren_intrinsics[:3, :3][None, ...]),
                                                 image_size=torch.from_numpy(
                                                     np.asarray([[height, width]]).astype(np.float32)))
        self.cameras = cameras.to(device)

        # SoftRas-style rendering
        # - [faces_per_pixel] faces are blended
        # - [sigma, gamma] controls opacity and sharpness of edges
        # - If [bin_size] and [max_faces_per_bin] are None (=default), coarse-to-fine rasterization is used.
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))  # TODO vary this, faces_per_pixel etc. to find good value
        soft_raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius= 0., #np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=20,
            max_faces_per_bin=20000,
            # perspective_correct=True, # TODO: Correct?
        )
        lights = PointLights(device=device, location=((0.0, 0.0, 0.0),), ambient_color=((1.0, 1.0, 1.0),),
                             diffuse_color=((0.0, 0.0, 0.0),), specular_color=((0.0, 0.0, 0.0),),
                             )  # at origin = camera center
        self.ren_opt = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=soft_raster_settings
            ),
            # shader=SoftSilhouetteShader(blend_params=blend_params)
            shader=SoftGouraudShader(blend_params=blend_params, device=device, cameras=self.cameras, lights=lights)
        ).to(device)

        # Placeholders for optimization parameters and the reference image
        # - rotation matrix must be orthogonal (i.e., element of SO(3)) - easier to use quaternion
        self.representation = representation
        if self.representation == 'se3':
            self.log_list = None
        elif self.representation == 'so3':
            self.r_list = None
            self.t_list = None
        elif self.representation == 'q':
            self.q_list = None
            self.t_list = None
        elif self.representation == 'in-plane':
            # TODO dummies
            self.scale_rotation = 10  # bigger impact of rotation st it's not dominated by translation

    def init(self, scene_objects, T_init_list): # TODO : Did I do it correctly here?
        self.scene_meshes = [
            self.meshes[object_name].clone().to(device)
            for object_name in scene_objects
        ]
        self.init_repr(T_init_list)

    def init_repr(self, T_init_list):
        if self.representation == 'se3':
            self.log_list = nn.Parameter(torch.stack([(se3_log_map(T_init)) for T_init in T_init_list]))
        elif self.representation == 'so3':
            self.r_list = nn.Parameter(torch.stack([(so3_log_map(T_init[:, :3, :3])) for T_init in T_init_list]))
            self.t_list = nn.Parameter(torch.stack([(T_init[:, :3, 3]) for T_init in T_init_list]))
        elif self.representation == 'q':  # [q, t] representation
            self.q_list = nn.Parameter(torch.stack([(matrix_to_quaternion(T_init[:, :3, :3])) for T_init in T_init_list])).to(self.device)
            self.t_list = nn.Parameter(torch.stack([(T_init[:, :3, 3]) for T_init in T_init_list])).to(self.device)
        # elif self.representation == 'in-plane': # TODO: Check for in-plane how should this change ?
        #     # in this representation, we only update two delta values
        #     # - rz = in-plane z rotation, txy = in-plane xy translation
        #     # - they are applied on top of T_init in place space
        #     assert T_plane is not None
        #     self.T_plane, self.T_plane_inv = T_plane, T_plane.inverse()
        #     self.T_init, self.T_init_inplane = T_init, T_init @ self.T_plane_inv
        #     self.rz = nn.Parameter(torch.zeros((T_plane.shape[0]), dtype=torch.float32, device=device))
        #     self.txy = nn.Parameter(torch.zeros((T_plane.shape[0], 2), dtype=torch.float32, device=device))

    def get_R_t(self): # TODO: Check other representations, does it work?? ** Use torch.stack
        if self.representation == 'se3': #TODO: adapt this
            self.log_list
            T = se3_exp_map(self.log)
            return T[:, :3, :3], T[:, :3, 3]
        elif self.representation == 'so3': #TODO: adapt this
            return so3_exp_map(self.r), self.t
        elif self.representation == 'q': #TODO: Is this correct?
            return torch.stack([quaternion_to_matrix(q) for q in self.q_list]), self.t_list
        elif self.representation == 'in-plane': #TODO: adapt this, no idea how
            eulers = matrix_to_euler_angles(self.T_init_inplane[:, :3, :3], "XYZ")
            eulers[:, 2] += self.rz * self.scale_rotation
            T = self.T_init_inplane.clone()
            T[:, :3, :3] = euler_angles_to_matrix(eulers, "XYZ")
            T[:, 3, :2] += self.txy
            T = T @ self.T_plane
            return T[:, :3, :3], T[:, :3, 3]

    def get_transform(self): # TODO: Is it correct?
        T_list = []
        r_list, t_list = self.get_R_t()
        for i in range(len(r_list)):
            T = torch.eye(4, device=device)[None, ...]
            T[:, :3, :3] = r_list[i]
            T[:, :3, 3] = t_list[i]
            T_list.append(T)
        return T_list

    def forward(self, max_scene_depth=4):
        # Render the silhouette using the estimated pose
        R, t = self.get_R_t()  # (N, 1, 3, 3), (N, 1, 3)

        as_scene = True

        if as_scene:  # 1 image
            meshes_transformed = []
            meshes_faces_num = [0]
            for mesh, mesh_r, mesh_t in zip(self.scene_meshes, R, t):
                new_verts_padded = \
                    ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                mesh = mesh.update_padded(new_verts_padded)
                meshes_transformed.append(mesh)
                meshes_faces_num.append(meshes_faces_num[-1] + mesh.faces_packed().shape[0])
            self.scene_transformed = join_meshes_as_scene(meshes_transformed)

            # Fragments have : pix_to_face, zbuf, bary_coords, dists
            image_est, self.fragments_est = self.ren_opt(
                meshes_world=self.scene_transformed,
                R=torch.eye(3)[None, ...].to(device),
                T=torch.zeros((1, 3)).to(device))

            zbuf = self.fragments_est.zbuf # (N, h, w, k ) where N in as_scene = 1
            image_depth_est = torch.where(zbuf >= 0, zbuf, max_scene_depth)
            image_depth_est, depth_indices = image_depth_est.min(-1)
            image_depth_est = torch.where(
                image_depth_est != max_scene_depth, image_depth_est, 0.)

            pix_to_close_face = self.fragments_est.pix_to_face[..., 0]
            obj_masks = []
            for val_idx, val in enumerate(meshes_faces_num[1:]):
                mesh_mask = (pix_to_close_face >= meshes_faces_num[val_idx]) & (pix_to_close_face < val)
                obj_masks.append(mesh_mask)

        return image_est, image_depth_est, obj_masks, self.fragments_est
