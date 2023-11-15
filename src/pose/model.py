import cv2
import pytorch3d.transforms
import torch
import torch.nn as nn
from trimesh import Trimesh
import open3d
from src.contour.contour import imsavePNG

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
from pytorch3d.transforms import (
    matrix_to_quaternion, quaternion_to_matrix, so3_log_map, so3_exp_map, se3_log_map, se3_exp_map,
    matrix_to_euler_angles, euler_angles_to_matrix
)
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    PointLights, BlendParams, SoftSilhouetteShader, SoftGouraudShader, HardPhongShader,
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from src.collision import transformations as tra
import numpy as np
import cv2 as cv
import open3d as o3d
from src.contour import contour

import matplotlib.pyplot as plt


class OptimizationModel(nn.Module):
    def __init__(self, meshes, intrinsics_dict, representation='q', image_scale=1, loss_function_num=1, BOP=False):
        super().__init__()
        self.meshes = meshes
        self.meshes_name = None
        self.sampled_meshes = None
        self.device = device  # TODO : should I check the device for all the objects ? Or assume that they are all set for cuda?
        self.loss_func_num = loss_function_num
        self.camera_ins = None

        self.meshes_diameter = None

        # Plane (Table in this dataset) point clouds and transformation matrix
        self.plane_pcd = None
        self.plane_T_matrix = None

        # Camera intrinsic
        cam = o3d.camera.PinholeCameraIntrinsic()

        if BOP :
            cam.intrinsic_matrix = (np.reshape(np.asarray(intrinsics_dict), (3, 3))).tolist()
            cam.height = 540
            cam.width = 720
            self.camera_ins = cam
            # Scale
            # - speed-up rendering
            # - need to adapt intrinsics accordingly
            width, height = 720 // image_scale, 540 // image_scale
            intrinsics = np.asarray(intrinsics_dict).reshape(3, 3)
            intrinsics[:2, :] //= image_scale

        else:
            cam.intrinsic_matrix = (np.reshape(np.asarray(intrinsics_dict["camera_matrix"]), (3, 3))).tolist()
            cam.height = intrinsics_dict["image_height"]
            cam.width = intrinsics_dict["image_width"]
            self.camera_ins = cam
            # Scale
            # - speed-up rendering
            # - need to adapt intrinsics accordingly
            width, height = intrinsics_dict['image_width'] // image_scale, intrinsics_dict['image_height'] // image_scale
            intrinsics = np.asarray(intrinsics_dict['camera_matrix']).reshape(3, 3)
            intrinsics[:2, :] //= image_scale

        # Camera
        # - "assume that +X points left, and +Y points up and +Z points out from the image plane"
        # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
        # - see https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/renderer_getting_started.md
        # - this is different from, e.g., OpenCV -> inverting focal length achieves the coordinate flip (hacky solution)
        # - see https://github.com/facebookresearch/pytorch3d/issues/522#issuecomment-762793832
        intrinsics[0, 0] *= -1  # Based on the differentiation between the coordinate systems: negative focal length
        intrinsics[1, 1] *= -1
        intrinsics = intrinsics.astype(np.float32)
        cameras = cameras_from_opencv_projection(R=torch.from_numpy(np.eye(4, dtype=np.float32)[None, ...]),
                                                 tvec=torch.from_numpy(np.asarray([[0, 0, 0]]).astype(np.float32)),
                                                 camera_matrix=torch.from_numpy(intrinsics[:3, :3][None, ...]),
                                                 image_size=torch.from_numpy(
                                                     np.asarray([[height, width]]).astype(np.float32)))
        self.cameras = cameras.to(device)

        # SoftRas-style rendering
        # - [faces_per_pixel] faces are blended
        # - [sigma, gamma] controls opacity and sharpness of edges
        # - If [bin_size] and [max_faces_per_bin] are None (=default), coarse-to-fine rasterization is used.
        blend_params = BlendParams(sigma=1e-5, gamma=1e-5, background_color=(0.0, 0.0, 0.0))  # TODO vary this, faces_per_pixel etc. to find good value
        soft_raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma,
            faces_per_pixel=100,
            perspective_correct=True, # TODO: Correct?
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
        )

        # Simple Phong-shaded renderer
        # - faster for visualization
        hard_raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        self.ren_vis = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=hard_raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=self.cameras, lights=lights)
        )

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
        self.image_ref = None  # Image mask reference

    def init(self, image_ref, T_init_list, T_plane=None): # TODO : Did I do it correctly here?
        self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(device)
        if self.representation == 'se3':
            self.log_list = nn.Parameter(torch.stack([(se3_log_map(T_init)) for T_init in T_init_list]))
        elif self.representation == 'so3':
            self.r_list = nn.Parameter(torch.stack([(so3_log_map(T_init[:, :3, :3])) for T_init in T_init_list]))
            self.t_list = nn.Parameter(torch.stack([(T_init[:, 3, :3]) for T_init in T_init_list]))
        elif self.representation == 'q':  # [q, t] representation
            self.q_list = nn.Parameter(torch.stack([(matrix_to_quaternion(T_init[:, :3, :3])) for T_init in T_init_list]))
            self.t_list = nn.Parameter(torch.stack([(T_init[:, 3, :3]) for T_init in T_init_list]))
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
            return T[:, :3, :3], T[:, 3, :3]
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
            return T[:, :3, :3], T[:, 3, :3]

    def get_transform(self): # TODO: Is it correct?
        T_list = []
        r_list, t_list = self.get_R_t()
        for i in range(len(r_list)):
            T = torch.eye(4, device=device)[None, ...]
            T[:, :3, :3] = r_list[i]
            T[:, 3, :3] = t_list[i]
            T_list.append(T)
        return T_list

    def signed_dis(self, isbop=False,k=10):
        """
        Calculating the signed distance of an object and the plane (The table)
        :return: two torch arrays with the length of number of points, showing the distance of that point and whether it
        is an intersection point or not
        """
        points = torch.cat([torch.tensor(self.sampled_meshes[i][None, ...], dtype=torch.float ,device=device) for i in range(len(self.sampled_meshes))],dim=0)# shape (num of meshes, num point in each obj , 6 (coordinates and norms))
        estimated_trans_matrixes = torch.cat([T.transpose(-2, -1) for T in self.get_transform()], dim=0) # Transposed because of the open3d and pytorch difference

        TOL_CONTACT = 0.01

        # === 1) all objects into plane space
        plane_T_matrix = torch.inverse(self.plane_T_matrix)
        transome_matrixes = plane_T_matrix @ estimated_trans_matrixes

        points_in_plane = (transome_matrixes[:, :3, :3] @ points[..., :3].transpose(2, 1)).transpose(2, 1) + transome_matrixes[ :, :3, 3][:, None, :]

        # TODO_Done if this is only for debugging, I'd comment it out st you save some time converting between GPU tensor and CPU/o3d point cloud
        # points_in_plane_array = points_in_plane[..., :3].cpu().detach().numpy()
        # pcd1 = o3d.geometry.PointCloud()
        # pcd2 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(points_in_plane_array[0])
        # pcd2.points = o3d.utility.Vector3dVector(points_in_plane_array[1])
        # o3d.visualization.draw_geometries([pcd1, pcd2])

        # === 2) get signed distance to plane
        signed_distance = points_in_plane[..., 2].clone() # shape (1, N)
        supported = torch.ones_like(signed_distance)


        # others_indices = [[1], [0]] # TODO : correct? Change it from hard code
        others_indices = [list(range(len(self.sampled_meshes))) for i in range(len(self.sampled_meshes))]
        [others_indices[i].remove(i) for i in range(len(others_indices))]

        if len(others_indices) > 0:  # === 3) get signed distance to other objects in the scene
            # remove scaling (via normalization) for rotation of the normal vectors
            targets_in_plane = points_in_plane.clone()
            normals_in_plane = points[..., 3:6]
            targets_in_plane = torch.cat([targets_in_plane, normals_in_plane], dim=-1) # The points are in the plane
            # space, but the norms for each individual object does not depend on other object and shows the outside
            # of the object

            batch_signed_distances = []
            batch_support = []
            for b, other_indices in enumerate(others_indices):
                num_other = len(other_indices)

                # get k nearest neighbors in each of the other objects
                distances, nearests = [], []
                for o in other_indices:
                    dist, idx = tra.nearest_neighbor(points_in_plane[o, ..., :3][None, ...],
                                                     points_in_plane[b, ..., :3][None, ...], k=k)
                    near = targets_in_plane[o][idx[0]]  # shape (k, num points, 3)
                    distances.append(dist)
                    nearests.append(near[None, ...])
                if num_other == 0:  # add plane distance instead (doesn't change min)
                    batch_signed_distances.append(signed_distance[b][None, ...])
                    batch_support.append(torch.ones_like(signed_distance[b][None, ...]))
                    continue
                distances = torch.cat(distances, dim=0)  # [num_other] x k x N
                nearests = torch.cat(nearests, dim=0)  # [num_other] x k x N x 6

                # check if query is inside or outside based on surface normal
                surface_normals = nearests[..., 3:6] # shape (num_other, k, N, 3)
                gradients = nearests[..., :3] - points_in_plane[b][None, :, :3]  # points towards surface (from b to o)
                gradients = gradients / torch.norm(gradients, dim=-1)[..., None]

                # # Debugging
                if torch.any(torch.isnan(nearests[..., :3])):
                    print("nearest :(")
                if torch.any(torch.isnan(gradients)):
                    print("gradient :(")
                if torch.any(torch.isnan(torch.norm(gradients, dim=-1)[..., None])):
                    print("gradient norm :(")

                # points_in_plane_array = points_in_plane[..., :3].cpu().detach().numpy()
                # nearest_array = nearests[0, 0, :, :3].cpu().detach().numpy()
                # pcd1 = o3d.geometry.PointCloud()
                # pcd2 = o3d.geometry.PointCloud()
                # pcd3 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(points_in_plane_array[b])
                # pcd1.normals = o3d.utility.Vector3dVector(gradients[0, 0, :, :3].cpu().detach().numpy())
                # pcd2.normals = o3d.utility.Vector3dVector(surface_normals[0, 0, :, :3].cpu().detach().numpy())
                # pcd2.points = o3d.utility.Vector3dVector(points_in_plane_array[o-1])
                # pcd3.points = o3d.utility.Vector3dVector(nearest_array)
                # pcd2.paint_uniform_color([1, 0.706, 0])
                # pcd1.paint_uniform_color([1, 0, 0])
                # pcd3.paint_uniform_color([0, 0, 0])
                # o3d.visualization.draw_geometries([pcd1, pcd3], point_show_normal=True)


                insides = torch.einsum('okij,okij->oki', surface_normals, gradients) > 0  # same direction -> inside  #dot-product
                # filter by quorum of votes
                inside = torch.sum(insides, dim=1) > k * 0.8 # shape (num_others, N)

                # get nearest neighbor (in each other object)
                distance, gradient, surface_normal = distances[:, 0, ...], gradients[:, 0, ...], surface_normals[:, 0,
                                                                                                 ...]

                # change sign of distance for points inside
                distance[inside] *= -1

                # take minimum over other points --> minimal SDF overall
                # = the closest outside/farthest inside each point is wrt any environment collider
                if num_other == 1:
                    batch_signed_distances.append(distance[0][None, ...])
                else:
                    distance, closest = distance.min(dim=0)
                    batch_signed_distances.append(distance[None, ...])

            signed_distances = torch.cat(batch_signed_distances, dim=0)

            signed_distance, closest = torch.cat([signed_distance[:, None], signed_distances[:, None]], dim=1).min(
                dim=1)  # the min distance, to which object we have the most collision ? that gives us the answer


        # === 4) derive critical points - allows to determine feasibility and stability
        contacts, intersects = signed_distance.abs() < TOL_CONTACT, signed_distance < -TOL_CONTACT
        # critical_points = torch.cat([contacts[..., None], intersects[..., None], supported[..., None]], dim=-1)

        return signed_distance, intersects


    def forward(self, ref_rgb_tensor, image_name_debug, debug_flag, isbop):
        # render the silhouette using the estimated pose
        R, t = self.get_R_t()  # (N, 1, 3, 3), (N, 1, 3)

        binary = True
        # binary = False
        as_scene = True
        contour_loss = None

        if as_scene:  # 1 image
            meshes_transformed = []
            for mesh, mesh_r, mesh_t in zip(self.meshes, R.transpose(-2, -1), t):
                new_verts_padded = \
                    ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
                mesh = mesh.update_padded(new_verts_padded)
                meshes_transformed.append(mesh)
            scene_transformed = join_meshes_as_scene(meshes_transformed)
            image_est, fragments_est = self.ren_opt(meshes_world=scene_transformed,
                                                    R=torch.eye(3)[None, ...].to(device),
                                                    T=torch.zeros((1, 3)).to(device))
            # Fragments have : pix_to_face, zbuf, bary_coords, dists

            zbuf = fragments_est.zbuf # (N, h, w, k ) where N in as_scene = 1
            zbuf[zbuf < 0] = torch.inf
            image_depth_est = zbuf.min(dim=-1)[0]
            image_depth_est[torch.isinf(image_depth_est)] = 0

        # else:  # N images
        #     scene = join_meshes_as_batch(self.meshes)
        #     image_est, fragments_est = self.ren_opt(meshes_world=scene.clone(), R=R[:, 0], T=t[:, 0])
        #     image_est = torch.clip(image_est.sum(dim=0)[None, ...], 0, 1)  # combine simple

        # Calculating the signed distance
        signed_dis, intersect_point = self.signed_dis(isbop=isbop)

        # import pdb; pdb.set_trace()
        # silhouette loss
        rounded_ref = torch.round(self.image_ref.sum(-1), decimals=4)
        vals = rounded_ref.unique()
        print(vals)
        rounded_image = torch.round(image_est[...,:3].sum(-1), decimals=4)

        diff_rend_loss = torch.zeros(vals.shape[0] - 1)
        for val_idx, val in enumerate(vals[1:]):
            image_unique_mask = torch.where(
                rounded_image == val, rounded_image / val, 0)
            ref_unique_mask = torch.where(
                rounded_ref == val, rounded_ref / val, 0)

            diff_rend_loss[val_idx-1] = torch.sum((rounded_image - rounded_ref)**2)
            # out_np = K.utils.tensor_to_image((image_unique_mask-ref_unique_mask)**2)
            # # out_np = K.utils.tensor_to_image(torch.movedim((model.image_ref - image[..., :3])**2, 3, 1))
            # plt.imshow(out_np); plt.savefig("/code/src/optim{:05d}-{}.png".format(i, val_idx))
        # if binary:
        #     d = (self.image_ref[..., 0] > 0).float() - image_est[..., 3]
        # else:  # per instance
        #     d = self.image_ref - image_est[..., :3]

        # if debug_flag:
        #     imsavePNG(image_est[:, :, :, 0], image_name_debug)

        if self.loss_func_num == 0:
            loss = torch.sum(torch.sum(d ** 2))
        elif self.loss_func_num == 1:
            # diff_rend_loss = torch.sum(torch.sum(d ** 2))
            signed_dis_loss = torch.max(signed_dis)
            # loss= torch.sum(torch.sum(d ** 2)) + torch.max(signed_dis) #torch.sum(intersect_point)
            diff_rend_loss = torch.sum(diff_rend_loss)
            loss= torch.sum(diff_rend_loss) + torch.max(signed_dis) #torch.sum(intersect_point)
            # loss= diff_rend_loss + torch.max(signed_dis) #torch.sum(intersect_point)
        # elif self.loss_func_num == 2:
        #     loss = torch.sum(torch.sum(d ** 2)) + torch.sum(torch.clamp_min(-signed_dis, 0))
        # elif self.loss_func_num == 3:
        #     loss = torch.sum(torch.sum((d[d > 0]) ** 2)) / torch.sum(torch.sum(self.image_ref))
        # elif self.loss_func_num == 4:
        #     loss_difference = torch.sum(d[d > 0] ** 2)  # sum(squared difference within reference mask) -> 0.. if no difference, at most N[umber of pixels] in reference
        #     loss_difference = loss_difference / torch.sum(self.image_ref)  # div by N -> [0,1]... 1 if complete outside
        #     loss_outside = torch.sum(-d[-d > 0])  # number of pixels in estimated mask (that are not in reference mask) -> at most M (pixels in estimate)
        #     loss_outside = loss_outside / torch.sum(image_est)  # div by M -> [0,1]... 1 if completely outside
        #     loss = (loss_difference + loss_outside) * 100
        # elif self.loss_func_num == 5:
        #     loss_difference = torch.sum(d[d > 0] ** 2)  # sum(squared difference within reference mask) -> 0.. if no difference, at most N[umber of pixels] in reference
        #     loss_difference = loss_difference / torch.sum(self.image_ref)  # div by N -> [0,1]... 1 if complete outside
        #     loss_outside = torch.sum(-d[-d > 0])  # number of pixels in estimated mask (that are not in reference mask) -> at most M (pixels in estimate)
        #     loss_outside = loss_outside / torch.sum(image_est)  # div by M -> [0,1]... 1 if completely outside
        #     loss_sign = signed_dis.clip(-torch.inf, 0).mean() / signed_dis.min() #TODO: this is zero! Why ?! What should I do?
        #     loss = (loss_difference + loss_outside + loss_sign) * 100
        # elif self.loss_func_num == 6:
        #     contour_diff = contour.contour_loss(
        #         torch.sum(image_est, dim=-1),
        #         # image_depth_est,
        #         ref_rgb_tensor,
        #         image_name_debug,
        #         debug_flag
        #     )
        #     contour_loss = torch.sum(
        #         contour_diff[contour_diff > 0])/(contour_diff.shape[-2] * contour_diff.shape[-1])


        #     diff_rend_loss = torch.sum(torch.sum(d ** 2))
        #     signed_dis_loss = torch.max(signed_dis)

        #     # print(contour_loss, signed_dis_loss)
        #     # loss = signed_dis_loss + contour_loss

        #     loss = diff_rend_loss + signed_dis_loss + contour_loss *  0.0001 # 0.00001

        else:
            raise ValueError()

        return loss, image_est, None, diff_rend_loss, signed_dis_loss, contour_loss # signed_dis

    def evaluate_progress(self, T_igt_list, isbop): # TODO: for checking
        # note: we use the [[R, t], [0, 1]] convention here -> transpose all matrices
        T_est_list = [T.transpose(-2, -1) for T in self.get_transform()]
        T_res_list = [T_igt_list[i].transpose(-2, -1) @ T_est_list[i] for i in range(len(T_igt_list))]  # T_est @ T_igt

        # metrics_list = []
        # metrics_str_list = []
        metrics_dict = {
            'R_iso': [],  # [deg] error between GT and estimated rotation
            't_iso': [],  # [mm] error between GT and estimated translation
            'ADD_abs': [],  # [mm] average distance between model points
            'ADI_abs': [],  # -//- nearest model points
            'ADD': [],  # [%] of model diameter
            'ADI': [],  # -//- nearest model points
        }

        for i in range(len(self.meshes)):
            T_res = T_res_list[i]

            # isometric errors
            R_trace = T_res[:, 0, 0] + T_res[:, 1, 1] + T_res[:, 2, 2]  # note: torch.trace only supports 2D matrices
            R_iso = torch.rad2deg(torch.arccos(torch.clamp(0.5 * (R_trace - 1), min=-1.0, max=1.0)))
            metrics_dict["R_iso"].append(float(R_iso))
            t_iso = torch.norm(T_res[:, :3, 3])
            metrics_dict["t_iso"].append(float(t_iso))
            # ADD/ADI error #TODO : number of samples for each object in the scene. if not equal => wrong

            if isbop:
                diameters = self.meshes_diameter[i]
                mesh_pytorch = self.meshes[i]
                # create from numpy arrays
                d_mesh = open3d.geometry.TriangleMesh(
                    vertices=open3d.utility.Vector3dVector(
                        mesh_pytorch.verts_list()[0].cpu().detach().numpy().copy()),
                    triangles=open3d.utility.Vector3iVector(
                        mesh_pytorch.faces_list()[0].cpu().detach().numpy().copy()))
                simple = d_mesh.simplify_quadric_decimation(
                    int(9000))
                mesh = torch.from_numpy(np.asarray(simple.vertices))[None, ...].type(torch.FloatTensor).to(device)
            else:
                mesh = self.meshes[i].verts_list()[0][None, ...]
                diameters = torch.sqrt(square_distance(mesh, mesh).max(dim=-1)[0]).max(dim=-1)[0]

            mesh_off = (T_res[:, :3, :3] @ mesh.transpose(-1, -2)).transpose(-1, -2) + T_res[:, :3, 3][:, None, :]
            dist_add = torch.norm(mesh - mesh_off, p=2, dim=-1).mean(dim=-1)
            dist_adi = torch.sqrt(square_distance(mesh, mesh_off)).min(dim=-1)[0].mean(dim=-1)
            metrics_dict["ADD_abs"].append(float(dist_add) * 1000)
            # TODO: fix here
            metrics_dict["ADI_abs"].append(float(dist_adi) * 1000)
            metrics_dict["ADD"].append(float(dist_add / diameters) * 100)
            metrics_dict["ADI"].append(float(dist_adi / diameters) * 100)
        # TODO: Is it ok to convert them to float? not being a tensor anymore
        metrics = {
            'R_iso': np.mean(np.asarray(metrics_dict["R_iso"])),  # [deg] error between GT and estimated rotation
            't_iso': np.mean(np.asarray(metrics_dict["t_iso"])),  # [mm] error between GT and estimated translation
            'ADD_abs': np.mean(np.asarray(metrics_dict["ADD_abs"])),  # [mm] average distance between model points
            'ADI_abs': np.mean(np.asarray(metrics_dict["ADI_abs"])),  # -//- nearest model points
            'ADD': np.mean(np.asarray(metrics_dict["ADD"])),  # [%] of model diameter
            'ADI': np.mean(np.asarray(metrics_dict["ADI"])),  # -//- nearest model points
        }
        # TODO: Fix the metrics_str
        metrics_str = f"R={metrics['R_iso']:0.1f}deg, t={metrics['t_iso']:0.1f}mm\n" \
                      f"ADD={metrics['ADD_abs']:0.1f}mm ({metrics['ADD']:0.1f}%)\n" \
                      f"ADI={metrics['ADI_abs']:0.1f}mm ({metrics['ADI']:0.1f}%)"

        return metrics, metrics_str

    def visualize_progress(self, background, text=""):
        # estimate
        R, t = self.get_R_t()
        image_est = self.ren_vis(meshes_world=self.meshes, R=R, T=t)
        estimate = image_est[0, ..., -1].detach().cpu().numpy()  # [0, 1]
        silhouette = estimate > 0

        # visualization
        vis = background[..., :3].copy()
        # add estimated silhouette
        vis *= 0.5
        vis[..., 2] += estimate * 0.5
        # # add estimated contour
        # contour, _ = cv.findContours(np.uint8(silhouette[..., -1] > 0), cv.RETR_CCOMP, cv.CHAIN_APPROX_TC89_L1)
        # vis = cv.drawContours(vis, contour, -1, (0, 0, 255), 1, lineType=cv.LINE_AA)
        if text != "":
            # add text
            rect = cv.rectangle((vis * 255).astype(np.uint8), (0, 0), (250, 100), (167, 168, 168), -1)
            vis = ((vis * 0.5 + rect/255 * 0.5) * 255).astype(np.uint8)
            font_scale, font_color, font_thickness = 0.5, (0, 0, 0), 1
            x0, y0 = 25, 25
            for i, line in enumerate(text.split('\n')):
                y = int(y0 + i * cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] * 1.5)
                vis = cv.putText(vis, line, (x0, y),
                                 cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv.LINE_AA)
        return vis


def square_distance(pcd1, pcd2):
    # via https://discuss.pytorch.org/t/fastest-way-to-find-nearest-neighbor-for-a-set-of-points/5938/13
    r_xyz1 = torch.sum(pcd1 * pcd1, dim=2, keepdim=True)  # (B,N,1)
    r_xyz2 = torch.sum(pcd2 * pcd2, dim=2, keepdim=True)  # (B,M,1)
    mul = torch.matmul(pcd1, pcd2.permute(0, 2, 1))  # (B,M,N)
    return r_xyz1 - 2 * mul + r_xyz2.permute(0, 2, 1)
