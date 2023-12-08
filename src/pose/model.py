
import torch
import torch.nn as nn
import open3d as o3d

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

from src.collision import transformations as tra
import numpy as np
import cv2 as cv
from src.contour import contour
from pose.renderer import Renderer


import matplotlib.pyplot as plt
import kornia as K

class OptimizationModel(nn.Module):
    def __init__(self, meshes, intrinsics, width, height, cfg, image_scale=1, BOP=False):
        super().__init__()
        self.meshes = meshes
        self.meshes_name = None
        self.sampled_meshes = None
        self.device = device  # TODO : should I check the device for all the objects ? Or assume that they are all set for cuda?
        self.cfg = cfg

        self.meshes_diameter = None

        # Plane (Table in this dataset) point clouds and transformation matrix
        self.plane_pcd = None
        self.plane_T_matrix = None

        # Set up renderer
        self.renderer = Renderer(meshes, intrinsics, width, height, representation=cfg.pose_representation)

        self.image_ref = None  # Image mask reference

    def init(self, image_ref, T_init_list, T_plane=None): # TODO : Did I do it correctly here?
        # self.image_ref = torch.from_numpy(image_ref.astype(np.float32)).to(device)
        self.renderer.init(T_init_list)

    def signed_dis(self, k=10):
        """
        Calculating the signed distance of an object and the plane (The table)
        :return: two torch arrays with the length of number of points, showing the distance of that point and whether it
        is an intersection point or not
        """
        points = torch.cat([torch.tensor(self.sampled_meshes[i][None, ...], dtype=torch.float ,device=device) for i in range(len(self.sampled_meshes))],dim=0)# shape (num of meshes, num point in each obj , 6 (coordinates and norms))
        estimated_trans_matrixes = torch.cat([T.transpose(-2, -1) for T in self.renderer.get_transform()], dim=0) # Transposed because of the o3d and pytorch difference

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

    def get_R_t(self):
        return self.renderer.get_R_t()

    def forward(self, ref_rgb, ref_depth, ref_masks, debug_flag):
        print("MODEL", 1, "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)
        # Render the silhouette using the estimated pose
        image_est, depth_est, obj_masks, fragments_est = self.renderer()

        loss = torch.zeros(1).to(device)
        contour_loss = None
        print("MODEL", 2, "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)

        # Silhouette loss -----------------------------------------------------
        if self.cfg.losses.silhouette_loss.active:
            diff_rend_loss = torch.zeros(len(ref_masks)).to(device)
            for mask_idx, ref_mask in enumerate(ref_masks):
                image_unique_mask = torch.where(
                    obj_masks[mask_idx] > 0, image_est[..., 3], 0)


                union = ((image_unique_mask + ref_mask) > 0).float()
                diff_rend_loss[mask_idx] = torch.sum(
                    (image_unique_mask - ref_mask)**2
                ) / torch.sum(union) # Corresponds to 1 - IoU

                if debug_flag:
                    out_np = K.utils.tensor_to_image((image_unique_mask - ref_mask)**2)
                    plt.imshow(out_np); plt.savefig("/code/debug/mask-{}.png".format(mask_idx))

            diff_rend_loss = torch.sum(diff_rend_loss)
            loss += diff_rend_loss
        else:
            diff_rend_loss = None


        print("MODEL", 3, "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)

        # Depth loss ----------------------------------------------------------
        if self.cfg.losses.depth_loss.active:
            ref_depth_tensor = torch.from_numpy(
                ref_depth.astype(np.float32)).to(device)
            # ref_depth_tensor *= (self.image_ref[..., 0] > 0).float()
            # d_depth = (ref_depth_tensor - depth_est)

            # depth = torch.gather(zbuf, 0, depth_indices[..., None])
            d_depth = (depth_est - ref_depth_tensor[None, ..., None])[depth_est > 0]

            # depth_loss = torch.sum(d_depth**2) / torch.sum((zbuf > -1).float())
            depth_loss = torch.sum(d_depth**2) / (d_depth.shape[0])
        else:
            depth_loss = None

        if debug_flag:
            out_np = K.utils.tensor_to_image(depth_est)
            plt.imshow(out_np); plt.savefig("/code/debug/depth_est.png")

        # Collision loss ------------------------------------------------------
        if self.cfg.losses.collision_loss.active:
            # Calculating the signed distance
            signed_dis, intersect_point = self.signed_dis()
            signed_dis_loss = torch.max(signed_dis)
            loss += torch.max(signed_dis_loss)
        else:
            signed_dis_loss = None

        return loss, image_est, None, diff_rend_loss, signed_dis_loss, contour_loss, depth_loss # signed_dis

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
                d_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(
                        mesh_pytorch.verts_list()[0].cpu().detach().numpy().copy()),
                    triangles=o3d.utility.Vector3iVector(
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
