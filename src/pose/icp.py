from copy import deepcopy
import numpy as np
import open3d as o3d
import os
import yaml


class ICPRefiner:
    def __init__(self, cfg, intrinsics, width, height, objects_names,
                 objects_to_optimize, scale=1000, debug_flag=False):
        if cfg.objects_loader == 'bop':
            meshes_fns = ["obj_{:06d}.obj".format(obj_id) for obj_id in objects_names]
        elif cfg.objects_loader == '3d-dat':
            with open(os.path.join(cfg.objects_path, "object_library.yaml")) as fp:
                objects_dict = yaml.safe_load(fp)

            if objects_names is not None:
                meshes_fns = [obj['mesh'] for obj in objects_dict if obj['id'] in objects_names]
                assert len(meshes_fns) == len(objects_names), "Corresponding models of some objects_names entry could not be found"
            else:
                meshes_fns = [obj['mesh'] for obj in objects_dict]
                objects_names = [obj['id'] for obj in objects_dict]
        else:
            raise Exception("Unknown object loader type, no object loaded. Valid are ['bop', 3d-dat]")

        self.pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            intrinsics[0,0],
            intrinsics[1,1],
            intrinsics[0,2],
            intrinsics[1,2]
        )

        self.debug = debug_flag

        self.obj_pcds = {}
        for oi, mesh_name in enumerate(meshes_fns):
            print("Loading object", objects_names[oi], mesh_name)
            # Load mesh
            obj_mesh = o3d.io.read_triangle_mesh(os.path.join(cfg.objects_path, mesh_name)).scale(1./scale, np.array([0.,0.,0.]))
            self.obj_pcds[objects_names[oi]] = obj_mesh.sample_points_uniformly(number_of_points=cfg.mesh_num_samples)
            self.obj_pcds[objects_names[oi]].estimate_normals()

        self.objects_to_optimize = objects_to_optimize

    def optimize(self, rgb, depth, scene_objects, init_poses, ref_masks):
        rgb_o3d = o3d.geometry.Image(rgb)

        if self.debug:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                o3d.geometry.Image(depth),
                convert_rgb_to_intensity=False)
            scene_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, self.pinhole_intrinsics)

        optimized_poses = []
        for obj_idx, obj_name in enumerate(scene_objects):
            if not self.objects_to_optimize[obj_name]:
                optimized_poses.append(init_poses[obj_idx])
                continue

            filt_depth = np.zeros(depth.shape, dtype=depth.dtype)
            filt_depth[ref_masks[obj_idx]] = depth[ref_masks[obj_idx]]

            filt_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d,
                o3d.geometry.Image(filt_depth),
                convert_rgb_to_intensity=False)
            filt_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                filt_rgbd, self.pinhole_intrinsics)
            # filt_pcd.estimate_normals()
            obj_pcd = deepcopy(self.obj_pcds[obj_name]).transform(init_poses[obj_idx])
            # o3d.visualization.draw_geometries([obj_pcd, filt_pcd])
            reg_p2p = o3d.pipelines.registration.registration_icp(
                filt_pcd, obj_pcd, 0.05, np.eye(4),
                # o3d.pipelines.registration.TransformationEstimationPointToPlane())
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1.000000e-06,
                    relative_rmse=1.000000e-06,
                    max_iteration=300))
            optimized_pose = np.dot(
                np.linalg.inv(reg_p2p.transformation),
                init_poses[obj_idx],
            )
            optimized_poses.append(optimized_pose)

            opt_obj_pcd = deepcopy(self.obj_pcds[obj_name]).transform(optimized_pose)
            opt_obj_pcd.colors = o3d.utility.Vector3dVector(0.8*np.ones(np.asarray(opt_obj_pcd.points).shape))

            if self.debug:
                o3d.visualization.draw_geometries([obj_pcd, opt_obj_pcd, scene_pcd])

        return optimized_poses