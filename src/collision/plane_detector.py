import numpy as np
# from skimage.transform import resize
import cv2
import open3d as o3d


class PlaneDetector:

    def __init__(self, width, height, intrinsics, to_meters=1e-3, distance_threshold=0.005):
        self.width, self.height = int(width), int(height)
        self.vmap, self.umap = np.meshgrid(range(self.width), range(self.height))

        self.intrinsics = intrinsics
        self.to_meters = to_meters
        self.distance_threshold = distance_threshold

    def detect(self, C, D, intrinsics, max_dist=1.0):
        # adapt intrinsics
        cam_fx, cam_fy = intrinsics[0, 0], intrinsics[1, 1]
        cam_cx, cam_cy = intrinsics[0, 2], intrinsics[1, 2]

        # === project to point cloud in XYZRGB format, Nx6
        pt2 = D * self.to_meters
        pt0 = (self.vmap - cam_cx) * pt2 / cam_fx
        pt1 = (self.umap - cam_cy) * pt2 / cam_fy
        points = np.dstack((pt0, pt1, pt2, C)).astype(np.float32).reshape((self.width * self.height, 6))

        # remove invalid points
        z_values = pt2.reshape((self.width * self.height))
        points = points[np.logical_and(z_values > 0, z_values < max_dist)]
        if points.shape[0] == 0:
            raise ValueError("no points left to detect plane")

        # === use Open3D for plane segmentation
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[:, :3]))
        cloud.colors = o3d.utility.Vector3dVector(points[:, 3:6])
        coefficients, indices = cloud.segment_plane(
            self.distance_threshold,
            ransac_n=3,
            num_iterations=1000)
        plane, scene = cloud.select_by_index(indices), cloud.select_by_index(indices, invert=True)
        # plane, scene = cloud.select_down_sample(indices), cloud.select_down_sample(indices, invert=True)  # open3d==0.9.0.0

        if len(coefficients) == 0:
            raise ValueError("no coefficients for plane - none detected")

        # === plane coefficients to transformation matrix: adapted from https://math.stackexchange.com/a/1957132
        # R: compute basis vectors from n
        if coefficients[2] > 0:  # make sure we're facing camera
            coefficients = [-c for c in coefficients]

        n = np.array(coefficients[:3]) / np.linalg.norm(coefficients[:3])
        nxy_norm = np.linalg.norm(n[:2])
        R = np.eye(3)
        # - b1 vector orthogonal to n
        R[0, 0] = n[1] / nxy_norm
        R[1, 0] = -n[0] / nxy_norm
        R[2, 0] = 0
        # - b2: vector normal to n and b1 -- n x b1 (-> right-handed)
        R[0, 1] = n[0] * n[2] / nxy_norm
        R[1, 1] = n[1] * n[2] / nxy_norm
        R[2, 1] = -nxy_norm
        # - b3: the normal vector itself
        R[:, 2] = n[:3]

        # t: move -d in direction of n
        t = -n * coefficients[3]
        centroid_in_plane = (R @ (points[:, :3]-t).T).T.mean(axis=0)
        centroid_in_plane[2] = 0  # only xy
        t += R @ centroid_in_plane

        # compose final matrix
        T = np.eye(4, 4)
        T[:3, :3] = R
        T[:3, 3] = t  # to mm

        return T, plane, scene, cloud, indices
