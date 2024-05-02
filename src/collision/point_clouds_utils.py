import numpy as np


def project_point_cloud(cloud, intrinsics, reference_height, reference_width):
    cloud_pts = np.asarray(cloud.points)
    us = cloud_pts[:, 0] / cloud_pts[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]
    vs = cloud_pts[:, 1] / cloud_pts[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]

    depth_map = np.zeros((reference_height, reference_width))
    us, vs = us.astype(int), vs.astype(int)
    depth_map[vs, us] = cloud_pts[:, 2]

    return depth_map
