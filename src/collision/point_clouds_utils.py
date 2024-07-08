import numpy as np
import torch


def project_point_cloud(cloud, intrinsics, reference_height, reference_width):
    cloud_pts = np.asarray(cloud.points)
    us = cloud_pts[:, 0] / cloud_pts[:, 2] * intrinsics[0, 0] + intrinsics[0, 2]
    vs = cloud_pts[:, 1] / cloud_pts[:, 2] * intrinsics[1, 1] + intrinsics[1, 2]

    depth_map = np.zeros((reference_height, reference_width))
    us, vs = us.astype(int), vs.astype(int)
    depth_map[vs, us] = cloud_pts[:, 2]

    return depth_map


def plane_pt_intersection_along_ray(ray, pts, plane_normal, plane_pt):
    # pts = np.array(pts)
    if len(pts.shape) == 2:
        # Choose the ray that has the smallest dot product
        # to the plane normal (signed)
        closest_pt_idx = torch.argmin(
            torch.einsum ('ij, j -> i', (pts - plane_pt), plane_normal))
        closest_pt = pts[closest_pt_idx]
    elif len(pts.shape) == 1:
        assert pts.shape[0] == 3
        closest_pt = pts
    else:
        raise Exception('Invalid shape of pts. Received {} but only accepts 1 dimension (a single 3D point) or 2 dimensions (an array of pts)'.format(pts.shape))

    # Normalize ray
    ray /= torch.linalg.vector_norm(ray)

    # Compute the vector along `ray` to add to `closest_pt` to intersect with the plane
    pt_to_plane_vec = torch.dot(plane_normal, plane_pt - closest_pt) / torch.dot(plane_normal, ray) * ray

    return pt_to_plane_vec
