import numpy as np
import torch

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


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


def plane_contact_loss(points, plane_normal, plane_pt, margin_col=0.01, margin_cont=0.05):
    plane_distance = torch.einsum ('ijk, k -> ij', points - plane_pt, plane_normal)
    critical_pts = plane_distance.min(dim=-1).values

    # Assymetric L2 and Huber loss
    huber_func_col = torch.nn.SmoothL1Loss(reduction='mean', beta=margin_col)  # 0.01
    huber_func_cont = torch.nn.HuberLoss(reduction='mean', delta=margin_cont)   # 0.05
    mask_col = critical_pts < 0
    mask_cont = critical_pts > 0

    target = torch.zeros(critical_pts.shape, device=device)

    return huber_func_col(mask_col * critical_pts, target) + huber_func_cont(mask_cont * critical_pts, target)
