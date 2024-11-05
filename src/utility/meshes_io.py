import numpy as np
import os
import torch
import trimesh

from pytorch3d.io import load_obj, load_ply
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

def load_objects_models(mesh_names, objects_path, obj_idx_keys, cmap, mesh_num_samples=500, scale=1000):
    meshes = {}
    sampled_down_meshes = {}

    for oi, mesh_name in enumerate(mesh_names):
        print("Loading object", obj_idx_keys[oi], mesh_name)
        # Load mesh
        if mesh_name.endswith('obj'):
            verts, faces_fields, _ = load_obj(os.path.join(objects_path, f'{mesh_name}'))
            faces_idx = faces_fields.verts_idx
        elif mesh_name.endswith('ply'):
            verts, faces_idx = load_ply(os.path.join(objects_path, f'{mesh_name}'))

        textures = TexturesVertex(
            verts_features=torch.from_numpy(cmap[oi][:3])[None, None, :]
                                    .expand(-1, verts.shape[0], -1).type_as(verts))

        mesh = Meshes(
            verts=[verts/scale],
            faces=[faces_idx],
            textures=textures)

        meshes[obj_idx_keys[oi]] = mesh

        # Create a randomly point-normal set
        # Same number of points for each individual object
        mesh_sampled_down = trimesh.load(os.path.join(objects_path, f'{mesh_name}'))
        norms = mesh_sampled_down.face_normals
        samples = trimesh.sample.sample_surface_even(mesh_sampled_down, mesh_num_samples) # either exactly NUM_samples, or <= NUM_SAMPLES --> pad by random.choice
        samples_norms = norms[samples[1]] # Norms pointing out of the object
        samples_point_norm = np.concatenate((np.asarray(samples[0]/scale), np.asarray(0-samples_norms)), axis=1)
        if samples_point_norm.shape[0] < mesh_num_samples:  # NUM_SAMPLES not equal to mesh_num_samples -> padding
            idx = np.random.choice(samples_point_norm.shape[0], mesh_num_samples - samples_point_norm.shape[0])
            samples_point_norm = np.concatenate((samples_point_norm, samples_point_norm[idx]), axis=0)

        sampled_down_meshes[obj_idx_keys[oi]] = torch.from_numpy(samples_point_norm.astype(np.float32))[None, ...]

    return meshes, sampled_down_meshes
