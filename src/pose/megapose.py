import cv2
import json
import numpy as np
import open3d as o3d
import os
from scipy.spatial.transform import Rotation
import subprocess
from time import time
import yaml

MEGAPOSE_DIR = "/home/jbweibel/code/megapose6d"
IN_HAND_DIR = os.path.join(MEGAPOSE_DIR, 'data/examples/tracebot_inhand/')

class MegaposeRefiner:
    def __init__(self, cfg, intrinsics, width, height, objects_names,
                 objects_to_optimize, scale=1000):
        camera_data = {
            "K": intrinsics.tolist(),
            "resolution": [height, width]
        }

        with open(os.path.join(IN_HAND_DIR, 'camera_data.json'), 'w') as fp:
                json.dump(camera_data, fp)

        self.objects_to_optimize = objects_to_optimize

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

        self.obj_pcds = {}
        for oi, mesh_name in enumerate(meshes_fns):
            print("Loading object", objects_names[oi], mesh_name)
            # Load mesh
            obj_mesh = o3d.io.read_triangle_mesh(os.path.join(cfg.objects_path, mesh_name)).scale(1./scale, np.array([0.,0.,0.]))
            self.obj_pcds[objects_names[oi]] = obj_mesh.sample_points_uniformly(number_of_points=2000)
            self.obj_pcds[objects_names[oi]].estimate_normals()


    def optimize(self, rgb, depth, scene_objects, init_poses, ref_masks):
        start = time()
        cv2.imwrite(os.path.join(IN_HAND_DIR, 'image_rgb.png'), rgb)
        cv2.imwrite(os.path.join(IN_HAND_DIR, 'image_depth.png'), depth)

        predicted_poses = []
        for obj_idx, obj_name in enumerate(scene_objects):
            if not self.objects_to_optimize[obj_name]:
                 continue

            mask = ref_masks[obj_idx]
            ys, xs = np.nonzero(mask)

            # Write up detection
            detections = [{
                "label": obj_name,
                "bbox_modal": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            }]

            with open(os.path.join(IN_HAND_DIR, 'inputs', 'object_data.json'), 'w') as fp:
                json.dump(detections, fp)

            # Run megapose
            # source /conda/bin/activate && python -m megapose.scripts.run_inference_on_example tracebot_inhand --run-inference --init-pose "0.98613097 -0.0745836 0.14826675 0.21597066 0.07227096 -0.61122766 -0.78814825 -0.19318982 0.14940768 0.78793282 -0.5973603 0.98283673 0. 0. 0. 1."
            # subprocess.run([

            # Run in another terminal:
            # docker run -it --gpus all --rm -e MEGAPOSE_DATA_DIR=/code/data -v /home/jbweibel/code/megapose6d:/code --name megapose_exp megapose-runner bash
            # docker run -it --gpus all --rm -e MEGAPOSE_DATA_DIR=/code/data -v /home/jbweibel/code/megapose6d:/code --name megapose_exp megapose-runner /bin/bash -c 'source /conda/bin/activate && python -m megapose.scripts.run_inference_on_example tracebot_inhand --run-inference'
            init_pose_str = ' '.join(init_poses[obj_idx].flatten().astype(str).tolist())
            print(init_pose_str)
            subprocess.run(
                # "docker run -it --gpus all " \
                # "-e MEGAPOSE_DATA_DIR=/code/data " \
                # "-v {}:/code megapose-runner " \
                "docker exec -it megapose_exp " \
                "/bin/bash -c 'python3 /code/src/megapose/scripts/run_client.py --init-pose \"{}\"'".format(init_pose_str),
                # "/bin/bash -c 'source /conda/bin/activate && python -m megapose.scripts.run_inference_on_example tracebot_inhand --run-inference --init-pose \"{}\"'".format(init_pose_str),
                shell=True
            )

            with open(os.path.join(IN_HAND_DIR, 'outputs', 'object_data.json')) as fp:
                output = json.load(fp)

            pose = np.eye(4)
            pose[:3,:3] = Rotation.from_quat(output[0]['TWO'][0]).as_matrix()
            pose[:3,3] = output[0]['TWO'][1]
            predicted_poses.append(pose)

        print('Optimization in', time()-start)

        return predicted_poses