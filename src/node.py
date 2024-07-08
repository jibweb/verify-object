import argparse
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

from pose.model import OptimizationModel
import pose.object_pose as object_pose
from collision.plane_detector import PlaneDetector
from collision.point_clouds_utils import plane_pt_intersection_along_ray
from contour.contour import compute_sdf_image
from utility.logger import Logger
from config import config
from matplotlib import pyplot as plt
import cv2
from collision.point_clouds_utils import project_point_cloud

from actionlib import SimpleActionServer
import rospy
import ros_numpy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from tracebot_msgs.msg import VerifyObjectAction, VerifyObjectGoal, VerifyObjectResult

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)  # To check whether we have nan or inf in our gradient calculation


INTERNAL_TO_PROJECT_NAMES = {
    1: "MediumBottle",
    2: "SmallBottle",
    3: "Needle",
    4: "NeedleCap",
    5: "RedPlug",
    6: "Canister",
    7: "LargeBottle",
    8: "YellowPlug",
    9: "WhiteClamp",
    10: "RedClamp",
    # 'container': 'Canister',
}


PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}


SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()


class VerifyPose:

    def __init__(self, name, cfg, debug_flag=False, intrinsics_from_file=False):
        self.debug_flag = debug_flag

        # Reading camera intrinsics
        if intrinsics_from_file:
            camera_intr_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml')
            intrinsics_yaml = yaml.load(open(camera_intr_path, 'r'), Loader=yaml.FullLoader)
            self.camera_info = CameraInfo()
            self.camera_info.K = intrinsics_yaml['camera_matrix']
            self.camera_info.height = intrinsics_yaml["image_height"]
            self.camera_info.width = intrinsics_yaml["image_width"]
        else:
            self.camera_info_topic = rospy.get_param('/locateobject/camera_info_topic',
                                                    '/camera/color/camera_info')
            rospy.loginfo(f"[{name}] Waiting for camera info ...")
            self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
            rospy.loginfo(f"[{name}] Camera info received")

        self.viz_pub = rospy.Publisher(f"{name}/debug_visualization", Image, queue_size=10, latch=True)

        self.plane_normal, self.plane_pt = None, None

        self.scale = self.camera_info.width // 640
        self.intrinsics = np.array(self.camera_info.K).reshape(3,3)
        self.intrinsics[:2, :] /= self.scale

        print("Intrisics", np.array(self.camera_info.K).reshape(3,3), self.intrinsics)

        # TODO: set from rosparam:
        self.cfg = cfg

        # load all meshes that will be needed
        self.cmap = plt.cm.tab20(range(20)) # 20 different colors, two consecutive ones are similar (for two instances)
        meshes, sampled_down_meshes = object_pose.load_objects_models(
            ["obj_{:06d}".format(obj_id) for obj_id in SUPPORTED_OBJECTS],
            cfg.objects_path,
            cmap=self.cmap,
            mesh_num_samples=cfg.mesh_num_samples)

        # Optimization model
        self.model = OptimizationModel(
            meshes,
            sampled_down_meshes,
            self.intrinsics,
            self.camera_info.width // self.scale,
            self.camera_info.height // self.scale,
            cfg.optim,
            debug=self.debug_flag,
            debug_path=self.cfg.debug_path)

        # Create server
        self._server = SimpleActionServer(name, VerifyObjectAction, execute_cb=self.callback, auto_start=False)
        self._server.start()
        rospy.loginfo(f"[{name}] Action Server ready")

    def create_masks_from_bounding_boxes(self, bounding_boxes, height, width):
        masks = []
        for bbox in bounding_boxes:
            mask = np.zeros((height, width))
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            masks.append(mask.astype(bool))

        return masks

    def refine_masks_with_scene_depth(self, masks, depth):
        return [
            np.logical_and(mask.astype(bool), depth != 0)
            for mask in masks
        ]

    def refine_masks_with_grabcut(self, ref_rgb, masks):
        kernel = np.ones((5,5),np.uint8)
        refined_masks = []
        for mask in tqdm(masks, desc='Mask refinement with GrabCut'):
            obj_mask = mask.astype(np.uint8)
            erosion = cv2.erode(obj_mask, kernel, iterations=1)
            dilation = cv2.dilate(obj_mask, kernel, iterations=1)
            new_mask = np.zeros(obj_mask.shape, dtype=np.uint8)
            new_mask[dilation.astype(bool)] = 2
            new_mask[erosion.astype(bool)] = 1

            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)

            gb_mask, bgdModel, fgdModel = cv2.grabCut(
                ref_rgb, new_mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

            refined_masks.append(
                np.where((gb_mask==2)|(gb_mask==0),False,True)
            )

        return refined_masks

    def callback(self, goal):
        # Parse goal message ==================================================
        scene_objects = [PROJECT_TO_INTERNAL_NAMES[scene_obj]
            for scene_obj in goal.object_types]
        rgb = ros_numpy.numpify(goal.color_image)[..., ::-1]
        depth = ros_numpy.numpify(goal.depth_image)
        init_poses = [ros_numpy.numpify(pose).astype(np.float32) for pose in goal.object_poses]
        bounding_boxes = [
            (bbox.x_offset, bbox.y_offset,
             bbox.x_offset + bbox.width, bbox.y_offset + bbox.height)
            for bbox in goal.bounding_boxes]
        if len(goal.object_masks) != 0:
            masks = [ros_numpy.numpify(mask).astype(bool)
                     for mask in goal.object_masks]

        intrinsics = self.intrinsics.copy()

        if self.debug_flag:
            cv2.imwrite(os.path.join(self.cfg.debug_path, "input_depth.png"), depth)
            cv2.imwrite(os.path.join(self.cfg.debug_path, "input_color.png"), rgb)

        # Scale images ========================================================
        if self.scale != 1:
            rgb = cv2.resize(rgb, (rgb.shape[1] // self.scale, rgb.shape[0] // self.scale))

            bounding_boxes = [
                [bbox[i] // self.scale for i in range(4)]
                for bbox in bounding_boxes
            ]

            if len(goal.object_masks) != 0:
                masks = [cv2.resize(
                            mask.astype(np.float32),
                            (mask.shape[1] // self.scale, mask.shape[0] // self.scale)
                         ).astype(bool)
                         for mask in masks]

            depth = cv2.resize(depth, (depth.shape[1] // self.scale, depth.shape[0] // self.scale)) # TODO: check size to see if scaling is necessary

        reference_height, reference_width = rgb.shape[:2]

        # prepare events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # record start
        start.record()

        # Extract plane =======================================================
        if self.plane_normal is None:
            plane_det = PlaneDetector(
                reference_width, reference_height,
                to_meters=1e-3,
                distance_threshold=0.01)
            plane_T, plane, scene, cloud, indices = plane_det.detect(
                rgb, depth, intrinsics, max_dist=1.5)

            # Create the filtered scene depth image
            scene_depth = project_point_cloud(
                scene, intrinsics, reference_height, reference_width)

            plane_depth = project_point_cloud(
                plane, intrinsics, reference_height, reference_width)

            self.plane_normal = torch.from_numpy(
                plane_T[:,2][:3].astype(np.float32)).to(device)
            self.plane_pt = torch.from_numpy(
                np.asarray(plane.points[0].astype(np.float32))).to(device)

        # Filter detection based on depth =====================================
        perc_valid_depth_per_det = []
        det_mean_dist = []
        for bbox in bounding_boxes:
            bbox_depth = scene_depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # print("Bboxes & depth", bbox, bbox_depth)
            perc_valid_depth_per_det.append(
                np.mean(bbox_depth != 0))
            det_mean_dist.append(np.mean(bbox_depth[bbox_depth != 0]))

        valid_det_mask = np.array(perc_valid_depth_per_det) > 0.1

        print("valid_det_mask", valid_det_mask)

        scene_objects = list(np.array(scene_objects)[valid_det_mask])
        init_poses = list(np.array(init_poses)[valid_det_mask])
        bounding_boxes = list(np.array(bounding_boxes)[valid_det_mask])

        self.publish_viz(rgb, scene_objects, bounding_boxes, init_poses, [0.5 for _ in scene_objects], intrinsics)

        # Prepare target silhouettes ==========================================
        if len(goal.object_masks) == 0:
            masks = self.create_masks_from_bounding_boxes(
                bounding_boxes, reference_height, reference_width)
        else:
            masks = list(np.array(masks)[valid_det_mask])

        if self.cfg.mask_grabcut_refinement:
            masks = self.refine_masks_with_grabcut(rgb, masks)

        # Create colored instance mask
        reference_mask = np.zeros((reference_height, reference_width, 3))
        for mask_idx, mask in enumerate(masks):
            reference_mask[mask] = self.cmap[mask_idx, :3]

        if self.debug_flag:
            cv2.imwrite(os.path.join(self.cfg.debug_path, "ref_mask.png"),
                        (0.7*255*reference_mask + 0.3*rgb).astype(np.uint8))
            cv2.imwrite(os.path.join(self.cfg.debug_path, "ref_rgb.png"), rgb)
            plt.imshow(scene_depth); plt.savefig(os.path.join(self.cfg.debug_path, "ref_depth.png"))

        # Set up optimization =================================================
        self.model.plane_pcd = plane
        T_init_list = [torch.from_numpy(pose[None, ...]).to(device)
                     for pose in init_poses]

        self.model.init(
            scene_objects,
            T_init_list,
            T_plane=torch.from_numpy(plane_T.astype(np.float32)))

        # Correct poses to align with plane ===================================
        for obj_idx, mask in enumerate(masks):
            # Prepare meshes
            rot, trans = T_init_list[obj_idx][:, :3, :3], T_init_list[obj_idx][:, :3, 3]
            mesh = self.model.renderer.scene_meshes[obj_idx]
            verts_trans = \
                ((rot @ mesh.verts_padded()[..., None]) + trans[..., None])[..., 0]

            # Get mask coordinates
            indices_v, indices_u = torch.nonzero(
                torch.from_numpy(mask),
                as_tuple=True)

            # Convert mask pixels to rays
            x = (indices_u - self.model.renderer.intrinsics[0 ,2]) / self.model.renderer.intrinsics[0,0]
            y = (indices_v - self.model.renderer.intrinsics[1,2]) / self.model.renderer.intrinsics[1,1]
            z = torch.ones(x.shape)

            # Create average ray
            ray = torch.Tensor(
                [x[:, None].mean(), y[:, None].mean(), z[:, None].mean()]).to(device)

            # Compute the difference to apply along ray to ensure plane contact
            vec_to_plane = plane_pt_intersection_along_ray(
                ray,
                verts_trans[0],
                self.plane_normal,
                self.plane_pt)

            # Adapt the initial pose
            T_init_list[obj_idx][:, :3, 3] += vec_to_plane

        # Re-init the model with the corrected filtered poses
        self.model.init(
            scene_objects,
            T_init_list,
            T_plane=torch.from_numpy(plane_T.astype(np.float32)))

        # Perform optimization ================================================
        best_metrics, iter_values = object_pose.optimization_step(
            self.model, rgb, scene_depth, masks, self.cfg.optim, None,
            self.debug_flag, self.cfg.debug_path, isbop=False)

        best_R_list, best_t_list = self.model.get_R_t()
        predicted_poses = []
        for best_R, best_t in zip(best_R_list, best_t_list):
            pose = np.eye(4)
            pose[:3, :3] = best_R.to('cpu').detach().numpy()[0]
            pose[:3, 3] = best_t.to('cpu').detach().numpy()[0]
            predicted_poses.append(pose)

        self.publish_viz(rgb, scene_objects, bounding_boxes, predicted_poses, [0.5 for _ in scene_objects], intrinsics)

        end.record()
        torch.cuda.synchronize()
        # get time between events (in ms)
        print("____Timing for the whole scene:______", start.elapsed_time(end))

        result = VerifyObjectResult()
        result.object_poses = [
            ros_numpy.msgify(Pose, pose) for pose in predicted_poses
        ]
        result.header = goal.header
        # result.bounding_boxes = bounding_boxes
        # for mask in obj_masks: #TODO obtain updated masks from renderer
        #     us, vs = np.nonzero(mask)
        #     bbox = RegionOfInterest()
        #     bbox.x_offset = vs.min()
        #     bbox.y_offset = us.min()
        #     bbox.width = vs.max() - vs.min()
        #     bbox.height = us.max() - us.min()
        #     result.bounding_boxes.append(bbox)
        for box in bounding_boxes:
            bbox = RegionOfInterest()
            bbox.x_offset = box[0]
            bbox.y_offset = box[1]
            bbox.width = box[2] - box[0]
            bbox.height = box[3] - box[1]
            goal.bounding_boxes.append(bbox)
        result.object_types = [INTERNAL_TO_PROJECT_NAMES[name]
                               for name in scene_objects]
        result.confidences = [1. for _ in scene_objects]

        self._server.set_succeeded(result)

    def publish_viz(self, rgb, obj_names, bboxes, poses, scores, intrinsics):
        viz_img = rgb.copy()

        for bbox in bboxes:
            cv2.rectangle(viz_img,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        for o_idx, obj_pose in enumerate(poses):
            rospy.loginfo("{}: {}".format(obj_names[o_idx], scores[o_idx]))
            rospy.loginfo("{}".format(obj_pose))

            rvec, _ = cv2.Rodrigues(obj_pose[:3, :3])
            cv2.drawFrameAxes(viz_img,
                              intrinsics,
                              np.zeros(5),
                              rvec, obj_pose[:3, 3], 0.08)


        data = ros_numpy.msgify(Image, viz_img, encoding='8UC3')
        data.header.frame_id = self.camera_info.header.frame_id
        data.header.stamp = self.camera_info.header.stamp
        self.viz_pub.publish(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tracebot project -- Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    parser.add_argument('--intrinsics_from_file', dest='intrinsics_from_file', default=False, action='store_true')
    args = parser.parse_args()

    cfg = config.GlobalConfig()

    rospy.init_node('verify_object')
    node = VerifyPose(
        rospy.get_name(),
        cfg,
        debug_flag=args.debug,
        intrinsics_from_file=args.intrinsics_from_file)
    rospy.spin()
