import argparse
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

from pose_refiner import RefinePose
from config import config
from matplotlib import pyplot as plt
import cv2

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

OBJECTS_TO_OPTIMIZE = {
    "MediumBottle": False,
    "SmallBottle": False,
    "Needle": True,
    "NeedleCap": True,
    "RedPlug": True,
    "Canister": True,
    "LargeBottle": False,
    "YellowPlug": True,
    "WhiteClamp": True,
    "RedClamp": True,
}

PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}

OBJECTS_TO_OPTIMIZE_INTERNAL = {
    PROJECT_TO_INTERNAL_NAMES[name]: val
    for name, val in OBJECTS_TO_OPTIMIZE.items()
}

SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()


class ROSPoseVerifier(RefinePose):

    def __init__(self, name, cfg, debug_flag=False):
        # Reading camera intrinsics
        self.camera_info_topic = rospy.get_param('/locateobject/camera_info_topic',
                                                '/camera/color/camera_info')
        rospy.loginfo(f"[{name}] Waiting for camera info ...")
        self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        rospy.loginfo(f"[{name}] Camera info received")

        self.viz_pub = rospy.Publisher(f"{name}/debug_visualization", Image, queue_size=10, latch=True)

        with open('/code/config/silhouette_plane.yml') as fp:
            self.plane_params = yaml.safe_load(fp)
        with open('/code/config/silhouette_contact.yml') as fp:
            self.inhand_params = yaml.safe_load(fp)

        assert self.plane_params['resolution'] == self.inhand_params['resolution'], "In-hand and on-plane configuration must use the same resolution"
        cfg.from_dict(self.plane_params)

        # TODO get those values from rosparam
        # self.plane_normal, self.plane_pt = None, None

        print("Pre-scaling Intrisics", np.array(self.camera_info.K).reshape(3,3))
        super().__init__(
            cfg=cfg, # TODO set from rosparam
            intrinsics=np.array(self.camera_info.K).reshape(3,3),
            objects_names=SUPPORTED_OBJECTS,
            objects_to_optimize=OBJECTS_TO_OPTIMIZE_INTERNAL,
            width=self.camera_info.width,
            height=self.camera_info.height,
            debug_flag=debug_flag)

        # Create server
        self._server = SimpleActionServer(name, VerifyObjectAction, execute_cb=self.callback_plane, auto_start=False)
        self._server.start()

        self._server_inhand = SimpleActionServer(name + '_inhand', VerifyObjectAction, execute_cb=self.callback_inhand, auto_start=False)
        self._server_inhand.start()

        rospy.loginfo(f"[{name}] Action Server ready")

    def callback_plane(self, goal):
        self.cfg.from_dict(self.plane_params)
        self.model.cfg = self.cfg.optim

        # TODO: re-initialize renderer appropriately

        result = self.generic_callback(goal)
        self._server.set_succeeded(result)

    def callback_inhand(self, goal):
        self.cfg.from_dict(self.inhand_params)
        self.model.cfg = self.cfg.optim

        # TODO: re-initialize renderer appropriately

        # TODO: Obtain poses from DT ?
        # TODO: Obtain masks from Yolov8 somehow (filtered only for hand result)
        # TODO: Obtain contact points information
        # TODO: Add gripper CAD in the scene

        result = self.generic_callback(goal)
        self._server_inhand.set_succeeded(result)

    def create_masks_from_bounding_boxes(self, bounding_boxes, height, width):
        masks = []
        for bbox in bounding_boxes:
            mask = np.zeros((height, width))
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            masks.append(mask.astype(bool))

        return masks

    def generic_callback(self, goal):
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
        else:
            masks = self.create_masks_from_bounding_boxes(
                bounding_boxes, rgb.shape[0] // self.scale, rgb.shape[1] // self.scale)

        predicted_poses, ref_rgb, ref_depth, masks, viz_img = super().optimize(
            rgb, depth, scene_objects, init_poses, masks)

        self.publish_viz(viz_img)

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

        return result

    def publish_viz(self, rgb):
        viz_img = rgb[:,:,::-1].copy()
        data = ros_numpy.msgify(Image, viz_img, encoding='8UC3')
        data.header.frame_id = self.camera_info.header.frame_id
        data.header.stamp = self.camera_info.header.stamp
        self.viz_pub.publish(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tracebot project -- Pose refinement and verification using differentiable rendering')
    parser.add_argument('--debug', dest='debug', default=False, action='store_true')
    args = parser.parse_args()

    cfg = config.GlobalConfig()

    rospy.init_node('verify_object')
    node = ROSPoseVerifier(
        rospy.get_name(),
        cfg,
        debug_flag=args.debug)
    rospy.spin()
