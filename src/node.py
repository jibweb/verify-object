import argparse
import copy
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
import yaml

from pose.pose_refiner import RefinePose
from config import config
from matplotlib import pyplot as plt
import cv2

from actionlib import SimpleActionServer
import rospy
import ros_numpy
import tf2_ros
from geometry_msgs.msg import PoseArray, Pose, TransformStamped
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
    11: "NeedleNeedleCap",
    # 'container': 'Canister',

    # CEA Gripper
    12: "distal_phalanx",
    13: "intermediate_phalanx",
    14: "proximal_phalanx",
}

OBJECTS_TO_OPTIMIZE = {
    "MediumBottle": True,
    "SmallBottle": False,
    "Needle": True,
    "NeedleCap": True,
    "RedPlug": True,
    "Canister": True,
    "LargeBottle": False,
    "YellowPlug": True,
    "WhiteClamp": True,
    "RedClamp": True,
    "NeedleNeedleCap": True,

    # CEA Gripper
    "distal_phalanx": False,
    "intermediate_phalanx": False,
    "proximal_phalanx": False,
}

STABLE_AXIS = {
    "MediumBottle": [0,0,1],
    "SmallBottle": [0,0,1],
    "Needle": [1,0,0],
    "NeedleCap": [1,1,0],
    "RedPlug": [0,0,0],
    "Canister": [1,1,1],
    "LargeBottle": [0,0,1],
    "YellowPlug": [0,0,0],
    "WhiteClamp": [0,0,0],
    "RedClamp": [0,0,0],
    "NeedleNeedleCap": [0,0,0],

    # CEA Gripper
    "distal_phalanx": [0,0,0],
    "intermediate_phalanx": [0,0,0],
    "proximal_phalanx": [0,0,0],
}

PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}

OBJECTS_TO_OPTIMIZE_INTERNAL = {
    PROJECT_TO_INTERNAL_NAMES[name]: val
    for name, val in OBJECTS_TO_OPTIMIZE.items()
}

STABLE_AXIS_INTERNAL = {
    PROJECT_TO_INTERNAL_NAMES[name]: val
    for name, val in STABLE_AXIS.items()
}

SUPPORTED_OBJECTS = INTERNAL_TO_PROJECT_NAMES.keys()

RELATIVE_POSES = {
    'cap_to_needle': (np.eye(3), np.zeros(3))
}

GRIPPER_JOINTS = [
    "tracebot_right_gripper_distal_phalanx_1",
    "tracebot_right_gripper_distal_phalanx_2",
    "tracebot_right_gripper_distal_phalanx_3",
    "tracebot_right_gripper_distal_phalanx_4",
    "tracebot_right_gripper_intermediate_phalanx_1",
    "tracebot_right_gripper_intermediate_phalanx_2",
    "tracebot_right_gripper_intermediate_phalanx_3",
    "tracebot_right_gripper_intermediate_phalanx_4",
    "tracebot_right_gripper_proximal_phalanx_1",
    "tracebot_right_gripper_proximal_phalanx_2",
    "tracebot_right_gripper_proximal_phalanx_3",
    "tracebot_right_gripper_proximal_phalanx_4",
]


def get_bbox_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : list
        ['x1', 'x2', 'y1', 'y2']
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : list
        ['x1', 'x2', 'y1', 'y2']
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class ROSPoseVerifier(RefinePose):

    def __init__(self, name, cfg, debug_flag=False):
        # Reading camera intrinsics
        self.camera_info_topic = rospy.get_param('/locateobject/camera_info_topic',
                                                '/tracebot_camera/color/camera_info')
        self.contact_pts_left_topic = '/tracebot_left_gripper_contacts_camera_frame'
        self.contact_pts_right_topic = '/tracebot_right_gripper_contacts_camera_frame'
        rospy.loginfo(f"[{name}] Waiting for camera info ...")
        self.camera_info = rospy.wait_for_message(self.camera_info_topic, CameraInfo)
        rospy.loginfo(f"[{name}] Camera info received")
        self.det_threshold = rospy.get_param(
                '/locateobject/detection_threshold',
                0.5)

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)

        self.viz_pub = rospy.Publisher(f"{name}/debug_visualization", Image, queue_size=10, latch=True)
        self.contact_pts = None

        with open('/code/config/silhouette_plane.yml') as fp:
            self.plane_params = yaml.safe_load(fp)
        with open('/code/config/silhouette_contact.yml') as fp:
            self.inhand_params = yaml.safe_load(fp)

        assert self.plane_params['resolution'] == self.inhand_params['resolution'], "In-hand and on-plane configuration must use the same resolution"
        cfg.from_dict(self.plane_params)

        plane_normal = np.array(
            rospy.get_param('/locateobject/plane_normal', []), dtype=np.float32)
        if len(plane_normal) != 0:
            plane_normal /= np.linalg.norm(plane_normal)
        plane_pt = np.array(
            rospy.get_param('/locateobject/plane_pt', []), dtype=np.float32)

        print("Pre-scaling Intrisics", np.array(self.camera_info.K).reshape(3,3))
        super().__init__(
            cfg=cfg,
            intrinsics=np.array(self.camera_info.K).reshape(3,3),
            objects_names=list(SUPPORTED_OBJECTS),
            objects_to_optimize=OBJECTS_TO_OPTIMIZE_INTERNAL,
            width=self.camera_info.width,
            height=self.camera_info.height,
            plane_normal=plane_normal if len(plane_normal) != 0 else None,
            plane_pt=plane_pt if len(plane_pt) != 0 else None,
            stable_axis=STABLE_AXIS_INTERNAL,
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

        result = self.generic_callback(goal)
        self._server.set_succeeded(result)

    def callback_inhand(self, goal):
        self.cfg.from_dict(self.inhand_params)
        self.model.cfg = self.cfg.optim

        contact_pts = None
        if goal.header.frame_id == 'left':
            try:
                contact_pts = rospy.wait_for_message(self.contact_pts_left_topic, PoseArray, 3)
                contact_pts = contact_pts.poses
            except Exception as e:
                rospy.logwarn(e)
        elif goal.header.frame_id == 'right':
            try:
                contact_pts = rospy.wait_for_message(self.contact_pts_right_topic, PoseArray, 3)
                contact_pts = contact_pts.poses
            except Exception as e:
                rospy.logwarn(e)

        if contact_pts is not None and len(contact_pts) != 0:
            self.contact_pts = np.array([[pose.position.x, pose.position.y, pose.position.z] for pose in contact_pts]).astype(np.float32)

        # Add gripper CAD in the scene
        try:
            empty_mask = ros_numpy.msgify(
                Image,
                np.zeros((goal.color_image.height, goal.color_image.width), dtype=np.float32),
                encoding='32FC1',
            )
            for joint in GRIPPER_JOINTS:
                trans = self.tfBuffer.lookup_transform("tracebot_right_arm_tool0", joint, rospy.Time(), rospy.Duration(5))

                goal.object_types.append('_'.join(joint.split('_')[3:5]))
                joint_pose = Pose()
                joint_pose.position.x = trans.transform.translation.x
                joint_pose.position.y = trans.transform.translation.y
                joint_pose.position.z = trans.transform.translation.z
                joint_pose.orientation = trans.transform.rotation
                goal.object_poses.append(joint_pose)
                goal.bounding_boxes.append(RegionOfInterest())
                goal.confidences = list(goal.confidences) + [1.]
                goal.object_masks.append(empty_mask)

        except Exception as e:
            print(e)

        result = self.generic_callback(goal)

        if result.object_types[0] in ['Needle', 'NeedleCap']:
            result.object_types = ['Needle', 'NeedleCap']
            result.object_poses.append(result.object_poses[0])
            result.bounding_boxes.append(result.bounding_boxes[0]) # TODO correct bbox
            result.confidences.append(result.confidences[0])

        self.contact_pts = None
        self._server_inhand.set_succeeded(result)

    def create_masks_from_bounding_boxes(self, bounding_boxes, height, width):
        masks = []
        for bbox in bounding_boxes:
            mask = np.zeros((height, width))
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
            masks.append(mask.astype(bool))

        return masks

    def generic_callback(self, goal):
        plane_normal = rospy.get_param('/locateobject/plane_normal', [])
        if len(plane_normal) != 0:
            self.plane_normal = np.array(plane_normal, dtype=np.float32)
            self.plane_normal /= np.linalg.norm(self.plane_normal)
        plane_pt = rospy.get_param('/locateobject/plane_pt', [])
        if len(plane_pt) != 0:
            self.plane_pt = np.array(plane_pt, dtype=np.float32)

        # Parse goal message ==================================================
        scene_objects = [PROJECT_TO_INTERNAL_NAMES[scene_obj]
            for scene_obj in goal.object_types]

        rgb = ros_numpy.numpify(goal.color_image)[..., ::-1]
        depth = ros_numpy.numpify(goal.depth_image)
        init_poses = [ros_numpy.numpify(pose).astype(np.float32) for p_idx, pose in enumerate(goal.object_poses)]
        bounding_boxes = [
            (bbox.x_offset, bbox.y_offset,
             bbox.x_offset + bbox.width, bbox.y_offset + bbox.height)
            for b_idx, bbox in enumerate(goal.bounding_boxes)]
        confidences = [conf for c_idx, conf in enumerate(goal.confidences)]

        if len(goal.object_masks) != 0:
            masks = [ros_numpy.numpify(mask).astype(bool)
                     for m_idx, mask in enumerate(goal.object_masks)]
        else:
            masks = self.create_masks_from_bounding_boxes(
                bounding_boxes, rgb.shape[0] // self.scale, rgb.shape[1] // self.scale)

        # Boost detection results of nearby needle cap needle/white clamp (due to confusion)
        needle_indices = np.nonzero(np.array(scene_objects) == 3)[0]
        needle_cap_indices = np.nonzero(np.array(scene_objects) == 4)[0]
        white_clamp_indices = np.nonzero(np.array(scene_objects) == 9)[0]

        boosted_boxes = []
        relative_poses = {}
        for cap_idx in needle_cap_indices:
            for needle_idx in needle_indices:
                if get_bbox_iou(bounding_boxes[cap_idx], bounding_boxes[needle_idx]) > 0.01 and \
                   max([0] + [get_bbox_iou(bbox, bounding_boxes[needle_idx]) for bbox in boosted_boxes]) < 0.01 and \
                   (confidences[cap_idx] >= self.det_threshold or confidences[needle_idx] >= self.det_threshold):
                    scene_objects[needle_idx] = 11
                    confidences[cap_idx] = 0.  # max(confidences[cap_idx], self.det_threshold)
                    confidences[needle_idx] = max(confidences[needle_idx], self.det_threshold)
                    masks[needle_idx] += masks[cap_idx]
                    boosted_boxes.append(copy.copy(bounding_boxes[needle_idx]))
                    # relative_poses[(cap_idx, needle_idx)] = RELATIVE_POSES['cap_to_needle']

            # for white_clamp_idx in white_clamp_indices:
            #     if get_bbox_iou(bounding_boxes[cap_idx], bounding_boxes[white_clamp_idx]) > 0.01 and \
            #        max([0] + [get_bbox_iou(bbox, bounding_boxes[white_clamp_idx]) for bbox in boosted_boxes]) < 0.01 and \
            #        (confidences[cap_idx] >= self.det_threshold or confidences[white_clamp_idx] >= self.det_threshold):
            #         confidences[cap_idx] = max(confidences[cap_idx], self.det_threshold)
            #         confidences[white_clamp_idx] = max(confidences[white_clamp_idx], self.det_threshold)
            #         scene_objects[white_clamp_idx] = 3  # Transform the white clamp det into a needle det
            #         boosted_boxes.append(copy.copy(bounding_boxes[white_clamp_idx]))
            #         # relative_poses[(cap_idx, white_clamp_idx)] = RELATIVE_POSES['cap_to_needle']

        # Filter results according to confidence
        scene_objects = [obj for obj_idx, obj in enumerate(scene_objects)
                         if confidences[obj_idx] >= self.det_threshold]
        init_poses = [pose for obj_idx, pose in enumerate(init_poses)
                      if confidences[obj_idx] >= self.det_threshold]
        masks = [mask for obj_idx, mask in enumerate(masks)
                 if confidences[obj_idx] >= self.det_threshold]
        kept_indices = np.nonzero(np.array(confidences) >= self.det_threshold)[0]
        new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
        relative_poses = {
            (new_indices[o1], new_indices[o2]): (R,t) for (o1, o2), (R,t) in relative_poses.items()
        }
        confidences = [conf for conf in confidences if conf >= self.det_threshold]

        if len(scene_objects) != 0:
            predicted_poses, ref_rgb, ref_depth, masks, viz_img = super().optimize(
                rgb, depth, scene_objects, init_poses, masks,
                point_contacts=self.contact_pts, relative_poses=relative_poses)
        else:
            rospy.loginfo("No object above detection threshold")
            predicted_poses = []
            viz_img = rgb

        self.publish_viz(viz_img)

        result = VerifyObjectResult()
        result.header = goal.header
        for obj_idx, obj_name in enumerate(scene_objects):
            if obj_name == 11:
                result.object_types.append('Needle')
                result.object_poses.append(ros_numpy.msgify(Pose, predicted_poses[obj_idx]))
                bbox = RegionOfInterest() # TODO correct bounding box from renderer
                bbox.x_offset = bounding_boxes[obj_idx][0]
                bbox.y_offset = bounding_boxes[obj_idx][1]
                bbox.width = bounding_boxes[obj_idx][2] - bounding_boxes[obj_idx][0]
                bbox.height = bounding_boxes[obj_idx][3] - bounding_boxes[obj_idx][1]
                result.bounding_boxes.append(bbox)
                result.confidences.append(1.)
                obj_name = 4

            if obj_name in [2,7]: # Replacing Small and Large bottle detections with Medium ones
                obj_name = 1

            result.object_types.append(INTERNAL_TO_PROJECT_NAMES[obj_name])
            result.object_poses.append(ros_numpy.msgify(Pose, predicted_poses[obj_idx]))
            # result.bounding_boxes = bounding_boxes
            # for mask in obj_masks: #TODO obtain updated masks from renderer
            #     us, vs = np.nonzero(mask)
            #     bbox = RegionOfInterest()
            #     bbox.x_offset = vs.min()
            #     bbox.y_offset = us.min()
            #     bbox.width = vs.max() - vs.min()
            #     bbox.height = us.max() - us.min()
            #     result.bounding_boxes.append(bbox)
            bbox = RegionOfInterest()
            bbox.x_offset = bounding_boxes[obj_idx][0]
            bbox.y_offset = bounding_boxes[obj_idx][1]
            bbox.width = bounding_boxes[obj_idx][2] - bounding_boxes[obj_idx][0]
            bbox.height = bounding_boxes[obj_idx][3] - bounding_boxes[obj_idx][1]
            result.bounding_boxes.append(bbox)
            result.confidences.append(1.)

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
