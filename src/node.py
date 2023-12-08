import argparse
import numpy as np
import os
import torch
import yaml
from pose.model import OptimizationModel
import pose.object_pose as object_pose
from collision.plane_detector import PlaneDetector
from src.contour.contour import compute_sdf_image
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



SUPPORTED_OBJECTS = [
    'container'
]

INTERNAL_TO_PROJECT_NAMES = {
    'container': 'Canister',
}

PROJECT_TO_INTERNAL_NAMES = {
    v: k for (k, v) in INTERNAL_TO_PROJECT_NAMES.items()
}


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

        self.viz_pub = rospy.Publisher(f"/{name}/debug_visualization", Image, queue_size=10, latch=True)

        self.plane_model = None

        self.scale = self.camera_info.width // 640
        self.intrinsics = np.array(self.camera_info.K).reshape(3,3)
        self.intrinsics[:2, :] //= self.scale

        # TODO: set from rosparam:
        self.cfg = cfg

        # load all meshes that will be needed
        self.cmap = plt.cm.tab20(range(20)) # 20 different colors, two consecutive ones are similar (for two instances)
        self.meshes, self.sampled_down_meshes = object_pose.load_objects_models(
            SUPPORTED_OBJECTS, cfg.objects_path,
            cmap=self.cmap, mesh_num_samples=cfg.mesh_num_samples)

        # Renderer


        # Optimization model
        self.model = OptimizationModel(
            None,
            self.intrinsics,
            self.camera_info.width // self.scale, self.camera_info.height // self.scale,
            cfg.optim,
            image_scale=self.scale).to(device)

        # create server
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

    def callback(self, goal):

        print(1, "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)
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
        intrinsics = self.intrinsics.copy()

        cv2.imwrite("/code/debug/input_depth.png", depth)
        cv2.imwrite("/code/debug/input_color.png", rgb)

        if self.scale != 1:
            rgb = cv2.resize(rgb, (rgb.shape[1] // self.scale, rgb.shape[0] // self.scale))

            bounding_boxes = [
                [bbox[i] // self.scale for i in range(4)]
                for bbox in bounding_boxes
            ]

            depth = cv2.resize(depth, (depth.shape[1] // self.scale, depth.shape[0] // self.scale)) # TODO: check size to see if scaling is necessary

        reference_height, reference_width = rgb.shape[:2]

        t_mag = 1
        scene_name = "ros_test_scene"
        logger = Logger(log_dir=os.path.join(config.PATH_REPO, f"logs/{scene_name}/loss_num_{1}/{self.cfg.optim.optimizer_name}"),
                                    log_name=f"{scene_name}_opt_{self.cfg.optim.optimizer_name}_lr_{self.cfg.optim.learning_rate}",
                                    reset_num_timesteps=True)

        print("------- Scene number: ", scene_name)
        # For each image in the scene

        # prepare events
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # record start
        start.record()

        im_id = 1 # range(1, number_of_scene_image+1)
        # for im_id in range(1, number_of_scene_image+1): # range(1, number_of_scene_image+1)
        # im_id = 37
        print(f"im {im_id}: optimizing...")


        # Extract plane =======================================================
        if self.plane_model is None:
            plane_det = PlaneDetector(
                reference_width, reference_height,
                to_meters=1e-3,
                distance_threshold=0.01)
            T, plane, scene, cloud, indices = plane_det.detect(
                rgb, depth, intrinsics, max_dist=1.0)

            # Create the filtered scene depth image
            scene_depth = project_point_cloud(
                scene, intrinsics, reference_height, reference_width)

            plane_depth = project_point_cloud(
                plane, intrinsics, reference_height, reference_width)

        # Filter detection based on depth =====================================
        perc_valid_depth_per_det = []
        det_mean_dist = []
        for bbox in bounding_boxes:
            bbox_depth = scene_depth[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # print("Bboxes & depth", bbox, bbox_depth)
            perc_valid_depth_per_det.append(
                np.mean(bbox_depth != 0))
            det_mean_dist.append(np.mean(bbox_depth))

        valid_det_mask = np.array(perc_valid_depth_per_det) > 0.1

        print("valid_det_mask", valid_det_mask)

        scene_objects = list(np.array(scene_objects)[valid_det_mask])
        init_poses = list(np.array(init_poses)[valid_det_mask])
        bounding_boxes = list(np.array(bounding_boxes)[valid_det_mask])

        self.publish_viz(rgb, scene_objects, bounding_boxes, init_poses, [0.5 for _ in scene_objects], intrinsics)

        #TODO order detections
        #TODO filter supported objects?
        #TODO filter out detections where the silhouette doesn't overlap with the detection

        # Prepare target silhouettes ==========================================
        masks = self.create_masks_from_bounding_boxes(bounding_boxes, reference_height, reference_width)
        # masks2 = self.refine_masks_with_scene_depth(masks, scene_depth)
        # masks3 = [
        #     np.logical_and(mask.astype(bool), plane_depth == 0)
        #     for mask in masks
        # ]


        # Create colored instance mask
        reference_mask = np.zeros((reference_height, reference_width, 3))
        # reference_mask2 = np.zeros((reference_height, reference_width, 3))
        # reference_mask3 = np.zeros((reference_height, reference_width, 3))
        for mask_idx, mask in enumerate(masks):
            reference_mask[mask] = self.cmap[mask_idx, :3]
            # reference_mask2[masks2[mask_idx]] = self.cmap[mask_idx, :3]
            # reference_mask3[masks3[mask_idx]] = self.cmap[mask_idx, :3]


        plt.imshow(reference_mask); plt.savefig("/code/debug/ref_mask.png")
        # plt.imshow(reference_mask2); plt.savefig("/code/debug/ref_mask2.png")
        # plt.imshow(reference_mask3); plt.savefig("/code/debug/ref_mask3.png")
        plt.imshow(scene_depth); plt.savefig("/code/debug/ref_depth.png")
        # plt.imshow(plane_depth); plt.savefig("/code/debug/ref_plane_depth.png")

        # Set up optimization =================================================
        # create scene geometry from known meshes
        object_names, object_counts = np.unique(scene_objects, return_counts=True)

        assert object_counts.max() <= 3  # only 2 instances atm
        counter = dict(zip(object_names, [0]*len(object_names)))
        scene_meshes = []
        scene_sampled_mesh = []
        scene_obj_names = [] # For storing the name of the objects
        for object_name in scene_objects:
            i = counter[object_name]
            counter[object_name] += 1
            mesh = self.meshes[f'{object_name}-{i}']
            scene_meshes.append(mesh)
            scene_sampled_mesh.append(self.sampled_down_meshes[f'{object_name}-{i}'])
            scene_obj_names.append(object_name)

        # Adding the properties to the model
        self.model.renderer.meshes = scene_meshes
        self.model.sampled_meshes = scene_sampled_mesh
        self.model.renderer.meshes_name = scene_obj_names

        self.model.plane_T_matrix = torch.from_numpy(T).type(torch.FloatTensor).to(device)
        self.model.plane_pcd = plane


        T_init_list = [torch.from_numpy(pose[None, ...]).to(device)
                     for pose in init_poses]
        print(2, "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)

        # Perform optimization ================================================
        best_metrics, iter_values = object_pose.optimization_step(
            self.model, rgb, scene_depth, masks, T_init_list, None, self.cfg.optim,
            #self.optimizer_name, self.max_num_iterations, self.early_stopping_loss, self.lr,
            logger, im_id, self.debug_flag, f"debug/{scene_name}/loss_num_{1}/{self.cfg.optim.optimizer_name}/{self.cfg.optim.learning_rate}",
            isbop=False)

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
        # torch.cuda.memory._dump_snapshot("my_snapshot.pickle")


        logger.close()

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
        result.object_types = scene_objects
        result.confidences = [1. for _ in scene_objects]

        print("LAST", "Mem allocated", torch.cuda.memory_allocated(0)/1024**2)

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
