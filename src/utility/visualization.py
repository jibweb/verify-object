import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
import torch
import cv2 as cv
import yaml
from pytorch3d.io import load_obj
from skimage import img_as_ubyte
from skimage.transform import resize
from pytorch3d.structures import Meshes
import pytorch3d.structures as pyst
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes, join_meshes_as_scene, join_meshes_as_batch
from src.pose.model import OptimizationModel
import open3d as o3d
# from src.pose.environment import scene_point_clouds
# from src.pose.environment import scene_point_clouds
import src.config as config
import matplotlib as mpl
from matplotlib import pyplot as plt


"""
Installing opencv-python-headless for cv2 because then matplotlib and opencv with have a problem with qt!
"""
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
cmap = plt.cm.tab20(range(20))


def visualization_progress(model, background_image, R_list, t_list, text=""):
    """

    :param model:
    :param background_image:
    :param r_t_dict:
    :param device:
    :param text:
    :return:
    """
    # TODO: Should I put the [0]? How can I get rid of it ?
    # R_list = torch.stack([torch.from_numpy(np.asarray(r_list)).to(model.device)[0] for r_list in r_objs_list])
    # t_list = torch.stack([torch.from_numpy(np.asarray(t_list)).to(model.device)[0] for t_list in t_objs_list])

    meshes_transformed = []
    for mesh, mesh_r, mesh_t in zip(model.meshes, R_list.transpose(-2, -1), t_list):
        new_verts_padded = \
            ((mesh_r @ mesh.verts_padded()[..., None]) + mesh_t[..., None])[..., 0]
        mesh = mesh.update_padded(new_verts_padded)
        meshes_transformed.append(mesh)
    scene_transformed = join_meshes_as_scene(meshes_transformed)

    image_est = model.ren_vis(
        meshes_world=scene_transformed,
        R=torch.eye(3)[None, ...].to(device),
        T=torch.zeros((1, 3)).to(device)
    )

    # image_est, fragments_est = model.ren_opt(meshes_world=scene_transformed,
    #                                                 R=torch.eye(3)[None, ...].to(device),
    #                                                 T=torch.zeros((1, 3)).to(device))
    image_est = image_est.sum(0)
    estimate = image_est[..., -1].detach().cpu().numpy()  # [0, 1]

    # visualization
    vis = background_image[..., :3].copy()
    vis *= 0.5
    vis[..., 2] += estimate * 0.5

    if text != "":
        # add text
        rect = cv.rectangle((vis * 255).astype(np.uint8), (0, 0), (250, 100), (167, 168, 168), -1)
        vis = ((vis * 0.5 + rect / 255 * 0.5) * 255).astype(np.uint8)
        font_scale, font_color, font_thickness = 0.5, (0, 0, 0), 1
        x0, y0 = 25, 25
        for i, line in enumerate(text.split('\n')):
            y = int(y0 + i * cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] * 1.5)
            vis = cv.putText(vis, line, (x0, y),
                             cv.FONT_HERSHEY_SIMPLEX, font_scale, font_color, font_thickness, cv.LINE_AA)

    # cv.imshow("Display window", vis)
    # k = cv.waitKey(0)
    plt.imshow(vis[..., ::-1])
    plt.show()

    return vis


def result_visualization_single_image(model, source_image_rgb, source_image_mask, scale, r_t_per_iter, best_metrics, visualization_saving_path, image_saving_path):
    """
    Visualizing the detection for a single image
    :param model: Object model, OptimizationModel type
    :param source_image_rgb:
    :param source_image_mask:
    :param scale: #TODO : should also save the scale in the files to avoid mistakes !
    :param r_t_per_iter:
    :param best_metrics:
    :param visualization_saving_path:
    :param image_saving_path:
    :return:
    """

    reference_height, reference_width = source_image_rgb.shape[:2]

    if scale != 1:
        reference_width //= scale
        reference_height //= scale
        source_image_rgb = resize(source_image_rgb[..., :3], (reference_height, reference_width))
        source_image_mask = resize(source_image_mask, (reference_height, reference_width))

    visualization_buffer = []
    source_image_rgb[source_image_mask > 0, 1] = source_image_rgb[source_image_mask > 0, 1] * 0.5 \
                                           + source_image_mask[source_image_mask > 0] * 0.5

    t = 0
    for iter_num in range(len(r_t_per_iter["r"])):
        t = iter_num
        visualization = visualization_progress(model, source_image_rgb, r_t_per_iter["r"][iter_num], r_t_per_iter["t"][iter_num], f"Iteration {iter_num + 1:03d}: loss={r_t_per_iter['loss'][iter_num]:0.1f}\n")
        visualization_buffer.append(img_as_ubyte(visualization))
    # TODO: Check this to be sure all good
    metrics_str = f"R={best_metrics['R_iso']:0.1f}deg, t={best_metrics['t_iso']:0.1f}mm\n" \
                  f"ADD={best_metrics['ADD_abs']:0.1f}mm ({best_metrics['ADD']:0.1f}%)\n" \
                  f"ADI={best_metrics['ADI_abs']:0.1f}mm ({best_metrics['ADI']:0.1f}%)"

    model.init(source_image_mask, [torch.from_numpy(np.asarray(r_t_per_iter["best_T"][i]))[None, ...].to(model.device).transpose(-2, -1) for i in range(len(r_t_per_iter["best_T"]))])
    visualization = visualization_progress(model, source_image_rgb, r_t_per_iter["r"][t], r_t_per_iter["t"][t], f"Best:\n{metrics_str}")
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(visualization_buffer[0])
    plt.axis('off')
    plt.title("initialization")
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.axis('off')
    plt.title(f"best ADI")
    plt.tight_layout()
    plt.savefig(image_saving_path)
    plt.close()

    reference_height, reference_width = source_image_rgb.shape[:2]
    # """
    visualization_buffer = [
        (resize(vis, (reference_height // 16 * 16, reference_width // 16 * 16)) * 255).astype(np.uint8)
        for vis in visualization_buffer]  # note: mp4 requires size to be multiple of macro block
    imageio.mimsave(visualization_saving_path, visualization_buffer, fps=5, quality=10)
    # """
    return


def scenes_mesh_reader(objects_path, datasets_path, scenes_id):
    """
    Reading the objects' meshes in one single scene with the same order of the pose.yaml file
    :param objects_path:
    :param datasets_path:
    :param scenes_id:
    :return:
    """

    meshes_dic = {} # dictionary to save the meshes for all the scenes
    scenes_obj_names_dic = {}
    for scene_id in scenes_id:

        scenes_obj_names_dic[f'{scene_id:03d}'] = []
        objects_poses_dic = yaml.load(open(os.path.join(datasets_path, f'{scene_id:03d}/poses.yaml'), 'r'), Loader=yaml.FullLoader)
        for object_dic in objects_poses_dic:
            obj_name = object_dic['id']
            scenes_obj_names_dic[f'{scene_id:03d}'].append(obj_name)
            if obj_name not in meshes_dic.keys():

                verts, faces_idx, _ = load_obj(os.path.join(objects_path, f'{obj_name}/{obj_name}_simple.obj'))
                mesh = Meshes(
                    verts=[verts.to(device)],
                    faces=[faces_idx.verts_idx.to(device)],
                    # textures=textures
                )  # (N, V, F)
                meshes_dic[f'{obj_name}'] = mesh

    return meshes_dic, scenes_obj_names_dic


def result_visualization(model, scene_path, predictions_path, meshes_dic, scenes_obj_names_dic, scenes, loss_nums, lr_list, optimizers_list):
    """
    Visualizing the detection for the whole dataset
    :param model: Object model, OptimizationModel type
    :param scene_path: path to the scene images
    :param predictions_path: path to the detection results
    :param scenes: number of the annotated scenes
    :param loss_nums:
    :param lr_list:
    :param optimizers_list:
    :return:
    """

    for scene_id in scenes:
        for loss_num in loss_nums:
            for optimizer_name in optimizers_list:
                for lr in lr_list:
                    with open(os.path.join(predictions_path, f"{scene_id:03d}/loss_num_{loss_num}/{optimizer_name}/{scene_id:03d}_images_dic_lr_{lr}.yaml"), 'r') as stream:
                        try:
                            scene_iter_value = yaml.safe_load(stream)
                        except yaml.YAMLError as exc:
                            print(exc)

                    with open(os.path.join(predictions_path, f"{scene_id:03d}/loss_num_{loss_num}/{optimizer_name}/{scene_id:03d}_best_iters_lr_{lr}.yaml"), 'r') as st:
                        try:
                            best_in_iters = yaml.safe_load(st)
                        except yaml.YAMLError as exc:
                            print(exc)

                    saving_path = f"/home/negar/Documents/Tracebot/Files/negar-layegh-inverse-rendering/result/{scene_id:03d}/loss_num_{loss_num}/{optimizer_name}/vis_{lr}"

                    if not os.path.exists(saving_path):
                        os.makedirs(saving_path)

                    meshes = []
                    for obj_id in range(len(scenes_obj_names_dic[f'{scene_id:03d}'])):

                        obj_name = scenes_obj_names_dic[f'{scene_id:03d}'][obj_id]
                        mesh = meshes_dic[obj_name]
                        textures = TexturesVertex(
                            verts_features=torch.from_numpy(np.array(cmap[obj_id * 2 + obj_id][:3]))[None, None, :]
                            .expand(-1, mesh.verts_list()[0].shape[0], -1).type_as(mesh.verts_list()[0]).to(device))
                        mesh.textures = textures
                        meshes.append(mesh)


                    # If the number of objects' vertices is different:
                    # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/batching.md
                    # https://pytorch3d.readthedocs.io/en/latest/modules/structures.html#pytorch3d.structures.join_meshes_as_scene

                    model.meshes = pyst.join_meshes_as_batch(meshes)

                    # # Debugging
                    """
                    import open3d as o3d
                    pcd1 = o3d.geometry.PointCloud()
                    pcd1.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[0].cpu().detach().numpy()))
                    pcd1.paint_uniform_color([1, 0, 0])

                    pcd2 = o3d.geometry.PointCloud()
                    pcd2.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[1].cpu().detach().numpy()))
                    pcd2.paint_uniform_color([0, 1, 0])

                    pcd3 = o3d.geometry.PointCloud()
                    pcd3.points = o3d.utility.Vector3dVector((model.meshes.verts_list()[2].cpu().detach().numpy()))
                    pcd3.paint_uniform_color([0, 0, 1])
                    o3d.visualization.draw_geometries([pcd1, pcd2, pcd3], point_show_normal=True)
                    """

                    for image_name in np.sort(os.listdir(os.path.join(scene_path, f"{scene_id:03d}/rgb/"))):
                        reference_rgb = imageio.imread(os.path.join(scene_path, f"{scene_id:03d}/rgb/{image_name}")) # TODO: WHAT? V3?
                        reference_mask = 0
                        obj_list = scenes_obj_names_dic[f'{scene_id:03d}']
                        for obj_num, obj_name  in enumerate(obj_list):
                            reference_mask += imageio.imread(
                                os.path.join(scene_path, f"{scene_id:03d}/masks/003_{obj_name}_00{obj_num}_{image_name}"))

                        im_id = int(image_name[3:-4]) - 1
                        print(im_id)
                        if im_id == 1:
                            print("Ere ")

                        r_t_per_iter = scene_iter_value[im_id]
                        best_metrics = {}
                        for key in best_in_iters.keys():
                            best_metrics.setdefault(key)
                            best_metrics[key] = best_in_iters[key][im_id]

                        result_visualization_single_image(model, reference_rgb, reference_mask, 4, r_t_per_iter, best_metrics,
                                                          os.path.join(saving_path, f"{im_id + 1}_optimization.mp4"), os.path.join(saving_path, f"{im_id + 1}_result.png"))
    return 0


if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    scene_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'scenes')
    objects_path = os.path.join(config.PATH_DATASET_TRACEBOT, 'objects')
    predictions_path = os.path.join('/home/negar/Documents/Tracebot/Files/negar-layegh-inverse-rendering/result')
    os.makedirs(predictions_path, exist_ok=True)

    # Useful for all scenes
    # Reading camera intrinsics
    intrinsics_yaml = yaml.load(open(os.path.join(config.PATH_DATASET_TRACEBOT, 'camera_d435.yaml'), 'r'), Loader=yaml.FullLoader)

    representation = 'q'
    scenes = [3]  # list(range(1, 6))  # only the annotated subset
    loss_num = [1] # from 0 to 6
    lr_list = [
           0.015,
        0.02,
        #    0.04,
        #    0.06
    ]
    optimizers_list = ['adam']
    scenes_meshes_dic, scenes_obj_names_dic = scenes_mesh_reader(objects_path, scene_path, scenes)
    model = OptimizationModel(None, intrinsics_yaml, representation=representation, image_scale=4).to(device)
    result_visualization(model, scene_path, predictions_path, scenes_meshes_dic, scenes_obj_names_dic, scenes, loss_num, lr_list, optimizers_list)
