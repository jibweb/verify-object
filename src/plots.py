import argparse
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
sys.path.append('/home/jbweibel/code/3d-dat/')  # Adapt to your 3d-dat repository location
import v4r_dataset_toolkit as v4r


plt.style.use('ggplot')


OBJECT_NAMES = {
    'canister': 'Canister',
    'large_bottle': 'BigBottle',
    'medium_bottle': 'MediumBottle',
    'small_bottle': 'SmallBottle',
    'needle': '1'
}


OBJECTS_TO_OPTIMIZE = {
    "MediumBottle": True,
    "SmallBottle": True,
    "Needle": True,
    "NeedleCap": True,
    "RedPlug": True,
    "Canister": True,
    "BigBottle": True,
    "YellowPlug": True,
    "WhiteClamp": True,
    "RedClamp": True,
    "distal_phalanx": False,
    "intermediate_phalanx": False,
    "proximal_phalanx": False,
}


SCENE_PROPS = {
    'canister_1': {'trans': 'true', 'size': 'small'},
    'canister_2': {'trans': 'true', 'size': 'small'},
    'canister_textured_1': {'trans': 'true', 'size': 'small'},
    'canister_textured_2': {'trans': 'true', 'size': 'small'},
    'large_bottle_2fingers_1': {'trans': 'true', 'size': 'large'},
    'large_bottle_2fingers_2': {'trans': 'true', 'size': 'large'},
    'large_bottle_2fingers_textured_1': {'trans': 'true', 'size': 'large'},
    'large_bottle_2fingers_textured_2': {'trans': 'true', 'size': 'large'},
    'large_bottle_4fingers_1': {'trans': 'true', 'size': 'large'},
    'large_bottle_4fingers_2': {'trans': 'true', 'size': 'large'},
    'large_bottle_empty_1': {'trans': 'true', 'size': 'large'},
    'large_bottle_empty_2': {'trans': 'true', 'size': 'large'},
    'large_bottle_empty_textured_1': {'trans': 'true', 'size': 'large'},
    'large_bottle_empty_textured_2': {'trans': 'true', 'size': 'large'},
    'medium_bottle_4fingers_textured_1': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_4fingers_textured_2': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_empty_1': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_empty_2': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_empty_textured_1': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_empty_textured_2': {'trans': 'true', 'size': 'medium'},
    'medium_bottle_filled_1': {'trans': 'filled', 'size': 'medium'},
    'medium_bottle_filled_2': {'trans': 'filled', 'size': 'medium'},
    'medium_bottle_filled_textured_1': {'trans': 'filled', 'size': 'medium'},
    'medium_bottle_filled_textured_2': {'trans': 'filled', 'size': 'medium'},
    'medium_bottle_sanded_1': {'trans': 'false', 'size': 'medium'},
    'medium_bottle_sanded_2': {'trans': 'false', 'size': 'medium'},
    'medium_bottle_sanded_textured_1': {'trans': 'false', 'size': 'medium'},
    'medium_bottle_sanded_textured_2': {'trans': 'false', 'size': 'medium'},
    'needle_1': {'trans': 'true', 'size': 'small'},
    'needle_2': {'trans': 'true', 'size': 'small'},
    'needle_flip_1': {'trans': 'true', 'size': 'small'},
    'needle_flip_2': {'trans': 'true', 'size': 'small'},
    'needle_textured_1': {'trans': 'true', 'size': 'small'},
    'needle_textured_2': {'trans': 'true', 'size': 'small'},
    'small_bottle_2fingers_1': {'trans': 'true', 'size': 'small'},
    'small_bottle_2fingers_2': {'trans': 'true', 'size': 'small'},
    'small_bottle_empty_1': {'trans': 'true', 'size': 'small'},
    'small_bottle_empty_2': {'trans': 'true', 'size': 'small'},
    'small_bottle_empty_textured_1': {'trans': 'true', 'size': 'small'},
    'small_bottle_empty_textured_2': {'trans': 'true', 'size': 'small'},
    'small_bottle_filled_1': {'trans': 'filled', 'size': 'small'},
    'small_bottle_filled_2': {'trans': 'filled', 'size': 'small'},
    'small_bottle_filled_textured_1': {'trans': 'filled', 'size': 'small'},
    'small_bottle_filled_textured_2': {'trans': 'filled', 'size': 'small'},
}


pretty_poses = {
    'diffrend': "Silhouette"
}


pretty_masks = {
    'gt': 'groundtruth',
    'sam2': 'SAM2'
}


def append_img_results(res_path, global_results, scene_objects, exp_name):
    scene = res_path.split('/')[-4]
    fname = res_path.split('/')[-1]
    img_name = fname.split('.')[0].split('__')[0]
    depth_method = fname.split('.')[0].split('__')[1]
    mask_method = fname.split('.')[0].split('__')[2]
    pose_method = fname.split('.')[0].split('__')[3]

    if mask_method == 'None':
        mask_method = 'gt'

    for scene_obj_name, obj_name in OBJECT_NAMES.items():
        if scene_obj_name in scene:
            object_name = obj_name
            break

    with open(res_path) as fp:
        img_results = json.load(fp)

    import pdb;pdb.set_trace()

    for noise_idx, noise_sample in enumerate(img_results['noise_samples']):
        for obj_idx in range(len(img_results['gt_poses'])):
            if not OBJECTS_TO_OPTIMIZE[scene_objects[obj_idx]]:
                continue

            global_results['exp_name'].append(exp_name)
            global_results['scene'].append(scene)
            global_results['pre_adi'].append(img_results['pre_post_adi'][noise_idx][obj_idx][0])
            global_results['post_adi'].append(img_results['pre_post_adi'][noise_idx][obj_idx][1])
            global_results['mask_method'].append(mask_method)
            global_results['depth_method'].append(depth_method)
            global_results['pose_method'].append(pose_method)
            global_results['object_size'].append(img_results['object_sizes'][object_name])
            global_results['distance'].append(img_results['gt_poses'][obj_idx][2][3])
            global_results['object_scale'].append(SCENE_PROPS[scene]['size'])
            global_results['transparency'].append(SCENE_PROPS[scene]['trans'])
            global_results['textured'].append('textured' in scene)
            global_results['iter_nb'].append(img_results['iter_nb'][noise_idx])
            global_results['validity'].append(img_results['mask_validity'][obj_idx])
            global_results['silhouette_weight'].append(
                img_results['cfg']['optim']['losses']['silhouette_loss']['weight'] if img_results['cfg']['optim']['losses']['silhouette_loss']['active'] else 0.)
            global_results['contact_pts_weight'].append(
                img_results['cfg']['optim']['losses']['point_contact_loss']['weight'] if img_results['cfg']['optim']['losses']['point_contact_loss']['active'] else 0.)


def append_exp_results(global_results, scene_file_reader, exp_name):
    scenes_ids = scene_file_reader.get_scene_ids()
    for scene_id in scenes_ids:
        scene_path = os.path.join(
            scene_file_reader.root_dir,
            scene_file_reader.scenes_dir,
            scene_id)

        objects = scene_file_reader.get_object_poses(scene_id)
        scene_objects = [obj.id for (obj,pose) in objects]

        rgb_paths = v4r.io.get_file_list(
            os.path.join(scene_path, scene_file_reader.rgb_dir), ('.png',))
        rgb_paths.sort()

        result_paths = v4r.io.get_file_list(
            os.path.join(
                scene_path,
                'refinement_res',
                exp_name),
            ('.json',))

        for img_idx, res_path in enumerate(result_paths):
            append_img_results(res_path, global_results, scene_objects, exp_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pose refinement and verification using differentiable rendering')
    parser.add_argument('--dataset_cfg', type=str, default='/home/jbweibel/code/inverse_rendering/cea_data_collection/dataset/config.cfg')
    parser.add_argument('--exp_names', nargs='+', help='Pick experiences to include', required=True)
    parser.add_argument('-p','--pose_methods', nargs='+', help='Pick pose methods')
    parser.add_argument('-m','--mask_methods', nargs='+', help='Pick mask methods')
    args = parser.parse_args()

    scene_file_reader = v4r.io.SceneFileReader.create(args.dataset_cfg)

    results = {
        'exp_name':[],
        'scene':[],
        'pre_adi':[],
        'post_adi':[],
        'mask_method':[],
        'depth_method':[],
        'pose_method':[],
        'object_size':[],
        'distance': [],
        'object_scale':[],
        'transparency': [],
        'textured': [],
        'iter_nb': [],
        'validity': [],
        'silhouette_weight': [],
        'contact_pts_weight': [],
    }

    for exp_name in args.exp_names:
        print("Including results for", exp_name)
        append_exp_results(results, scene_file_reader, exp_name)
    df = pd.DataFrame.from_dict(results)

    pose_methods = args.pose_methods if args.pose_methods else pd.unique(df['pose_method'])
    mask_methods = args.mask_methods if args.mask_methods else pd.unique(df['mask_method'])
    # depth_methods = args.depth_methods if args.depth_methods else pd.unique(df['depth_method'])
    depth_methods = ['depth']

    legends = []
    for pose_method in pose_methods:
        for mask_method in mask_methods:
            for depth_method in depth_methods:
                print('Adding', pose_method, mask_method)
                subset = df.loc[
                    df['pose_method'] == pose_method
                ].loc[
                    df['mask_method'] == mask_method
                ].loc[
                    df['depth_method'] == depth_method
                ]

                legends.append(
                    "{} with {} masks".format(
                        pretty_poses[pose_method],
                        pretty_masks[mask_method]
                    )
                )

                axis = subset['post_adi'].hist(
                    bins=100,
                    range=(0.,0.11),
                    histtype='step',
                    cumulative=True,
                    density=True)

    legends.append('Input Noise')
    axis = subset['pre_adi'].hist(
        bins=100,
        range=(0.,0.11),
        histtype='step',
        cumulative=True,
        density=True)
    axis.set_xlabel('Threshold in m')
    axis.set_ylabel('Percentage below threshold')
    axis.legend(legends, loc='lower right')
    plt.show()
