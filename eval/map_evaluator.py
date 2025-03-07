# eval utils
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# os / filsystem
import os
from os import listdir

# cv2
import cv2

# numpy
import numpy as np
import torch

from vision_models.clip_dense import ClipModel

SEQ_LEN = 3
results_path = "results_multi"
path_evaluation = "map_evaluation"
object_nav_path = "datasets/multiobject_episodes/"
device = "cuda"


def evaluate_maps():

    generated_map_dir = os.path.join(results_path, "saved_maps")
    gt_map_dir = os.path.join(results_path, "saved_maps_gt")

    # Check if the state directory exists
    if not os.path.isdir(generated_map_dir):
        print(f"Error: {generated_map_dir} is not a valid directory")

    # load episode details
    scene_data = {}
    episodes = []
    episodes, scene_data = HM3DMultiDataset.load_hm3d_multi_episodes(
        episodes, scene_data, object_nav_path
    )

    # Iterate through all files in the directory
    for filename in sorted(listdir(generated_map_dir)):

        # load generated maps
        map_objects = np.load(
            os.path.join(generated_map_dir, filename), allow_pickle=True
        )
        navigable_map = map_objects["nav_map"]
        feature_map = map_objects["feature_map"]
        confidence_map = map_objects["confidence_map"]
        query = str(map_objects["query"])
        pose_observations = map_objects["pose_observations"]
        gt_goal_objects = map_objects["gt_goal_objects"]

        # load ground-truth maps
        gt_map_objects = np.load(os.path.join(gt_map_dir, filename), allow_pickle=True)
        gt_topdown_map = gt_map_objects["gt_topdown_map"]
        gt_object_goals = gt_map_objects["gt_object_goals"]
        experiment_result = gt_map_objects["experiment_result"][0]
        is_success = 'Y' if int(experiment_result["state"]) == 1 else 'N'
        goal_sequence = experiment_result["sequence"]
        experiment_num = int(experiment_result["experiment"])

        # find similarity map
        feature_map_tensor = (
            torch.from_numpy(feature_map)
            .type(torch.float)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        clip = ClipModel("weights/clip.pth")
        query_text_features = clip.get_text_features(["a " + query]).to(device)
        similarity = clip.compute_similarity(feature_map_tensor, query_text_features)
        # similarity[similarity<0] = 0
        final_sim = ((similarity + 1.0) / 2.0).cpu().numpy()
        final_sim = final_sim[0]

        # save as image
        final_sim_img = monochannel_to_inferno_rgb(final_sim)
        final_sim_img = final_sim_img.transpose((1, 0, 2))
        final_sim_img = np.flip(final_sim_img, axis=0)

        # Create the plot
        fig = plt.figure(figsize=(50, 10))
        outer_grid = fig.add_gridspec(1, 2, width_ratios=[1, 5])

        # Add the three "normal" subplots
        ax1 = fig.add_subplot(outer_grid[0, 0])

        poses_ = np.array([po["pose_map"] for po in pose_observations])
        ax1.imshow(
            final_sim_img[:, :, ::-1],
            interpolation="nearest",
            aspect="equal",
            extent=(0, final_sim_img.shape[1], 0, final_sim_img.shape[0]),
        )

        ax1.plot(poses_[:, 0], poses_[:, 1], "b-")
        ax1.plot(poses_[0, 0], poses_[0, 1], "ro")
        ax1.plot(poses_[-1, 0], poses_[-1, 1], "gs")
        
        obj_locs = np.array([go["center"] for go in gt_goal_objects])
        for obj in obj_locs:
            ax1.plot(obj[:, 0], obj[:, 1], "g*")

        # Set equal aspect ratio to ensure accurate positions
        ax1.axis("equal")

        # Add labels and title
        ax1.set_xlabel("X position")
        ax1.set_ylabel("Y position")
        ax1.set_title("Path of Poses")

        # Add grid for better readability
        ax1.grid(True)

        inner = gridspec.GridSpecFromSubplotSpec(
            4, 7, subplot_spec=outer_grid[1]
        )
        for j in range(28):
            ax = plt.Subplot(fig, inner[j])
            t = ax.text(0.1, 0.0, '(T-%d)' % (j))
            t.set_ha('center')
            ax.imshow(pose_observations[-1-j]["obs_from_pose"]["rgb"])
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

        fig.suptitle(f"[Episode:{experiment_num}; Goal:({goal_sequence}){gt_object_goals[goal_sequence]}; Success:{is_success}]")

        # Save the plot as SVG
        plt.savefig(
            f"{results_path}/{path_evaluation}/{filename[:-4]}_sim_map.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
        )

        # cv2.imwrite(
        #     f"{results_path}/{path_evaluation}/{filename[:-4]}_sim_map.png",
        #     final_sim_img,
        # )

        # save topdown map
        # cv2.putText(
        #     gt_topdown_map,
        #     f'[Goals: {gt_object_goals[goal_sequence]}]',
        #     (10, 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.3,
        #     (0, 0, 0),
        #     1,
        #     lineType=cv2.LINE_AA,
        # )
        # cv2.imwrite(
        #     f"{results_path}/{path_evaluation}/{filename[:-4]}_gt_map.png",
        #     gt_topdown_map,
        # )


if __name__ == "__main__":
    evaluate_maps()
