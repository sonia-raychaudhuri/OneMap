# numpy
import os
import imageio
import numpy as np
import torch
import tqdm

# cv2
import cv2

# rerun
import rerun as rr

from typing import (
    Dict,
    List,
    Optional,
    Union,
)
from habitat.utils.visualizations.utils import (
    observations_to_image,
    append_text_underneath_image,
    tile_images
)
import supervision as sv
from PIL import Image

SENSORS_TO_INCLUDE = ["rgb"]

def log_map_rerun(map_, path, needs_orientation=False):
    """
    Applies the inferno colormap to the map and logs it to rerun at the given path
    :param map_: 2D array
    :param path: logging path
    :param needs_orientation:
    :return:
    """
    if needs_orientation:
        map_ = map_.transpose((1, 0))
        map_ = np.flip(map_, axis=0)
    map_ = monochannel_to_inferno_rgb(map_)
    rr.log(path, rr.Image(np.flip(map_, axis=-1)).compress(jpeg_quality=50))


def publish_sim_map(sim_map, br, publisher):
    sim_map = sim_map.transpose((1, 0))
    sim_map = np.flip(sim_map, axis=0)
    sim_map = monochannel_to_inferno_rgb(sim_map)
    # upscale to 1000x1000
    sim_map = cv2.resize(sim_map, (1000, 1000))
    img_msg = br.cv2_to_imgmsg(sim_map, encoding="bgr8")
    publisher.publish(img_msg)

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_")

    # File names are not allowed to be over 255 characters
    video_name_split = video_name.split("/")
    video_name = "/".join(
        video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
    )

    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
    )
    images_iter: List[np.ndarray] = images
    for im in images_iter:
        try:
            writer.append_data(im)
        except:
            im1=cv2.resize(im, (images[0].shape[1], images[0].shape[0]))
            writer.append_data(im1)
    writer.close()

def generate_video(
    video_dir: Optional[str],
    video_name: str,
    images: List[np.ndarray],
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r""" similar to habitat_baselines.utils.common.generate_video

    Args:
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    images_to_video(
        images, video_dir, video_name, fps=fps, verbose=verbose
    )

def add_sim_maps_to_image(observation: Dict, maps: Dict=None, info: Dict=None, text_to_append: str = "") -> np.ndarray:

    render_obs_images: List[np.ndarray] = []
    if (
        len(maps) > 0
        and "object_detected" in maps
        and maps["object_detected"]
        and "predictions" in maps
        and len(maps["predictions"]) > 0
        and len(maps["predictions"]["boxes"]) > 0
    ):
        predictions = maps["predictions"]
        detections = sv.Detections(
            xyxy=np.array(predictions['boxes']),
            class_id=np.array([0 for _ in range(len(predictions['boxes']))]),
            confidence=np.array(predictions['scores'])
        )

        obs_k = observation["rgb"][:,:,:3]
        if not isinstance(obs_k, np.ndarray):
            obs_k = obs_k.cpu().numpy()
        svimage = Image.fromarray(obs_k)

        bounding_box_annotator = sv.BoxAnnotator()
        svimage = bounding_box_annotator.annotate(svimage, detections)
        render_obs_images.append(np.array(svimage))
    else:
        for sensor_name in observation:
            if sensor_name not in SENSORS_TO_INCLUDE:
                continue

            if isinstance(observation[sensor_name], np.ndarray) and len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if len(obs_k.shape) == 3 and obs_k.shape[2] == 4:
                    obs_k = obs_k[:, :, :3]

                if sensor_name == "depth":
                    mask = obs_k == float('inf')
                    obs_k[mask] = obs_k[~mask].max()
                    kernel_size = 11
                    pad = kernel_size // 2

                    depth_image_smoothed = -torch.nn.functional.max_pool2d(torch.from_numpy(-obs_k).unsqueeze(0), kernel_size,
                                                                        padding=pad,
                                                                        stride=1).squeeze(0)
                    obs_k = depth_image_smoothed.cpu().numpy()
                    if len(obs_k.shape) == 2:
                        obs_k = obs_k[..., np.newaxis]
                    if obs_k.shape[2] == 1:
                        obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    ## add sim maps to the frame
    if len(maps) > 0:
        if "sim_map" in maps:
            _image = maps["sim_map"]
            old_h, old_w, _ = _image.shape
            img_height = render_frame.shape[0]
            img_width = int(float(img_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            _image = cv2.resize(
                _image,
                (img_width, img_height),
                interpolation=cv2.INTER_CUBIC,
            )
            render_frame = np.concatenate((render_frame, _image), axis=1)

    ## append text
    if len(text_to_append) > 0:
        render_frame = append_text_underneath_image(render_frame, text_to_append)

    return render_frame
