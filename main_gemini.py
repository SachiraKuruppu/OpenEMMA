import base64
import os.path
import re
import argparse
from datetime import datetime
from dotenv import load_dotenv
from math import atan2
from typing import List, Tuple, Optional, Dict, Any, Union

import cv2
import numpy as np
from google import genai
from google.genai import types
from nuscenes import NuScenes
from scipy.integrate import cumulative_trapezoid

import json
from openemma.YOLO3D.inference import yolo3d_nuScenes
from utils import EstimateCurvatureFromTrajectory, IntegrateCurvatureForPoints, OverlayTrajectory, WriteImageSequenceToVideo
import time
import textwrap

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client with API key from environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

OBS_LEN = 10
FUT_LEN = 10
TTL_LEN = OBS_LEN + FUT_LEN

def vlm_inference(text: Optional[str] = None, images: Optional[Union[str, List[str]]] = None, 
                 sys_message: Optional[str] = None, processor: Optional[Any] = None, 
                 model: Optional[Any] = None, tokenizer: Optional[Any] = None, 
                 args: Optional[argparse.Namespace] = None) -> str:
    """Gemini-specific VLM inference function"""
    # Convert base64 images to proper format for Gemini
    image_parts = []
    if isinstance(images, list):
        for img_b64 in images:
            img_bytes = base64.b64decode(img_b64)
            image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
    else:
        # Single image
        img_bytes = base64.b64decode(images)
        image_parts.append(types.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))
    
    # Create contents list with images and text (no system message here)
    contents = []
    contents.extend(image_parts)
    contents.append(text)
    
    # Create config with system instruction if provided
    config = None
    if sys_message is not None:
        config = types.GenerateContentConfig(
            system_instruction=sys_message
        )
    
    response = None
    retries = 0
    while response is None and retries < 3:
        try:
            response = client.models.generate_content(
                model="models/gemini-2.0-flash", #"models/gemini-2.5-flash",
                contents=contents,
                config=config
            )
        except Exception as e:
            print("Error during Gemini inference:", e)
            retries += 1
            time.sleep(2)  # Wait before retrying
            continue
    
    if response is None:
        raise RuntimeError("Failed to get response from Gemini after 3 retries")
    
    return response.text

def SceneDescription(obs_images: List[str], processor: Optional[Any] = None, 
                    model: Optional[Any] = None, tokenizer: Optional[Any] = None, 
                    args: Optional[argparse.Namespace] = None) -> str:
    prompt = f"""You are a autonomous driving labeller. You have access to these front-view camera images of a car taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Describe the driving scene according to traffic lights, movements of other cars or pedestrians and lane markings."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeObjects(obs_images: List[str], processor: Optional[Any] = None, 
                   model: Optional[Any] = None, tokenizer: Optional[Any] = None, 
                   args: Optional[argparse.Namespace] = None) -> str:
    prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. What other road users should you pay attention to in the driving scene? List two or three of them, specifying its location within the image of the driving scene and provide a short description of the that road user on what it is doing, and why it is important to you."""

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def DescribeOrUpdateIntent(obs_images: List[str], prev_intent: Optional[str] = None, 
                          processor: Optional[Any] = None, model: Optional[Any] = None, 
                          tokenizer: Optional[Any] = None, 
                          args: Optional[argparse.Namespace] = None) -> str:
    if prev_intent is None:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Based on the lane markings and the movement of other cars and pedestrians, describe the desired intent of the ego car. Is it going to follow the lane to turn left, turn right, or go straight? Should it maintain the current speed or slow down or speed up?"""
    else:
        prompt = f"""You are a autonomous driving labeller. You have access to a front-view camera images of a vehicle taken at a 0.5 second interval over the past 5 seconds. Imagine you are driving the car. Half a second ago your intent was to {prev_intent}. Based on the updated lane markings and the updated movement of other cars and pedestrians, do you keep your intent or do you change it? Explain your current intent: """

    result = vlm_inference(text=prompt, images=obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    return result

def GenerateMotion(obs_images: List[str], obs_waypoints: List[List[float]], 
                  obs_velocities: np.ndarray, obs_curvatures: np.ndarray, 
                  given_intent: Optional[str], processor: Optional[Any] = None, 
                  model: Optional[Any] = None, tokenizer: Optional[Any] = None, 
                  args: Optional[argparse.Namespace] = None) -> Tuple[str, str, str, str]:
    # Always use OpenEMMA method for Gemini implementation
    scene_description = SceneDescription(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    object_description = DescribeObjects(obs_images, processor=processor, model=model, tokenizer=tokenizer, args=args)
    intent_description = DescribeOrUpdateIntent(obs_images, prev_intent=given_intent, processor=processor, model=model, tokenizer=tokenizer, args=args)
    print(f'Scene Description: {scene_description}')
    print(f'Object Description: {object_description}')
    print(f'Intent Description: {intent_description}')

    # Convert array waypoints to string.
    obs_waypoints_str = [f"[{x[0]:.2f},{x[1]:.2f}]" for x in obs_waypoints]
    obs_waypoints_str = ", ".join(obs_waypoints_str)
    obs_velocities_norm = np.linalg.norm(obs_velocities, axis=1)
    obs_curvatures = obs_curvatures * 100
    obs_speed_curvature_str = [f"[{x[0]:.1f},{x[1]:.1f}]" for x in zip(obs_velocities_norm, obs_curvatures)]
    obs_speed_curvature_str = ", ".join(obs_speed_curvature_str)

    print(f'Observed Speed and Curvature: {obs_speed_curvature_str}')

    sys_message = ("You are a autonomous driving labeller. You have access to a front-view camera image of a vehicle, a sequence of past speeds, a sequence of past curvatures, and a driving rationale. Each speed, curvature is represented as [v, k], where v corresponds to the speed, and k corresponds to the curvature. A positive k means the vehicle is turning left. A negative k means the vehicle is turning right. The larger the absolute value of k, the sharper the turn. A close to zero k means the vehicle is driving straight. As a driver on the road, you should follow any common sense traffic rules. You should try to stay in the middle of your lane. You should maintain necessary distance from the leading vehicle. You should observe lane markings and follow them.  Your task is to do your best to predict future speeds and curvatures for the vehicle over the next 10 timesteps given vehicle intent inferred from the image. Make a best guess if the problem is too difficult for you. If you cannot provide a response people will get injured.\n")

    # Always use OpenEMMA prompt with scene understanding
    prompt = f"""These are frames from a video taken by a camera mounted in the front of a car. The images are taken at a 0.5 second interval. 
    The scene is described as follows: {scene_description}. 
    The identified critical objects are {object_description}. 
    The car's intent is {intent_description}. 
    The 5 second historical velocities and curvatures of the ego car are {obs_speed_curvature_str}. 
    Infer the association between these numbers and the image sequence. Generate the predicted future speeds and curvatures in the format [speed_1, curvature_1], [speed_2, curvature_2],..., [speed_10, curvature_10]. Write the raw text not markdown or latex. Future speeds and curvatures:"""
    
    for rho in range(3):
        result = vlm_inference(text=prompt, images=obs_images, sys_message=sys_message, processor=processor, model=model, tokenizer=tokenizer, args=args)
        if not "unable" in result and not "sorry" in result and "[" in result:
            break
    return result, scene_description, object_description, intent_description

def extract_scene_data(nusc: NuScenes, first_sample_token: str, 
                      last_sample_token: str) -> Tuple[List[str], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract camera images, ego poses, and camera parameters for a scene"""
    front_camera_images = []
    ego_poses = []
    camera_params = []
    curr_sample_token = first_sample_token
    
    while True:
        sample = nusc.get('sample', curr_sample_token)

        # Get the front camera image of the sample.
        cam_front_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])

        # For Gemini, always load images as base64
        with open(os.path.join(nusc.dataroot, cam_front_data['filename']), "rb") as image_file:
            front_camera_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

        # Get the ego pose of the sample.
        pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
        ego_poses.append(pose)

        # Get the camera parameters of the sample.
        camera_params.append(nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token']))

        # Advance the pointer.
        if curr_sample_token == last_sample_token:
            break
        curr_sample_token = sample['next']
    
    return front_camera_images, ego_poses, camera_params

def compute_trajectory_data(ego_poses: List[Dict[str, Any]], 
                           scene_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[float]]]:
    """Compute trajectory-related data including velocities and curvatures"""
    # Get the velocities of the ego vehicle.
    ego_poses_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
    ego_poses_world = np.array(ego_poses_world)

    ego_velocities = np.zeros_like(ego_poses_world)
    ego_velocities[1:] = ego_poses_world[1:] - ego_poses_world[:-1]
    ego_velocities[0] = ego_velocities[1]

    # Get the curvature of the ego vehicle.
    ego_curvatures = EstimateCurvatureFromTrajectory(ego_poses_world)

    # Get the waypoints of the ego vehicle.
    ego_traj_world = [ego_poses[t]['translation'][:3] for t in range(scene_length)]
    
    return ego_poses_world, ego_velocities, ego_curvatures, ego_traj_world

def process_frame_prediction(obs_images: List[str], obs_ego_traj_world: List[List[float]], 
                           obs_ego_velocities: np.ndarray, obs_ego_curvatures: np.ndarray, 
                           prev_intent: Optional[str], processor: Optional[Any], 
                           model: Optional[Any], tokenizer: Optional[Any], 
                           args: argparse.Namespace) -> Tuple[Optional[List[List[float]]], Optional[str], Optional[str], Optional[str]]:
    """Process a single frame to generate motion prediction"""
    for rho in range(3):
        # Use obs_images (list of base64 images) for Gemini inference
        (prediction,
        scene_description,
        object_description,
        updated_intent) = GenerateMotion(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                        obs_ego_curvatures, prev_intent, processor=processor, 
                                        model=model, tokenizer=tokenizer, args=args)

        # Process the output.
        pred_waypoints = prediction.replace("Future speeds and curvatures:", "").strip()
        coordinates = re.findall(r"\[([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\]", pred_waypoints)
        if coordinates:
            break
    
    if not coordinates:
        return None, None, None, None
    
    speed_curvature_pred = [[float(v), float(k)] for v, k in coordinates]
    speed_curvature_pred = speed_curvature_pred[:10]
    print(f"Got {len(speed_curvature_pred)} future actions: {speed_curvature_pred}")
    
    return speed_curvature_pred, scene_description, object_description, updated_intent

def compute_prediction_trajectory(speed_curvature_pred: List[List[float]], 
                                fut_start_world: List[float], 
                                obs_ego_velocities: np.ndarray) -> Tuple[np.ndarray, int]:
    """Compute predicted trajectory from speed and curvature predictions"""
    pred_len = min(FUT_LEN, len(speed_curvature_pred))
    pred_curvatures = np.array(speed_curvature_pred)[:, 1] / 100
    pred_speeds = np.array(speed_curvature_pred)[:, 0]
    pred_traj = np.zeros((pred_len, 3))
    pred_traj[:pred_len, :2] = IntegrateCurvatureForPoints(pred_curvatures,
                                                           pred_speeds,
                                                           fut_start_world,
                                                           atan2(obs_ego_velocities[-1][1],
                                                                 obs_ego_velocities[-1][0]), pred_len)
    return pred_traj, pred_len

def compute_ade_metrics(fut_ego_traj_world: List[List[float]], pred_traj: np.ndarray, 
                       pred_len: int) -> Tuple[float, float, float, float]:
    """Compute Average Displacement Error (ADE) metrics"""
    fut_ego_traj_world = np.array(fut_ego_traj_world)
    ade = np.mean(np.linalg.norm(fut_ego_traj_world[:pred_len] - pred_traj, axis=1))
    
    pred1_len = min(pred_len, 2)
    ade1s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred1_len] - pred_traj[:pred1_len], axis=1))

    pred2_len = min(pred_len, 4)
    ade2s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred2_len] - pred_traj[:pred2_len], axis=1))

    pred3_len = min(pred_len, 6)
    ade3s = np.mean(np.linalg.norm(fut_ego_traj_world[:pred3_len] - pred_traj[:pred3_len], axis=1))
    
    return ade, ade1s, ade2s, ade3s

def create_visualization_image(img: np.ndarray, i: int, ade: float, 
                             updated_intent: str, 
                             speed_curvature_pred: List[List[float]]) -> np.ndarray:
    """Create visualization image with text overlays"""
    img_with_text = img.copy()
    
    # Add text overlay with predictions and intent
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)  # White text
    thickness = 1
    line_type = cv2.LINE_AA
    
    # Create semi-transparent overlay for better text readability
    overlay = img_with_text.copy()
    cv2.rectangle(overlay, (10, 10), (img_with_text.shape[1]-10, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_with_text, 0.3, 0, img_with_text)
    
    # Prepare text content
    y_offset = 30
    line_height = 20
    
    # Add frame info
    frame_text = f"Frame: {i}, ADE: {ade:.2f}"
    cv2.putText(img_with_text, frame_text, (15, y_offset), font, font_scale, color, thickness, line_type)
    y_offset += line_height
    
    # Add intent (wrap text if too long)
    intent_lines = textwrap.wrap(f"Intent: {updated_intent}", width=80)
    for line in intent_lines[:2]:  # Show max 2 lines
        cv2.putText(img_with_text, line, (15, y_offset), font, font_scale, color, thickness, line_type)
        y_offset += line_height
    
    # Add prediction summary
    pred_summary = f"Prediction: {len(speed_curvature_pred)} actions"
    cv2.putText(img_with_text, pred_summary, (15, y_offset), font, font_scale, color, thickness, line_type)
    y_offset += line_height
    
    # Add first few predicted actions
    if len(speed_curvature_pred) > 0:
        actions_text = f"Next actions: [{speed_curvature_pred[0][0]:.1f},{speed_curvature_pred[0][1]:.2f}]"
        if len(speed_curvature_pred) > 1:
            actions_text += f" [{speed_curvature_pred[1][0]:.1f},{speed_curvature_pred[1][1]:.2f}]"
        cv2.putText(img_with_text, actions_text, (15, y_offset), font, font_scale, color, thickness, line_type)
    
    return img_with_text

def process_scene_frames(front_camera_images: List[str], ego_poses: List[Dict[str, Any]], 
                        camera_params: List[Dict[str, Any]], ego_traj_world: List[List[float]], 
                        ego_velocities: np.ndarray, ego_curvatures: np.ndarray, 
                        scene_length: int, processor: Optional[Any], model: Optional[Any], 
                        tokenizer: Optional[Any], 
                        args: argparse.Namespace) -> Tuple[List[np.ndarray], List[float], List[float], List[float]]:
    """Process all frames in a scene to generate predictions and visualizations"""
    prev_intent = None
    cam_images_sequence = []
    ade1s_list = []
    ade2s_list = []
    ade3s_list = []
    
    for i in range(scene_length - TTL_LEN):
        # Get the raw image data.
        obs_images = front_camera_images[i:i+OBS_LEN]
        obs_ego_poses = ego_poses[i:i+OBS_LEN]
        obs_camera_params = camera_params[i:i+OBS_LEN]
        obs_ego_traj_world = ego_traj_world[i:i+OBS_LEN]
        fut_ego_traj_world = ego_traj_world[i+OBS_LEN:i+TTL_LEN]
        obs_ego_velocities = ego_velocities[i:i+OBS_LEN]
        obs_ego_curvatures = ego_curvatures[i:i+OBS_LEN]

        # Get positions of the vehicle.
        fut_start_world = obs_ego_traj_world[-1]
        curr_image = obs_images[-1]

        # Process images for Gemini (decode from base64 for visualization)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(curr_image), dtype=np.uint8), cv2.IMREAD_COLOR)
        img = yolo3d_nuScenes(img, calib=obs_camera_params[-1])[0]

        # Generate motion prediction
        result = process_frame_prediction(obs_images, obs_ego_traj_world, obs_ego_velocities,
                                        obs_ego_curvatures, prev_intent, processor, model, tokenizer, args)
        
        speed_curvature_pred, scene_description, object_description, updated_intent = result
        
        if speed_curvature_pred is None:
            continue
            
        prev_intent = updated_intent  # Stateful intent

        # Compute predicted trajectory
        pred_traj, pred_len = compute_prediction_trajectory(speed_curvature_pred, fut_start_world, obs_ego_velocities)

        # Overlay the trajectory.
        check_flag = OverlayTrajectory(img, pred_traj.tolist(), obs_camera_params[-1], obs_ego_poses[-1], color=(255, 0, 0), args=args)
        
        # Compute ADE metrics
        ade, ade1s, ade2s, ade3s = compute_ade_metrics(fut_ego_traj_world, pred_traj, pred_len)
        
        ade1s_list.append(ade1s)
        ade2s_list.append(ade2s)
        ade3s_list.append(ade3s)

        # Create visualization if plotting enabled
        if args.plot:
            img_with_text = create_visualization_image(img, i, ade, updated_intent, speed_curvature_pred)
            cam_images_sequence.append(img_with_text)
    
    return cam_images_sequence, ade1s_list, ade2s_list, ade3s_list

def save_scene_results(name: str, token: str, ade1s_list: List[float], 
                      ade2s_list: List[float], ade3s_list: List[float], 
                      cam_images_sequence: List[np.ndarray], timestamp: str, 
                      args: argparse.Namespace) -> None:
    """Compute and save scene results"""
    mean_ade1s = np.mean(ade1s_list)
    mean_ade2s = np.mean(ade2s_list)
    mean_ade3s = np.mean(ade3s_list)
    aveg_ade = np.mean([mean_ade1s, mean_ade2s, mean_ade3s])

    result = {
        "name": name,
        "token": token,
        "ade1s": mean_ade1s,
        "ade2s": mean_ade2s,
        "ade3s": mean_ade3s,
        "avgade": aveg_ade
    }

    with open(f"{timestamp}/ade_results.jsonl", "a") as f:
        f.write(json.dumps(result))
        f.write("\n")

    if args.plot:
        WriteImageSequenceToVideo(cam_images_sequence, f"{timestamp}/{name}")

def setup_and_initialize() -> Tuple[argparse.Namespace, Optional[Any], Optional[Any], Optional[Any], str, NuScenes]:
    """Parse arguments, initialize models, and load dataset"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--dataroot", type=str, default='datasets/NuScenes')
    parser.add_argument("--version", type=str, default='v1.0-mini')
    args = parser.parse_args()

    # For Gemini, we don't need to load any local models
    model = None
    processor = None
    tokenizer = None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    timestamp = "gemini_results/" + timestamp
    os.makedirs(timestamp, exist_ok=True)

    # Load the dataset
    nusc = NuScenes(version=args.version, dataroot=args.dataroot)
    
    return args, model, processor, tokenizer, timestamp, nusc

if __name__ == '__main__':
    args, model, processor, tokenizer, timestamp, nusc = setup_and_initialize()

    # Iterate the scenes
    scenes = nusc.scene
    
    print(f"Number of scenes: {len(scenes)}")

    for scene in scenes:
        token = scene['token']
        first_sample_token = scene['first_sample_token']
        last_sample_token = scene['last_sample_token']
        name = scene['name']
        description = scene['description']

        if not name in ["scene-0103", "scene-1077"]:
            continue

        # Get all image and pose in this scene
        front_camera_images, ego_poses, camera_params = extract_scene_data(
            nusc, first_sample_token, last_sample_token
        )

        scene_length = len(front_camera_images)
        print(f"Scene {name} has {scene_length} frames")

        if scene_length < TTL_LEN:
            print(f"Scene {name} has less than {TTL_LEN} frames, skipping...")
            continue

        ## Compute interpolated trajectory.
        ego_poses_world, ego_velocities, ego_curvatures, ego_traj_world = compute_trajectory_data(
            ego_poses, scene_length
        )

        # Process frames and compute predictions
        cam_images_sequence, ade1s_list, ade2s_list, ade3s_list = process_scene_frames(
            front_camera_images, ego_poses, camera_params, ego_traj_world, 
            ego_velocities, ego_curvatures, scene_length, 
            processor, model, tokenizer, args
        )

        # Compute and save results
        save_scene_results(name, token, ade1s_list, ade2s_list, ade3s_list, 
                          cam_images_sequence, timestamp, args)
