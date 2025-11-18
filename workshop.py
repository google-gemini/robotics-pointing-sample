import argparse
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
from google import genai
from google.genai import types
from lerobot.model.kinematics import RobotKinematics
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
import numpy as np
from PIL import Image
import requests

import config

# Suppress noisy logs from libraries
logging.getLogger("lerobot").setLevel(logging.WARNING)


# --- Helper Functions ---


def show_image(image_bgr, window_name="Vision Feedback"):
  """Displays the image using OpenCV.

  NOTE: This requires a running X server or GUI environment. Press 'q' to close
  the window.
  """
  if image_bgr is None:
    return

  # Create the window and show the image
  cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
  # Resize image for better viewing in a separate window
  # Assuming standard resolution, scale it down slightly
  h, w = image_bgr.shape[:2]
  scale_factor = 500 / h
  disp = cv2.resize(image_bgr, (int(w * scale_factor), 500))
  cv2.imshow(window_name, disp)

  # Wait for a short duration to update the display
  key = cv2.waitKey(1)
  if key == ord("q"):
    cv2.destroyAllWindows()


def move_to_joints(bot, target_joints_deg, gripper_pos=0, duration=1.5):
  """Interpolates directly to specific joint angles (no IK)."""
  if bot is None:
    print(
        "⚠️ [SIM] Robot not connected. Simulating joint move to"
        f" {np.round(target_joints_deg, 2)} deg."
    )
    time.sleep(duration) # Simulate time taken
    return

  # Get current angles
  q_current = np.array([bot.get_observation()[n] for n in config.JOINT_NAMES])
  target_joints_deg_full = np.copy(target_joints_deg)

  # Simple interpolation loop
  steps = int(duration * 50)
  for i in range(1, steps + 1):
    t = i / steps
    q_interp = q_current + t * (target_joints_deg_full - q_current)
    bot.send_action({name: val for name, val in zip(config.JOINT_NAMES, q_interp)})
    time.sleep(duration / steps)


# --- Kinematics Engine Class ---


class KinematicsEngine:
  """A unified interface for robot kinematics, abstracting away the underlying

  math library. Handles asset downloading automatically.
  """

  def __init__(self, backend="lerobot", model_dir="third_party/SO101"):
    self.backend = backend.lower()
    self.model_dir = Path(model_dir)
    self.model_dir.mkdir(exist_ok=True)
    self.urdf_path = self.model_dir / "so101_new_calib.urdf"

    self.solver = None
    self.ee_link = "gripper_frame_link"  # End-effector link name

    print(
        f"\n⚙️ Initializing Kinematics with backend: {self.backend.upper()}..."
    )

    # 1. Download common assets if missing
    self._ensure_assets()

    # 2. Setup specific backend
    if self.backend == "lerobot":
      self._setup_lerobot()
    elif self.backend == "argo":
      self._setup_argo()
    elif self.backend == "mujoco":
      self._setup_mujoco()
    else:
      raise ValueError(f"Unknown backend: {backend}")

  def _download_project_files(
      self, base_url: str, file_paths: list[str], output_dir: Path
  ):
    """Helper method: Downloads files maintaining relative directory structure."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ⏳ Checking {len(file_paths)} assets...")

    for rel_path in file_paths:
      local_path = output_dir / rel_path
      remote_url = f"{base_url.rstrip('/')}/{rel_path.lstrip('/')}"

      if local_path.exists():
        continue

      local_path.parent.mkdir(parents=True, exist_ok=True)

      try:
        response = requests.get(remote_url)
        response.raise_for_status()

        with open(local_path, "wb") as f:
          f.write(response.content)
        print(f"   ⬇️ Downloaded: {rel_path}")

      except Exception as e:
        print(f"   ❌ Failed to download {rel_path}: {e}")

    print("   ✅ Asset check complete.")

  def _ensure_assets(self):
    """Downloads URDF and Meshes directly from GitHub, fixing paths."""
    repo_base = "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Simulation/SO101"

    files_to_download = [
        "so101_new_calib.urdf",
        "assets/waveshare_mounting_plate_so101_v2.stl",
        "assets/sts3215_03a_v1.stl",
        "assets/motor_holder_so101_base_v1.stl",
        "assets/wrist_roll_follower_so101_v1.stl",
        "assets/moving_jaw_so101_v1.stl",
        "assets/base_motor_holder_so101_v1.stl",
        "assets/upper_arm_so101_v1.stl",
        "assets/wrist_roll_pitch_so101_v2.stl",
        "assets/under_arm_so101_v1.stl",
        "assets/rotation_pitch_so101_v1.stl",
        "assets/motor_holder_so101_wrist_v1.stl",
        "assets/sts3215_03a_no_horn_v1.stl",
        "assets/base_so101_v2.stl",
    ]

    # 1. Download files using the generic helper
    self._download_project_files(repo_base, files_to_download, self.model_dir)

  def _setup_lerobot(self):
    """Sets up the official LeRobot kinematics solver."""
    self.solver = RobotKinematics(urdf_path=str(self.urdf_path))
    print("   ✅ LeRobot Kinematics ready.")

  def _setup_mujoco(self):
    """Sets up MuJoCo physics engine (Stub for future IK)."""
    import mujoco

    try:
      print("   ✅ MuJoCo Model loaded (IK not yet implemented).")
    except Exception as e:
      print(f"   ❌ MuJoCo Load Error: {e}")

  def _setup_argo(self):
    """Sets up the custom 'Argo' control library/solver."""
    self.argo_dir = Path("third_party/Argo-Robot/controls")
    
    try:
      if str(self.argo_dir.resolve()) not in sys.path:
        sys.path.append(str(self.argo_dir.resolve()))
      from scripts.model import URDF_loader, RobotModel
      from scripts.kinematics import URDF_Kinematics
    except ImportError as e:
      print(
        "   ❌ Failed to import Argo controls, files or prereqs"
        f" may be missing. Error: {e}"
      )
    try:
      loader = URDF_loader()
      loader.load(str(self.urdf_path))
      self.argo_model = RobotModel(loader)
      self.solver = URDF_Kinematics()
      print("   ✅ Argo Kinematics ready.")
    except Exception as e:
      print(
          f"   ❌ Failed to ready Argo Kinematics. Error: {e}"
      )
      raise

  def compute_ik(self, current_joints_deg, target_pose_4x4):
    """Computes IK returning joint degrees."""
    try:
      if self.backend == "lerobot":
        q_sol = self.solver.inverse_kinematics(
            current_joints_deg, target_pose_4x4
        )

      elif self.backend == "argo":
        # Argo expects radians and reversed joint order
        q_start = np.deg2rad(current_joints_deg)[::-1]
        q_sol = self.solver.inverse_kinematics(
            self.argo_model,
            q_start,
            target_pose_4x4,
            self.ee_link,
            use_orientation=False,
            k=0.8,
            n_iter=100,
        )
        if q_sol is None:
          return None
        q_sol = np.rad2deg(q_sol[::-1])

      elif self.backend == "mujoco":
        print("⚠️ MuJoCo IK not implemented yet.")
        return None

      # Check if gripper pos is missing and restore if needed (LeRobot/Argo only compute 5 arm joints)
      if q_sol is not None and len(q_sol) == 5:
        # Append current gripper position to the 5 arm joint solutions
        q_sol = np.append(q_sol, current_joints_deg[-1])

      return q_sol

    except Exception as e:
      print(f"IK Computation Error: {e}")
      return None
    # Should not be reachable, but ensure we don't return None if q_sol is set but not returned
    return None  # Return None if no solution path was followed


# --- Main Logic Functions ---


def perform_move(bot, engine, target_xyz, gripper_pos=0, duration=1.5):
  """Calculates IK and moves the robot smoothly to the target XYZ."""
  if bot is None:
    print(
        f"⚠️ [SIM] Robot not connected. Target: {np.round(target_xyz, 3)}m."
        " Skipping move."
    )
    return True

  # 1. Get current state (all 6 joints, including gripper)
  q_current = np.array([bot.get_observation()[n] for n in config.JOINT_NAMES])

  # 2. Construct Target Pose (4x4 matrix)
  target_pose = np.eye(4)
  target_pose[:3, 3] = target_xyz

  # 3. Compute IK
  q_target_arm_full = engine.compute_ik(q_current, target_pose)

  if q_target_arm_full is None:
    print(f"❌ Unreachable Target: {np.round(target_xyz, 3)}")
    return False

  # 4. Execute (Reuse joint mover logic)
  move_to_joints(bot, q_target_arm_full, gripper_pos, duration)
  return True


def get_object_center_gemini(client, image_bgr, target_name):
  """Uses Gemini to find 'target_name' in the image.

  Returns: [x_pixel, y_pixel] or None if failed.
  """
  # Convert OpenCV BGR to PIL RGB
  img_pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
  h_px, w_px = image_bgr.shape[:2]

  prompt = (
      f"Locate the center of the {target_name}. "
      "Return ONLY JSON in this format: {'point': [y, x]} "
      "where y and x are normalized coordinates from 0 to 1000."
  )

  retries = 3
  for attempt in range(retries):
    try:
      # Using the specialized Robotics ER model
      response = client.models.generate_content(
          model="gemini-robotics-er-1.5-preview",
          contents=[img_pil, prompt],
          config=types.GenerateContentConfig(
              response_mime_type="application/json", temperature=0.5
          ),
      )

      # Parse JSON
      coords = json.loads(response.text.strip())["point"]
      y_norm, x_norm = coords

      # Convert normalized (0-1000) to pixels
      x_px = int(x_norm / 1000.0 * w_px)
      y_px = int(y_norm / 1000.0 * h_px)

      return np.array([x_px, y_px])

    except Exception as e:
      print(f"⚠️ Gemini Vision Error (Attempt {attempt + 1}/{retries}): {e}")
      time.sleep(1)
  
  print("❌ Gemini failed after multiple attempts.")
  return None


def calibrate_system(cap, board_origin_robot_m):
  """Detects the ChArUco board and computes the Homography matrix (H)
  to map pixels to robot (X, Y) meters.
  """
  aruco_dict = cv2.aruco.getPredefinedDictionary(config.DICT_TYPE)
  board = cv2.aruco.CharucoBoard(
      (config.SQUARES_X, config.SQUARES_Y), config.SQUARE_LENGTH, config.MARKER_LENGTH, aruco_dict
  )
  detector = cv2.aruco.CharucoDetector(board)

  print("📸 Looking for ChArUco board. Ensure the arm is not obscuring it.")
  for _ in range(5):
    cap.read()  # Clear buffer
  ret, frame = cap.read()
  if not ret:
    print("❌ Camera failed to capture frame.")
    return None, None

  # Detect the board and corners
  corners, ids, _, _ = detector.detectBoard(frame)

  if ids is not None and len(ids) > 4:
    print(f"✅ Detected {len(ids)} corners. Calculating Homography...")

    obj_points = []  # Robot coordinates (Meters)
    img_points = []  # Camera coordinates (Pixels)

    all_board_corners = board.getChessboardCorners()

    # 1. Get the local board coordinate of Corner ID 0 (The Anchor)
    origin_local = all_board_corners[0]

    for i, charuco_id in enumerate(ids.flatten()):
      img_points.append(corners[i][0])

      # Get local board XYZ for this specific corner
      current_local = all_board_corners[charuco_id]

      # 2. Calculate Relative Distance from Anchor (ID 0)
      diff_x_board = current_local[0] - origin_local[0]
      diff_y_board = current_local[1] - origin_local[1]

      # 3. Map to Robot Frame (Relative to User Measurement)
      # Robot X (Forward) = User_X - Relative_Board_Y
      rx = board_origin_robot_m[0] - diff_y_board

      # Robot Y (Left) = User_Y - Relative_Board_X
      ry = board_origin_robot_m[1] - diff_x_board

      obj_points.append([rx, ry])

    # Compute Homography
    H, _ = cv2.findHomography(np.array(img_points), np.array(obj_points))

    # Save to file
    np.save(config.CALIBRATION_FILE, {"H": H, "z": 0.0})
    print(f"✅ Calibration Saved to '{config.CALIBRATION_FILE}'")

    # --- Verification Output & Visualization ---
    origin_indices = np.where(ids == 0)[0]
    if len(origin_indices) > 0:
      idx = origin_indices[0]
      px = corners[idx][0].astype(int)
      rob = obj_points[idx]
      print(
          f"   🎯 VERIFICATION (ID 0): Pixel {px} -> Robot {np.round(rob, 4)}m"
      )

    disp = frame.copy()
    cv2.aruco.drawDetectedCornersCharuco(disp, corners, ids)

    if len(origin_indices) > 0:
      idx = origin_indices[0]
      origin_px = corners[idx][0].astype(int)
      cv2.circle(disp, tuple(origin_px), 10, (0, 0, 255), -1)
      cv2.putText(
          disp,
          "Anchor",
          (origin_px[0] + 15, origin_px[1]),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.6,
          (0, 0, 255),
          2,
          cv2.LINE_AA,
      )

    show_image(disp, "Calibration Verification (Press 'q' to close or wait 10s)")
    cv2.waitKey(10000)
    cv2.destroyAllWindows()

    return H, 0.0  # Return H matrix and z_surface
  else:
    print("❌ Not enough corners detected for calibration.")
    return None, None


def main(args):
  # --- 1. Initialization ---
  print("Setting things up, might be slow the first time...")

  # Kinematics Engine Setup
  kin_engine = KinematicsEngine(backend=args.backend)

  # Robot Hardware Connection (SO101Follower)
  robot = None
  if not args.sim:
    print(f"\n⏳ Connecting to robot on {args.port}...")
    try:
      if args.calibration_dir is None:
        config_robot = SO101FollowerConfig(port=args.port, id=args.robot_id)
      else:
        config_robot = SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            calibration_dir=args.calibration_dir,
        )
      robot = SO101Follower(config_robot)
      robot.connect()
      robot.bus.disable_torque()
      print("✅ Robot Hardware Connected & Torque Disabled.")

    except Exception as e:
      print(f"⚠️ Hardware connection failed: {e}")
      print(
          "   ➡️ Proceeding in Simulation Mode (motion commands will be skipped)."
      )
  else:
     print("⚠️ Simulation Mode Enabled: Skipping robot connection.")

  # Camera Setup
  cap = cv2.VideoCapture(args.camera_index)
  for _ in range(5):
    cap.read()  # Warmup buffer

  ret, frame = cap.read()
  if not ret:
    print(f"\n❌ Camera failed on index {args.camera_index}.")
    cap.release()
    return
  print(f"✅ Camera Connected! Resolution: {frame.shape[1]}x{frame.shape[0]}")
  
  # Gemini API Setup
  if not args.api_key:
    print(
        "\n❌ Error: Google API Key is required. Please provide it via"
        " --api-key."
    )
    cap.release()
    return

  os.environ["GOOGLE_API_KEY"] = args.api_key
  try:
    client = genai.Client(api_key=args.api_key)
    print("✅ Gemini API Client Configured.")
  except Exception as e:
    print(f"⚠️ Gemini API Setup Failed: {e}")
    cap.release()
    return

  # --- 2. Calibration ---

  h_matrix, z_surface = None, None

  if not args.sim:
    if Path(config.CALIBRATION_FILE).exists() and not args.recalibrate:
      print(f"\nFound existing calibration file: {config.CALIBRATION_FILE}.")
      try:
        calib_data = np.load(config.CALIBRATION_FILE, allow_pickle=True).item()
        h_matrix = calib_data["H"]
        z_surface = calib_data["z"]
        print(f"✅ Calibration Loaded. Table Z-Plane: {z_surface:.4f}m")
      except Exception as e:
        print(f"❌ Failed to load calibration: {e}. Recalibrating.")
        h_matrix, z_surface = calibrate_system(cap, args.board_origin)
    else:
      h_matrix, z_surface = calibrate_system(cap, args.board_origin)

    if h_matrix is None:
      print("\nFATAL ERROR: System is not calibrated. Exiting.")
      if cap:
        cap.release()
      return
  else:
    print("\n⚠️ Simulation Mode: Skipping Calibration (Visual Only).")

  # --- 3. Main Action Loop ---

  print("\n🤖 SYSTEM READY.")
  print("   - Press 'SPACE' in the video window to capture and command.")
  print("   - Press 'q' to quit.")

  # Move to Home first
  if robot:
    print("🏠 Moving to Home Position...")
    move_to_joints(robot, config.HOME_POSE, gripper_pos=0, duration=config.MOVE_DURATION_HOME)

  cv2.namedWindow("Vision Feedback", cv2.WINDOW_AUTOSIZE)

  while True:
    # 1. Continuous Video Feed
    ret, frame = cap.read()
    if not ret:
      print("❌ Camera Error")
      time.sleep(0.5)
      continue

    # Overlay Instructions
    disp = frame.copy()
    cv2.putText(disp, "Ready. Press SPACE to command, 'q' to quit.", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show Frame
    cv2.imshow("Vision Feedback", disp)
    
    # Check Input
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
      print("👋 Exiting.")
      break
    
    elif key == ord(' '):
      # --- Capture & Command Sequence ---
      
      # 1. Freeze Frame & Prompt
      print("\n📸 Image Captured.")
      cv2.putText(disp, "PAUSED - Enter prompt in terminal...", (20, 70), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
      cv2.imshow("Vision Feedback", disp)
      cv2.waitKey(1) # Update window

      target_name = input("⌨️  What should I point at? (e.g., 'blue block'): ").strip()
      
      if not target_name:
        print("⚠️ Empty prompt, resuming feed.")
        continue

      # 2. Thinking...
      print(f"🤔 Asking Gemini to find '{target_name}'...")
      cv2.putText(disp, f"Thinking: {target_name}...", (20, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
      cv2.imshow("Vision Feedback", disp)
      cv2.waitKey(1)

      pixel_center = get_object_center_gemini(client, frame, target_name)

      if pixel_center is not None:
        # Visualize Result (Always show this)
        cv2.circle(disp, tuple(pixel_center), 10, (0, 255, 0), 2)
        cv2.drawMarker(
            disp, tuple(pixel_center), (0, 255, 0), cv2.MARKER_CROSS, 20, 2
        )
        cv2.putText(disp, f"Target: {target_name}", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Vision Feedback", disp)
        cv2.waitKey(1)

        if h_matrix is not None:
          # 2. Grounding (Pixel -> Robot Meter)
          px_array = np.array([[pixel_center]], dtype="float32")
          px_array = px_array.reshape(-1, 1, 2)
          robot_xy = cv2.perspectiveTransform(px_array, h_matrix)[0][0]

          target_xyz = [robot_xy[0], robot_xy[1], z_surface + config.POINT_HEIGHT]
          hover_xyz = [robot_xy[0], robot_xy[1], z_surface + config.HOVER_HEIGHT]

          print(
              f"📍 Mapped: Pixels {pixel_center} -> Robot"
              f" {np.round(target_xyz, 3)}m"
          )

          # 3. Action Sequence
          print("🏠 Moving to HOME...")
          move_to_joints(robot, config.HOME_POSE, duration=1.5)
          time.sleep(0.2)
          print("🚀 Moving to HOVER...")
          if perform_move(robot, kin_engine, hover_xyz, duration=config.MOVE_DURATION_HOVER):

            time.sleep(0.2)
            print("👇 Descending to POINT...")
            perform_move(robot, kin_engine, target_xyz, duration=config.MOVE_DURATION_POINT)
            
            # Wait a bit to show the result
            time.sleep(1.0)
            
            # Return Home
            print("🏠 Returning to HOME...")
            move_to_joints(robot, config.HOME_POSE, duration=1.5)
        else:
           print(f"📍 Target found at {pixel_center} (Visual Only).")
           time.sleep(1.0) # Pause to let user see the result

      else:
        print("🤷 Gemini could not locate the object.")
        cv2.putText(disp, "Not Found", (20, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Vision Feedback", disp)
        cv2.waitKey(1000) # Show error for 1s

  # --- Cleanup ---
  if robot:
    print("Disabling torque and closing robot connection.")
    robot.bus.disable_torque()
    robot.disconnect()
  if cap:
    print("Releasing camera.")
    cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Vision-Guided Manipulation Script for SO-101 Robot Arm."
  )

  # Hardware/Connection Parameters
  parser.add_argument(
      "--port",
      type=str,
      default=config.DEFAULT_PORT,
      help=(
          "The serial port for the robot arm (e.g., /dev/tty.usbmodem... or"
          " COM3)."
      ),
  )
  parser.add_argument(
      "--robot-id",
      type=str,
      default=config.DEFAULT_ROBOT_ID,
      help=(
          "Identifier for the robot; must match calibration filename without"
          " extension.",
      ),
  )
  parser.add_argument(
      "--calibration-dir",
      type=str,
      default=config.DEFAULT_CALIBRATION_DIR,
      help=(
          "Directory containing the arm calibration files, when not using"
          " default location or lerobot-calibrate command."
      ),
  )
  parser.add_argument(
      "--camera-index",
      type=int,
      default=config.DEFAULT_CAMERA_INDEX,
      help="The index of the USB camera to use (e.g., 0, 1, 2).",
  )
  parser.add_argument(
      "--sim",
      action="store_true",
      help="Run in simulation mode (no robot hardware required).",
  )

  # Kinematics/Brain Parameters
  parser.add_argument(
      "--backend",
      type=str,
      choices=["lerobot", "argo", "mujoco"],
      default=config.DEFAULT_BACKEND,
      help="The kinematics solver backend to use.",
  )
  parser.add_argument(
      "--api-key",
      type=str,
      default=config.GOOGLE_API_KEY,
      help="Your Google AI Studio API Key for Gemini Robotics ER 1.5.",
  )

  # Calibration Parameters
  parser.add_argument(
      "--board-origin",
      type=float,
      nargs=2,
      default=config.DEFAULT_BOARD_ORIGIN,
      metavar=("X_FORWARD", "Y_LEFT"),
      help=(
          "Robot coordinates (meters) for the ChArUco board origin (Corner ID"
          " 0). Format: X_FORWARD Y_LEFT (e.g., 0.29 0.0525)"
      ),
  )
  parser.add_argument(
      "--recalibrate",
      action="store_true",
      help="Force recalibration even if a calibration file exists.",
  )

  args = parser.parse_args()
  
  # --- Validation & Defaults ---
  
  # 1. API Key
  if not args.api_key:
      print("\n❌ Error: Google API Key is missing.")
      print("   Please provide it via --api-key OR set GOOGLE_API_KEY in config.py.")
      sys.exit(1)

  # 2. Hardware (if not sim)
  if not args.sim:
      missing_args = []
      if not args.port:
          missing_args.append("--port (or config.DEFAULT_PORT)")
      if not args.robot_id:
          missing_args.append("--robot-id (or config.DEFAULT_ROBOT_ID)")
      
      if missing_args:
          print(f"\n❌ Error: Missing required hardware arguments: {', '.join(missing_args)}")
          print("   Please provide them via command line OR set them in config.py.")
          sys.exit(1)

  main(args)
