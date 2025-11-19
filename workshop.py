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
try:
  # Import SO101 specific classes
  from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
  from lerobot.robots.so101_follower.so101_follower import SO101Follower
except ImportError:
  print("⚠️ LeRobot not installed. Hardware connection will fail if not in sim mode.")
  # Define dummies to prevent NameError in class definition or usage
  SO101FollowerConfig = None
  SO101Follower = None
import numpy as np
from PIL import Image
import requests

import config
import robot_utils
from robot_utils import KinematicsEngine, move_to_joints, perform_move

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




# --- Main Logic Functions ---

def get_object_center_gemini(client, image_bgr, target_name):
  """Queries Gemini to find the pixel coordinates of a target object.

  Args:
      client: The configured Google GenAI client.
      image_bgr: The input image in BGR format (OpenCV default).
      target_name: The name/description of the object to find.

  Returns:
      np.array([x, y]): The pixel coordinates of the object center, or None if
      not found.
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
  """Performs hand-eye calibration using a ChArUco board.

  Detects the board in the camera view, calculates the homography matrix
  mapping image pixels to the robot's coordinate system (meters), and saves
  the result to a file.

  Args:
      cap: The OpenCV VideoCapture object.
      board_origin_robot_m: A tuple/list (x, y) representing the physical
        coordinates of the board's anchor corner (ID 0) in the robot's frame.

  Returns:
      tuple: (homography_matrix, z_surface_height)
          - homography_matrix: 3x3 numpy array for perspective transform.
          - z_surface_height: The Z-height of the calibration surface (usually 0).
          Returns (None, None) if calibration fails.
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
  """Main execution loop for the robotics pointing demo.

  Handles:
  1. Hardware/Simulation setup (Robot, Camera, Kinematics).
  2. Gemini API configuration.
  3. System calibration (loading or performing).
  4. Interactive loop:
     - Shows live video feed.
     - Captures image on user input (SPACE).
     - Sends image + prompt to Gemini.
     - Converts result to robot coordinates.
     - Moves robot to point at the target.

  Args:
      args: Parsed command-line arguments.
  """
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
      robot = None
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

  # State for persistent visualization
  last_target = None # (pixel_center, target_name)

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
    
    # Draw Last Target (Persistent)
    if last_target:
        px, name = last_target
        cv2.circle(disp, tuple(px), 10, (0, 255, 0), 2)
        cv2.drawMarker(disp, tuple(px), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(disp, f"Target: {name}", (20, 150), 
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
        # Update Persistent Target
        last_target = (pixel_center, target_name)
        
        # Visualize Result (Immediate)
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
          move_to_joints(robot, config.HOME_POSE, duration=config.MOVE_DURATION_HOME)
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
           # No sleep needed, loop continues and draws target

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
