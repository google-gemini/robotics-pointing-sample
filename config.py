import cv2
import numpy as np

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================

# --- Required / Hardware-Specific ---
# Set these to your specific hardware values to avoid passing them as arguments every time.
# If set to None, you MUST provide them via command-line arguments.
GOOGLE_API_KEY = None       # e.g., "AIzaSy..."
DEFAULT_PORT = None         # e.g., "/dev/tty.usbmodem12345"
DEFAULT_ROBOT_ID = None     # e.g., "so101_follower"

# --- Optional / Defaults ---
DEFAULT_CAMERA_INDEX = 0
DEFAULT_BACKEND = "argo"

# Path to the directory containing the robot calibration file (e.g., "my_calib_dir").
# The file must be named "{DEFAULT_ROBOT_ID}.json" inside this directory.
# Leave as None to use the default LeRobot calibration location.
DEFAULT_CALIBRATION_DIR = None 

# ==============================================================================
# ADVANCED CONFIGURATION
# ==============================================================================

# --- Robot Constants ---
# Joint names for the SO-101 arm
JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

# Sensible "Home" pose for arm in degrees
HOME_POSE = np.array([0.0, -30.0, -30.0, 75.0, -60.0, 0.0])

# --- Calibration Configuration ---
CALIBRATION_FILE = "homography_calibration.npy"
DEFAULT_BOARD_ORIGIN = [0.29, 0.0525]  # X (Forward), Y (Left) in meters

# ChArUco Board Configuration (Must match your physical board)
SQUARES_X = 5
SQUARES_Y = 7
SQUARE_LENGTH = 0.035  # meters
MARKER_LENGTH = 0.026  # meters
DICT_TYPE = cv2.aruco.DICT_4X4_250

# --- Action Configuration ---
HOVER_HEIGHT = 0.10  # Meters above table
POINT_HEIGHT = 0.02  # Meters above table
MOVE_DURATION_HOME = 2.0  # Seconds
MOVE_DURATION_HOVER = 1.5 # Seconds
MOVE_DURATION_POINT = 1.0 # Seconds
CONTROL_FREQUENCY = 50    # Hz (Control loop frequency)
