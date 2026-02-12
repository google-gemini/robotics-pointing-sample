# Vision-Guided Manipulation with Gemini & Robot Arms

This repository contains a demonstration of Language-Guided Manipulation using the SO-101 robot arm. It utilizes Gemini (gemini-3-flash-preview by default or gemini-robotics-er-1.5-preview) to enable zero-shot detection and pointing. Gemini is used to process images and return the 2D pixel coordinates of objects in the query. Then, the script uses a kinematics module to calculate the joint angles required to reach a target position in 3D space.

## 1. Prerequisites

Before running the script, set up your environment and install dependencies.

### Environment setup (Linux/macOS)

The recommended way to set up your environment is based on the
[LeRobot installation instructions](https://huggingface.co/docs/lerobot/en/installation#installation).

1.  **Install Conda (Miniforge recommended):**

    ```bash
    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3-$(uname)-$(uname -m).sh
    ```

2.  **Create and Activate Environment:**

    ```bash
    conda create -y -n gemini-robotics-pointing python=3.10
    conda activate gemini-robotics-pointing
    ```

3.  **Install ffmpeg:**

    ```bash
    conda install ffmpeg -c conda-forge
    ```

4.  **Install Python dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## 2. Conceptual Walkthrough

This script (`workshop.py`) integrates vision, language, and action through into a single loop. Here is how it works:

### 1. Initialization
- **Robot**: Connects to the SO-101 arm via serial port.
- **Camera**: Opens the USB camera for video capture.
- **Gemini**: Initializes the Google GenAI client with your API key.
- **Kinematics**: Loads a MuJoCo model of the robot to calculate the joint angles required to reach specific 3D coordinates using Inverse Kinematics.

### 2. Calibration (The "Eye-Hand" Connection)
Before the robot can point at what it sees, it needs to know how pixels in the camera image relate to meters in the real world.
- The script looks for a **ChArUco board** on the table.
- It calculates a **Homography Matrix**, which maps 2D image points to 2D table coordinates.
- This calibration is saved to `homography_calibration.npy` so you don't have to recalibrate every time.

### 3. The Vision-Guided Control Loop
Once calibrated, the system enters a continuous loop:
1.  **Observe**: The camera captures a static image of the workspace.
2.  **Ask**: You type a natural language query (e.g., "Where is the blue block?").
3.  **Think (Gemini)**: The system sends the current image and your text to Gemini. Gemini analyzes the image and returns the 2D pixel coordinates of the object.
4.  **Ground**: The script uses the calibration matrix to convert those pixels into real-world robot coordinates (X, Y, Z).
5.  **Act**: The robot calculates the necessary joint angles (Inverse Kinematics) and moves its arm to point at the object.

## 3. Get configuration parameters and run the script

To identify the USB port for your robot arm, use the
[`lerobot-find-port`](https://huggingface.co/docs/lerobot/en/so101#1-find-the-usb-ports-associated-with-each-arm)
command.

To identify the correct camera index, use the
[`lerobot-find-cameras opencv`](https://huggingface.co/docs/lerobot/en/cameras#finding-your-camera)
command.

Provide configuration parameters as command-line arguments.

### Execution

Before running the script, manually position the arm so it does not block the
camera's view of the ChArUco board.

To run the script, provide your API key and adjust the other required values as
needed:

```bash
python workshop.py \
--api-key "MY_GEMINI_API_KEY" \
--port "/dev/tty.usbmodem12345" \
--robot-id "my_so101_follower" \
--camera-index 0
```

Need a Gemini API key? Go to
[Google AI Studio](https://aistudio.google.com/apikey) to get one!

To use an existing arm calibration file, include the arg `--calibration-dir`:

```bash
python workshop.py \
--calibration-dir "path/to/calib_dir" \
...rest of args...
```

This should point to the directory that contains the calibration file, not the
file itself. The filename must match the `robot-id`, e.g.
`my_so101_follower.json`.

### Calibration step

The first time you run the script, or if you use the `--recalibrate flag`, it
will perform the ChArUco calibration.

1.  **Ensure the ChArUco board is in the camera's view and unobstructed** before
    running the script.
2.  The script will display a window showing the detected board (if successful)
    and save the homography matrix to homography_calibration.npy. **Press `q` to
    close the window and continue the script.**
3.  The workshop's ChArUco board includes a registration outline which sets a
    known physical X (forward) and Y (left) distance from the robot's base to
    the Anchor Corner (ID 0) of the board. If you're placing the board
    differently, measure and provide the actual distance as the `--board-origin
    X Y` argument.

### Interactive loop

After successful calibration, the script will enter an interactive loop:

1.  The robot moves to the home position.
2.  The script prompts: "⌨️ What should I point at? (e.g., 'blue block',
    'pen'):"
3.  The camera captures an image, and the image is sent to Gemini with your
    prompt.
4.  Gemini returns the 2D pixel coordinate of the object's center.
5.  The script uses the saved Homography matrix to convert the pixel coordinate
    to real-world (X, Y) robot coordinates.
6.  The robot executes a sequence: Home -> Hover 10cm above table -> Descend to
    Point 2cm above table.
7.  The loop repeats until you type `q`.

### Key script arguments

<table>
  <tr>
   <td style="background-color: #f8fafd"><strong>Argument</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Description</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Default Value</strong>
   </td>
   <td style="background-color: #f8fafd"><strong>Required?</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--api-key
   </td>
   <td style="background-color: #f8fafd">Your Google AI Studio API Key.
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd"><strong>Yes</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--port
   </td>
   <td style="background-color: #f8fafd">Serial port connected to the robot arm.
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd"><strong>Yes</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--robot-id
   </td>
   <td style="background-color: #f8fafd">ID of the robot arm; must match calibration filename without extension.
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd"><strong>Yes</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--camera-index
   </td>
   <td style="background-color: #f8fafd">Index of the USB camera (try 0, 1, or 2).
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd"><strong>Yes</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--calibration-dir
   </td>
   <td style="background-color: #f8fafd">Directory containing the arm calibration files, when not using the default location or the lerobot-calibrate command.
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd"><strong>No</strong>
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--board-origin
   </td>
   <td style="background-color: #f8fafd">X (forward) and Y (left) robot coordinates (meters) of the ChArUco board's Anchor Corner (ID 0).
   </td>
   <td style="background-color: #f8fafd">0.29 0.0525
   </td>
   <td style="background-color: #f8fafd">No
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--recalibrate
   </td>
   <td style="background-color: #f8fafd">Flag to force recalibration.
   </td>
   <td style="background-color: #f8fafd">N/A
   </td>
   <td style="background-color: #f8fafd">No
   </td>
  </tr>
</table>
