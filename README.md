# Vision-Guided Manipulation with Gemini & Robot Arms

This repository contains a demonstration of a Vision-Language-Action (VLA)
system using the SO-101 robot arm, an attached USB camera, and the Gemini
Robotics ER 1.5 model for zero-shot object detection and pointing.

## 1. Quick Start

1.  **Clone the repository:**
    ```bash
    git clone <repo_url>
    cd robotics-pointing-sample
    ```

2.  **Run the setup script:**
    ```bash
    bash setup.sh
    ```

3.  **Activate the environment:**
    ```bash
    conda activate gemini-robotics-pointing
    ```

### Manual Setup (Alternative)

If you prefer to set up manually:

1.  **Create Environment:**
    ```bash
    conda create -y -n gemini-robotics-pointing python=3.10
    conda activate gemini-robotics-pointing
    ```

2.  **Install ffmpeg:**
    ```bash
    conda install ffmpeg -c conda-forge
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Configuration

You can avoid typing long command-line arguments by setting your hardware details and API key in `config.py`.

1.  Open `config.py`.
2.  Set `GOOGLE_API_KEY` to your API key.
3.  Set `DEFAULT_PORT` and `DEFAULT_ROBOT_ID` to match your hardware.

Example `config.py`:
```python
GOOGLE_API_KEY = "AIzaSy..."
DEFAULT_PORT = "/dev/tty.usbmodem1101"
DEFAULT_ROBOT_ID = "so101_follower"
```

Now you can run the script simply:
```bash
python workshop.py
```

## 3. Run the script

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
    <td style="background-color: #f8fafd"><strong>Default (from config.py)</strong>
    </td>
    <td style="background-color: #f8fafd"><strong>Required?</strong>
    </td>
  </tr>
  <tr>
    <td style="background-color: #f8fafd">--api-key
    </td>
    <td style="background-color: #f8fafd">Your Google AI Studio API Key.
    </td>
    <td style="background-color: #f8fafd"><code>GOOGLE_API_KEY</code>
    </td>
    <td style="background-color: #f8fafd">Yes (CLI or Config)
    </td>
  </tr>
  <tr>
    <td style="background-color: #f8fafd">--port
    </td>
    <td style="background-color: #f8fafd">Serial port connected to the robot arm.
    </td>
    <td style="background-color: #f8fafd"><code>DEFAULT_PORT</code>
    </td>
    <td style="background-color: #f8fafd">Yes (CLI or Config)
    </td>
  </tr>
  <tr>
    <td style="background-color: #f8fafd">--robot-id
    </td>
    <td style="background-color: #f8fafd">ID of the robot arm.
    </td>
    <td style="background-color: #f8fafd"><code>DEFAULT_ROBOT_ID</code>
    </td>
    <td style="background-color: #f8fafd">Yes (CLI or Config)
    </td>
  </tr>
  <tr>
    <td style="background-color: #f8fafd">--camera-index
    </td>
    <td style="background-color: #f8fafd">Index of the USB camera.
    </td>
    <td style="background-color: #f8fafd"><code>DEFAULT_CAMERA_INDEX</code>
    </td>
    <td style="background-color: #f8fafd">No
    </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--calibration-dir
   </td>
   <td style="background-color: #f8fafd">Directory containing the arm calibration files (e.g., <code>./calib</code>). The file must be named <code>{robot_id}.json</code> inside this directory.
   </td>
   <td style="background-color: #f8fafd"><code>DEFAULT_CALIBRATION_DIR</code>
   </td>
   <td style="background-color: #f8fafd">No
   </td>
  </tr>
  <tr>
   <td style="background-color: #f8fafd">--backend
   </td>
   <td style="background-color: #f8fafd">Kinematics solver: lerobot, argo, or mujoco.
   </td>
   <td style="background-color: #f8fafd">argo
   </td>
   <td style="background-color: #f8fafd">No
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

## 3. Troubleshooting

| Issue | Possible Cause | Solution |
| :--- | :--- | :--- |
| **Camera not found** | Wrong index or permission denied | Try `--camera-index 1` or check OS permissions. |
| **Robot not moving** | Port error or torque disabled | Check `--port` (use `lerobot-find-port`). Ensure power is on. |
| **Calibration fails** | Board obscured or lighting | Ensure ChArUco board is fully visible and well-lit. |
| **Gemini Error** | Invalid API Key or Quota | Check your API key and quota in AI Studio. |

## 4. Hackathon Challenges

Want to extend this project? Try these challenges:

1.  **Pick & Place**: Instead of just pointing, try to grasp the object!
2.  **Voice Control**: Integrate a speech-to-text library to control the robot with your voice.
3.  **Multi-Step Reasoning**: Ask Gemini to "point to the object to the left of the red block".
