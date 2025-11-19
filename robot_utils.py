import os
import sys
import time
import requests
import numpy as np
from pathlib import Path

# Import config for defaults
import config

# Optional imports for Kinematics backends
try:
    import placo
except ImportError:
    placo = None

try:
    import mujoco
except ImportError:
    mujoco = None



def download_project_files(repo_path, file_list, target_dir):
    """
    Downloads files from a GitHub repository to a local directory.
    
    Args:
        repo_path: String in format "user/repo/branch/base_dir" 
                   (e.g. "TheRobotStudio/SO-ARM100/main/Simulation/SO101")
        file_list: List of relative file paths to download.
        target_dir: Local directory to save files to.
    """
    base_url = f"https://raw.githubusercontent.com/{repo_path}"
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"⬇️ Checking assets in '{target_dir}'...")

    for rel_path in file_list:
        file_path = target_path / rel_path
        if file_path.exists():
            continue

        # Create parent dirs if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"{base_url}/{rel_path}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print(f"   ⬇️ Downloaded: {rel_path}")
            else:
                print(f"   ❌ Failed to download {rel_path} (Status {response.status_code})")
        except Exception as e:
            print(f"   ❌ Failed to download {rel_path}: {e}")

    print("   ✅ Asset check complete.")


class KinematicsEngine:
    """Handles Inverse Kinematics (IK) using different backends."""

    def __init__(self, backend="argo", model_path="third_party/SO101/so101_new_calib.urdf"):
        self.backend = backend
        self.model_path = model_path
        self.solver = None
        self.model = None
        self.data = None  # For MuJoCo
        self.ee_link = "gripper_frame_link"  # End-effector link name

        print(f"\n⚙️ Initializing Kinematics with backend: {backend.upper()}...")

        if backend == "argo":
            self._setup_argo()
        elif backend == "mujoco":
            self._setup_mujoco()
        elif backend == "lerobot":
            self._setup_lerobot()
        else:
            print(f"❌ Unknown backend: {backend}")

    def _setup_argo(self):
        """Sets up the custom 'Argo' control library/solver."""
        argo_controls_path = Path(__file__).parent / "third_party/Argo-Robot/controls"
        
        if str(argo_controls_path) not in sys.path:
            sys.path.append(str(argo_controls_path))
            
        try:
            from scripts.model import URDF_loader, RobotModel
            from scripts.kinematics import URDF_Kinematics
        except ImportError as e:
            print(f"❌ Failed to import Argo scripts: {e}")
            print("   Ensure 'urchin' and 'scipy' are installed.")
            return

        try:
            loader = URDF_loader()
            loader.load(self.model_path)
            self.argo_model = RobotModel(loader)
            self.solver = URDF_Kinematics()
            print("   ✅ Argo Kinematics ready.")
        except Exception as e:
            print(f"   ❌ Failed to load URDF for Argo: {e}")

    def _setup_mujoco(self):
        if mujoco is None:
            print("❌ MuJoCo not installed. Install with `pip install mujoco`.")
            return
        
        try:
            self.model = mujoco.MjModel.from_xml_path(self.model_path)
            self.data = mujoco.MjData(self.model)
            print("   ✅ MuJoCo Kinematics ready.")
        except Exception as e:
            print(f"   ❌ Failed to load URDF for MuJoCo: {e}")

    def _setup_lerobot(self):
        """Sets up the official LeRobot kinematics solver."""
        try:
            from lerobot.model.kinematics import RobotKinematics
            self.solver = RobotKinematics(urdf_path=self.model_path)
            print("   ✅ LeRobot Kinematics ready.")
        except ImportError:
            print("   ❌ Failed to import RobotKinematics from lerobot.model.kinematics.")
            print("      Ensure you have the latest version of lerobot installed.")
        except Exception as e:
            print(f"   ❌ Failed to setup LeRobot IK: {e}")

    def compute_ik(self, current_joints, target_pose_matrix):
        """
        Computes joint angles to reach target_pose_matrix (4x4).
        Returns: np.array(6) of joint angles in DEGREES, or None if failed.
        """
        q_sol = None
        
        if self.backend == "lerobot" and self.solver:
            try:
                # LeRobot IK expects degrees and returns degrees (based on user snippet)
                q_sol = self.solver.inverse_kinematics(
                    current_joints, target_pose_matrix
                )
            except Exception as e:
                print(f"   ❌ LeRobot IK failed: {e}")
                return None

        elif self.backend == "argo" and self.solver:
            # Argo IK
            
            # Argo expects radians and reversed joint order
            q_start = np.deg2rad(current_joints)[::-1]
            
            try:
                q_sol_rad = self.solver.inverse_kinematics(
                    self.argo_model,
                    q_start,
                    target_pose_matrix,
                    target_link_name=self.ee_link,
                    use_orientation=False,
                    k=0.8,
                    n_iter=50
                )
                
                if q_sol_rad is not None:
                    # Reverse back and convert to degrees
                    q_sol = np.rad2deg(q_sol_rad[::-1])
                else:
                    return None
            except Exception as e:
                print(f"   ❌ Argo IK failed: {e}")
                return None

        elif self.backend == "mujoco" and self.model:
            # MuJoCo IK (Differential IK usually)
            print("   ⚠️ MuJoCo IK not fully implemented in this snippet.")
            return None
            
        # Check if gripper pos is missing and restore if needed (LeRobot/Argo only compute 5 arm joints)
        if q_sol is not None and len(q_sol) == 5:
            # Append current gripper position to the 5 arm joint solutions
            q_sol = np.append(q_sol, current_joints[-1])

        return q_sol


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
  # Simple orientation: Pointing down? Or keeping current orientation?
  # For "pointing", usually we want the gripper pointing down or forward.
  # Let's assume a fixed orientation for now (Identity rotation + translation)
  # Or better: Look at the target.
  target_pose = np.eye(4)
  target_pose[:3, 3] = target_xyz
  
  # Rotate gripper to point down (if Z is up)
  # This depends heavily on the arm's zero pose. 
  # For SO-101, Home is usually up/forward.
  # Let's try to maintain a "natural" pointing angle.
  # For now, we pass Identity rotation (which might be wrong for pointing down).
  # Ideally we want the Z-axis of the EE to point towards the target or down.
  
  # 3. Compute IK
  q_target_arm_full = engine.compute_ik(q_current, target_pose)

  if q_target_arm_full is None:
    print(f"❌ Unreachable Target: {np.round(target_xyz, 3)}")
    return False

  # 4. Execute (Reuse joint mover logic)
  move_to_joints(bot, q_target_arm_full, gripper_pos, duration)
  return True
