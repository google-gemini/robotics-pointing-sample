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
            # Try to load XML version if available (often fixes path issues)
            xml_path = str(Path(self.model_path).with_suffix('.xml'))
            if os.path.exists(xml_path):
                print(f"   ℹ️ Loading MuJoCo model from: {xml_path}")
                self.model = mujoco.MjModel.from_xml_path(xml_path)
            else:
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
                
            self.data = mujoco.MjData(self.model)
            
            # IK Solver Setup
            # Map config joint names (e.g. "shoulder_pan.pos") to MuJoCo names ("shoulder_pan")
            self.mj_joint_names = [n.split('.')[0] for n in config.JOINT_NAMES[:-1]] # Exclude gripper
            
            self.dof_ids = []
            for n in self.mj_joint_names:
                id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                if id == -1:
                    print(f"   ⚠️ Warning: Joint '{n}' not found in MuJoCo model.")
                self.dof_ids.append(id)
                
            self.qpos_indices = [self.model.jnt_qposadr[i] for i in self.dof_ids]
            self.dof_indices = [self.model.jnt_dofadr[i] for i in self.dof_ids]

            # Look for end-effector site
            self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
            self.use_site = (self.site_id != -1)
            if not self.use_site:
                 self.site_id = self.model.nbody - 1 # Fallback to last body
                 print("   ℹ️ 'gripperframe' site not found, using last body as EE.")

            # Solver parameters
            self.ik_iterations = 20
            self.ik_damping = 0.15
            self.ik_step_size = 0.5
            self.jac = np.zeros((6, self.model.nv))
            self.err = np.zeros(6)
            
            print("   ✅ MuJoCo Kinematics ready.")
        except Exception as e:
            print(f"   ❌ Failed to load URDF for MuJoCo: {e}")

    def _setup_lerobot(self):
        """Sets up the official LeRobot kinematics solver."""
        try:
            from lerobot.model.kinematics import RobotKinematics
            self.solver = RobotKinematics(urdf_path=self.model_path, target_frame_name=self.ee_link)
            print("   ✅ LeRobot Kinematics ready.")
        except ImportError:
            print("   ❌ Failed to import RobotKinematics from lerobot.model.kinematics.")
            print("      Ensure you have the latest version of lerobot installed.")
        except Exception as e:
            print(f"   ❌ Failed to setup LeRobot IK: {e}")

    def compute_ik(self, current_joints, target_pose_matrix, use_orientation=True):
        """
        Computes joint angles to reach target_pose_matrix (4x4).
        Returns: np.array(6) of joint angles in DEGREES, or None if failed.
        """
        q_sol = None
        
        if self.backend == "lerobot" and self.solver:
            try:
                # LeRobot IK expects degrees and returns degrees
                orientation_weight = 1.0 if use_orientation else 0.0
                q_sol = self.solver.inverse_kinematics(
                    current_joints, 
                    target_pose_matrix,
                    orientation_weight=orientation_weight
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
                    use_orientation=use_orientation,
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
            # MuJoCo IK (Jacobian Pseudo-Inverse)
            try:
                # 1. Extract Target Pos and Quat from Matrix
                target_pos = target_pose_matrix[:3, 3]
                # Convert rotation matrix to quaternion [w, x, y, z]
                # MuJoCo uses [w, x, y, z] convention
                from scipy.spatial.transform import Rotation as R
                r = R.from_matrix(target_pose_matrix[:3, :3])
                # scipy returns [x, y, z, w], need to swap to [w, x, y, z]
                x, y, z, w = r.as_quat()
                target_quat = np.array([w, x, y, z])

                # 2. Set initial joint state (radians)
                # Note: MuJoCo expects radians. current_joints is degrees.
                # Also need to map to correct indices.
                current_joints_rad = np.deg2rad(current_joints)
                for i, q_idx in enumerate(self.qpos_indices):
                    # Assuming current_joints order matches config.JOINT_NAMES order
                    self.data.qpos[q_idx] = current_joints_rad[i]

                # 3. Iterative Solve
                for _ in range(self.ik_iterations):
                    mujoco.mj_forward(self.model, self.data)

                    # Get current pose
                    if self.use_site:
                        curr_pos = self.data.site_xpos[self.site_id]
                        curr_quat = np.zeros(4)
                        mujoco.mju_mat2Quat(curr_quat, self.data.site_xmat[self.site_id])
                    else:
                        curr_pos = self.data.xpos[self.site_id]
                        curr_quat = self.data.xquat[self.site_id]

                    # Calculate error
                    self.err[:3] = target_pos - curr_pos
                    neg_quat = np.array([curr_quat[0], -curr_quat[1], -curr_quat[2], -curr_quat[3]])
                    err_quat = np.zeros(4)
                    mujoco.mju_mulQuat(err_quat, target_quat, neg_quat)
                    if err_quat[0] < 0: err_quat = -err_quat
                    
                    # Orientation error (scaled)
                    if use_orientation:
                        self.err[3:] = err_quat[1:] * (2 / np.sinc(np.arccos(np.clip(err_quat[0],-1,1))/np.pi))
                    else:
                        self.err[3:] = 0

                    if np.linalg.norm(self.err) < 1e-4: break

                    # Calculate Jacobian
                    if self.use_site:
                        mujoco.mj_jacSite(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)
                    else:
                        mujoco.mj_jacBody(self.model, self.data, self.jac[:3], self.jac[3:], self.site_id)

                    # Solve
                    J = self.jac[:, self.dof_indices]
                    H = J.T @ J + np.eye(len(self.dof_ids)) * self.ik_damping
                    delta_q = np.linalg.solve(H, J.T @ self.err)

                    # Update
                    for i, q_idx in enumerate(self.qpos_indices):
                        val = self.data.qpos[q_idx] + self.ik_step_size * delta_q[i]
                        # Clip to limits
                        limit_id = self.dof_ids[i]
                        val = np.clip(val, *self.model.jnt_range[limit_id])
                        self.data.qpos[q_idx] = val
                
                # 4. Extract Result (Degrees)
                sol_rad = [self.data.qpos[i] for i in self.qpos_indices]
                q_sol = np.rad2deg(sol_rad)

            except Exception as e:
                print(f"   ❌ MuJoCo IK failed: {e}")
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


def perform_move(bot, engine, target_xyz, gripper_pos=0, duration=1.5, use_orientation=False):
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
  q_target_arm_full = engine.compute_ik(q_current, target_pose, use_orientation=use_orientation)

  if q_target_arm_full is None:
    print(f"❌ Unreachable Target: {np.round(target_xyz, 3)}")
    return False

  # 4. Execute (Reuse joint mover logic)
  move_to_joints(bot, q_target_arm_full, gripper_pos, duration)
  return True
