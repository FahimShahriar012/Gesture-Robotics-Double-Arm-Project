from __future__ import annotations

import time
from dataclasses import dataclass

import glfw
import mujoco
import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class ArmSpec:
    name: str
    hand_body: str
    joint_prefix: str
    finger_prefix: str


class DualPandaDemo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = np.array([900.0, 900.0, 900.0, 40.0, 40.0, 40.0])
    ctrl_hz = 400
    sim_hz = 500
    width = 1280
    height = 720

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world_dual.xml")
        self.data = mujoco.MjData(self.model)

        self.arms = {
            "left": ArmSpec(
                name="left",
                hand_body="left_panda_hand",
                joint_prefix="left_panda_joint",
                finger_prefix="left_panda_finger_joint",
            ),
            "right": ArmSpec(
                name="right",
                hand_body="right_panda_hand",
                joint_prefix="right_panda_joint",
                finger_prefix="right_panda_finger_joint",
            ),
        }

        self.target_pos: dict[str, np.ndarray] = {}
        self.target_quat: dict[str, np.ndarray] = {}
        self.home_pos: dict[str, np.ndarray] = {}
        self.home_quat: dict[str, np.ndarray] = {}
        self._act_id_cache: dict[str, int] = {}

        for arm in self.arms:
            self.gripper(arm, open_state=True)
            self._set_home_qpos(arm)

        mujoco.mj_forward(self.model, self.data)

        for arm, spec in self.arms.items():
            hand = self.data.body(spec.hand_body)
            self.target_pos[arm] = hand.xpos.copy()
            self.target_quat[arm] = hand.xquat.copy()
            self.home_pos[arm] = hand.xpos.copy()
            self.home_quat[arm] = hand.xquat.copy()

    def _set_home_qpos(self, arm: str) -> None:
        spec = self.arms[arm]
        for i in range(1, 8):
            self.data.joint(f"{spec.joint_prefix}{i}").qpos = self.qpos0[i - 1]

    def _act_id(self, name: str) -> int:
        if name not in self._act_id_cache:
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid < 0:
                raise KeyError(f"Actuator '{name}' not found.")
            self._act_id_cache[name] = int(aid)
        return self._act_id_cache[name]

    def _set_act(self, name: str, value: float) -> None:
        self.data.ctrl[self._act_id(name)] = float(value)

    def gripper(self, arm: str, open_state: bool = True) -> None:
        spec = self.arms[arm]
        value = 0.04 if open_state else 0.0
        self._set_act(f"pos_{spec.finger_prefix}1", value)
        self._set_act(f"pos_{spec.finger_prefix}2", value)

    def control_arm(self, arm: str) -> None:
        spec = self.arms[arm]
        xpos_d = self.target_pos[arm]
        xquat_d = self.target_quat[arm]

        xpos = self.data.body(spec.hand_body).xpos
        xquat = self.data.body(spec.hand_body).xquat

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, spec.hand_body)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bodyid)

        error = np.zeros(6)
        error[:3] = xpos_d - xpos

        rot_error = np.zeros(3)
        mujoco.mju_subQuat(rot_error, xquat, xquat_d)
        mujoco.mju_rotVecQuat(rot_error, rot_error, xquat)
        error[3:] = -rot_error

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        Kp = np.diag(self.K)
        Kd = np.diag(2.0 * np.sqrt(self.K))

        for i in range(1, 8):
            joint_name = f"{spec.joint_prefix}{i}"
            dofadr = self.model.joint(joint_name).dofadr
            self.data.actuator(joint_name).ctrl = self.data.joint(joint_name).qfrc_bias
            self.data.actuator(joint_name).ctrl += J[:, dofadr].T @ Kp @ error
            self.data.actuator(joint_name).ctrl -= J[:, dofadr].T @ Kd @ v

    def step(self) -> None:
        for arm in self.arms:
            self.control_arm(arm)
        mujoco.mj_step(self.model, self.data)

    @staticmethod
    def _quat_err(q: np.ndarray, r: np.ndarray) -> float:
        dot = abs(float(np.dot(q, r)))
        dot = max(min(dot, 1.0), -1.0)
        return 2.0 * np.arccos(dot)

    def move_arm_linear(self, arm: str, target_pos: np.ndarray, duration_s: float = 1.5) -> None:
        hand = self.data.body(self.arms[arm].hand_body)
        start = hand.xpos.copy()
        quat = hand.xquat.copy()
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        for k in range(steps):
            alpha = (k + 1) / steps
            self.target_pos[arm] = (1.0 - alpha) * start + alpha * target_pos
            self.target_quat[arm] = quat
            for _ in range(max(1, int(self.sim_hz / self.ctrl_hz))):
                self.step()
            time.sleep(dt)

    def move_both_linear(
        self,
        left_target: np.ndarray,
        right_target: np.ndarray,
        duration_s: float = 1.5,
    ) -> None:
        left_start = self.data.body(self.arms["left"].hand_body).xpos.copy()
        right_start = self.data.body(self.arms["right"].hand_body).xpos.copy()
        left_quat = self.data.body(self.arms["left"].hand_body).xquat.copy()
        right_quat = self.data.body(self.arms["right"].hand_body).xquat.copy()

        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)
        for k in range(steps):
            alpha = (k + 1) / steps
            self.target_pos["left"] = (1.0 - alpha) * left_start + alpha * left_target
            self.target_pos["right"] = (1.0 - alpha) * right_start + alpha * right_target
            self.target_quat["left"] = left_quat
            self.target_quat["right"] = right_quat
            for _ in range(max(1, int(self.sim_hz / self.ctrl_hz))):
                self.step()
            time.sleep(dt)

    def wait_until_pose(
        self,
        arm: str,
        pos_tol: float = 0.01,
        ang_tol: float = 0.08,
        timeout: float = 2.0,
    ) -> bool:
        spec = self.arms[arm]
        t0 = time.time()
        while time.time() - t0 < timeout:
            hand = self.data.body(spec.hand_body)
            pos_err = float(np.linalg.norm(self.target_pos[arm] - hand.xpos))
            ang_err = float(self._quat_err(self.target_quat[arm], hand.xquat))
            if pos_err < pos_tol and ang_err < ang_tol:
                return True
            self.step()
            time.sleep(1.0 / self.sim_hz)
        return False

    def return_home(self, arm: str, duration_s: float = 1.5) -> None:
        self.move_arm_linear(arm, self.home_pos[arm], duration_s=duration_s)
        self.target_quat[arm] = self.home_quat[arm].copy()
        self.wait_until_pose(arm)

    def run_demo(self) -> None:
        left_home = self.home_pos["left"].copy()
        right_home = self.home_pos["right"].copy()

        left_mid = left_home.copy()
        left_mid[0] = _clamp(left_mid[0] + 0.22, 0.20, 0.85)
        left_mid[1] = 0.18
        left_mid[2] = max(left_mid[2], 0.34)

        right_mid = right_home.copy()
        right_mid[0] = _clamp(right_mid[0] + 0.22, 0.20, 0.85)
        right_mid[1] = -0.18
        right_mid[2] = max(right_mid[2], 0.34)

        center_left = np.array([0.58, 0.06, 0.24], dtype=float)
        center_right = np.array([0.58, -0.06, 0.24], dtype=float)

        self.move_both_linear(left_mid, right_mid, duration_s=1.6)
        self.wait_until_pose("left")
        self.wait_until_pose("right")
        self.move_both_linear(center_left, center_right, duration_s=1.8)
        self.wait_until_pose("left")
        self.wait_until_pose("right")
        self.gripper("left", open_state=False)
        self.gripper("right", open_state=False)
        for _ in range(200):
            self.step()
            time.sleep(1.0 / self.sim_hz)
        self.gripper("left", open_state=True)
        self.gripper("right", open_state=True)
        for _ in range(200):
            self.step()
            time.sleep(1.0 / self.sim_hz)
        self.return_home("left")
        self.return_home("right")

    def launch_viewer(self) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW init failed.")

        window = glfw.create_window(self.width, self.height, "Dual Panda Demo", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed.")

        glfw.make_context_current(window)
        glfw.swap_interval(1)

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0
        opt = mujoco.MjvOption()
        scene = mujoco.MjvScene(self.model, maxgeom=10000)
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.run_demo()

        while not glfw.window_should_close(window):
            viewport = mujoco.MjrRect(0, 0, self.width, self.height)
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                opt,
                None,
                cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                scene,
            )
            mujoco.mjr_render(viewport, scene, context)
            glfw.swap_buffers(window)
            glfw.poll_events()
            self.step()
            time.sleep(1.0 / self.sim_hz)

        glfw.terminate()


if __name__ == "__main__":
    demo = DualPandaDemo()
    demo.launch_viewer()
