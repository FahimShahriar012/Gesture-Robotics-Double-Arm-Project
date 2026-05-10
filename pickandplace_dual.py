from __future__ import annotations

import argparse
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from threading import Thread
from tkinter import messagebox, ttk

import glfw
import mujoco
import numpy as np


@dataclass(frozen=True)
class ArmSpec:
    name: str
    hand_body: str
    joint_prefix: str
    finger_pos_prefix: str


class DualDemo:
    qpos0 = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    K = np.array([420.0, 420.0, 420.0, 18.0, 18.0, 18.0])

    height, width = 780, 1320
    fps = 30
    ctrl_hz = 160
    hold_hz = 320

    speed_profiles = {
        "slow": 2.4,
        "normal": 1.8,
        "fast": 0.9,
    }

    hover_clearance = 0.14
    grasp_clearance = 0.006
    lift_height = 0.30
    held_block_offset = np.array([0.0, 0.0, 0.075], dtype=float)

    def __init__(self) -> None:
        self.model = mujoco.MjModel.from_xml_path("world_dual.xml")
        self.data = mujoco.MjData(self.model)

        self.arms = {
            "left": ArmSpec(
                name="left",
                hand_body="left_panda_hand",
                joint_prefix="left_panda_joint",
                finger_pos_prefix="pos_left_panda_finger_joint",
            ),
            "right": ArmSpec(
                name="right",
                hand_body="right_panda_hand",
                joint_prefix="right_panda_joint",
                finger_pos_prefix="pos_right_panda_finger_joint",
            ),
        }
        self.active_arm = "left"

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 0
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

        self.run = True
        self._hold_running = True
        self.keep_viewer_alive = False
        self._act_id_cache: dict[str, int] = {}
        self._console_lock = threading.Lock()
        self._motion_lock = threading.Lock()
        self._data_lock = threading.RLock()
        self.console_status = "Ready"

        self.target_pos: dict[str, np.ndarray] = {}
        self.target_quat: dict[str, np.ndarray] = {}
        self.home_pos: dict[str, np.ndarray] = {}
        self.home_quat: dict[str, np.ndarray] = {}
        self.held_object: dict[str, str | None] = {"left": None, "right": None}
        self.held_origin_qpos: dict[str, np.ndarray | None] = {"left": None, "right": None}
        self.initial_object_qpos: dict[str, np.ndarray] = {}

        for arm in self.arms:
            self.gripper(True, arm)
            self._set_home_qpos(arm)
        mujoco.mj_forward(self.model, self.data)

        for arm, spec in self.arms.items():
            hand = self.data.body(spec.hand_body)
            self.target_pos[arm] = hand.xpos.copy()
            self.target_quat[arm] = hand.xquat.copy()
            self.home_pos[arm] = hand.xpos.copy()
            self.home_quat[arm] = hand.xquat.copy()

        for obj_name in self.object_names():
            joint_name = f"{obj_name}_free"
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid >= 0:
                qadr = int(self.model.jnt_qposadr[jid])
                self.initial_object_qpos[obj_name] = self.data.qpos[qadr:qadr + 7].copy()

    def _spec(self, arm: str | None = None) -> ArmSpec:
        arm_name = self.active_arm if arm is None else str(arm).lower().strip()
        if arm_name not in self.arms:
            raise KeyError(f"Arm '{arm_name}' not found. Valid arms: {list(self.arms)}")
        return self.arms[arm_name]

    def set_active_arm(self, arm: str) -> None:
        self.active_arm = self._spec(arm).name
        self._set_status(f"Active arm: {self.active_arm}")

    def _set_status(self, text: str) -> None:
        with self._console_lock:
            self.console_status = text

    def _set_home_qpos(self, arm: str) -> None:
        spec = self._spec(arm)
        for i in range(1, 8):
            self.data.joint(f"{spec.joint_prefix}{i}").qpos = self.qpos0[i - 1]

    def _act_id(self, name: str) -> int:
        if name in self._act_id_cache:
            return self._act_id_cache[name]
        aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid < 0:
            raise KeyError(f"Actuator '{name}' not found in model.")
        self._act_id_cache[name] = int(aid)
        return int(aid)

    def _set_act(self, name: str, value: float) -> None:
        self.data.ctrl[self._act_id(name)] = float(value)

    def gripper(self, open_state: bool = True, arm: str | None = None) -> None:
        spec = self._spec(arm)
        value = 0.04 if open_state else 0.0
        self._set_act(f"{spec.finger_pos_prefix}1", value)
        self._set_act(f"{spec.finger_pos_prefix}2", value)

    def gripper_both(self, open_state: bool = True) -> None:
        for arm in self.arms:
            self.gripper(open_state, arm)
        self._set_status("Opened both grippers" if open_state else "Closed both grippers")

    def reset_blocks(self) -> None:
        with self._motion_lock:
            self.held_object = {"left": None, "right": None}
            self.held_origin_qpos = {"left": None, "right": None}
            self.gripper_both(True)
            for obj_name, qpos in self.initial_object_qpos.items():
                joint_name = f"{obj_name}_free"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid < 0:
                    continue
                qadr = int(self.model.jnt_qposadr[jid])
                vadr = int(self.model.jnt_dofadr[jid])
                self.data.qpos[qadr:qadr + 7] = qpos.copy()
                self.data.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)
            self.return_home_both("normal")
            self._set_status("Reset blocks and ready for next command")

    def control(self, arm: str) -> None:
        spec = self._spec(arm)
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

        res = np.zeros(3)
        mujoco.mju_subQuat(res, xquat, xquat_d)
        mujoco.mju_rotVecQuat(res, res, xquat)
        error[3:] = -res

        J = np.concatenate((jacp, jacr))
        v = J @ self.data.qvel
        Kp = np.diag(self.K)
        Kd = np.diag(2.0 * np.sqrt(self.K))

        for i in range(1, 8):
            joint_name = f"{spec.joint_prefix}{i}"
            dofadr = self.model.joint(joint_name).dofadr
            self.data.actuator(joint_name).ctrl = self.data.joint(joint_name).qfrc_bias
            self.data.actuator(joint_name).ctrl += (J[:, dofadr].T @ Kp @ error)
            self.data.actuator(joint_name).ctrl -= (J[:, dofadr].T @ Kd @ v)

    def step(self) -> None:
        with self._data_lock:
            for arm in self.arms:
                self.control(arm)
            mujoco.mj_step(self.model, self.data)
            self._sync_held_objects()

    def _hold_loop(self) -> None:
        dt = 1.0 / float(self.hold_hz)
        while self.run and self._hold_running:
            self.step()
            time.sleep(dt)

    def object_names(self) -> list[str]:
        return ["red_box", "green_box", "blue_box", "yellow_box", "box"]

    def _body_xy(self, body_name: str) -> np.ndarray:
        return self.data.body(body_name).xpos[:2].copy()

    def _body_top_z(self, body_name: str) -> float:
        body = self.data.body(body_name)
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        z_top = float(body.xpos[2])
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != bid:
                continue
            size = np.array(self.model.geom_size[g])
            gtype = int(self.model.geom_type[g])
            if gtype == mujoco.mjtGeom.mjGEOM_BOX and size.size >= 3:
                return float(body.xpos[2] + size[2])
        return z_top

    def _set_free_body_pose(self, body_name: str, pos: np.ndarray, quat: np.ndarray | None = None, forward: bool = True) -> None:
        joint_name = f"{body_name}_free"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return
        with self._data_lock:
            qadr = int(self.model.jnt_qposadr[jid])
            vadr = int(self.model.jnt_dofadr[jid])
            self.data.qpos[qadr:qadr + 3] = np.asarray(pos, dtype=float)
            self.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if quat is None else np.asarray(quat, dtype=float)
            self.data.qvel[vadr:vadr + 6] = 0.0
            if forward:
                mujoco.mj_forward(self.model, self.data)

    def _free_body_qpos(self, body_name: str) -> np.ndarray | None:
        joint_name = f"{body_name}_free"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return None
        qadr = int(self.model.jnt_qposadr[jid])
        with self._data_lock:
            return self.data.qpos[qadr:qadr + 7].copy()

    def _restore_free_body_qpos(self, body_name: str, qpos: np.ndarray) -> None:
        joint_name = f"{body_name}_free"
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            return
        with self._data_lock:
            qadr = int(self.model.jnt_qposadr[jid])
            vadr = int(self.model.jnt_dofadr[jid])
            self.data.qpos[qadr:qadr + 7] = np.asarray(qpos, dtype=float)
            self.data.qvel[vadr:vadr + 6] = 0.0
            mujoco.mj_forward(self.model, self.data)

    def _held_block_pos(self, arm: str, hand_pos: np.ndarray | None = None) -> np.ndarray:
        hand = self.data.body(self._spec(arm).hand_body)
        base = hand.xpos.copy() if hand_pos is None else np.asarray(hand_pos, dtype=float)
        rot = np.asarray(hand.xmat, dtype=float).reshape(3, 3)
        return base + rot @ self.held_block_offset

    def _sync_held_objects(self) -> None:
        synced = False
        for arm, obj_name in self.held_object.items():
            if obj_name is None:
                continue
            pos = self._held_block_pos(arm)
            pos[2] = max(float(pos[2]), 0.04)
            self._set_free_body_pose(obj_name, pos, forward=False)
            synced = True
        if synced:
            mujoco.mj_forward(self.model, self.data)

    def _duration_for_speed(self, speed: str, override: float | None = None) -> float:
        if override is not None:
            return max(0.2, float(override))
        return self.speed_profiles.get(str(speed).lower().strip(), self.speed_profiles["normal"])

    def _park_pose(self, arm: str) -> np.ndarray:
        y = 0.50 if arm == "left" else -0.50
        return np.array([0.25, y, 0.50], dtype=float)

    def _lane_pose(self, arm: str, obj_name: str, z: float) -> np.ndarray:
        xy = self._body_xy(obj_name)
        lane_y = max(float(xy[1]), 0.28) if arm == "left" else min(float(xy[1]), -0.28)
        return np.array([float(xy[0]), lane_y, z], dtype=float)

    def _safe_lift_pose(self, arm: str, z: float) -> np.ndarray:
        spec = self._spec(arm)
        pose = self.data.body(spec.hand_body).xpos.copy()
        pose[2] = max(float(pose[2]), z)
        return pose

    def move_arm_linear(
        self,
        arm: str,
        target_pos: np.ndarray,
        duration_s: float = 1.5,
        target_quat: np.ndarray | None = None,
    ) -> None:
        spec = self._spec(arm)
        start = self.data.body(spec.hand_body).xpos.copy()
        quat = self.data.body(spec.hand_body).xquat.copy() if target_quat is None else target_quat.copy()
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        for k in range(steps):
            t = (k + 1) / steps
            alpha = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
            self.target_pos[spec.name] = (1.0 - alpha) * start + alpha * target_pos
            self.target_quat[spec.name] = quat
            time.sleep(dt)

        self.target_pos[spec.name] = target_pos.copy()
        self.target_quat[spec.name] = quat.copy()
        self._settle_arm(spec.name, 0.12)

        self._set_status(f"Moved {spec.name} arm")

    def move_both_linear(self, left_target: np.ndarray, right_target: np.ndarray, duration_s: float = 1.5) -> None:
        left_start = self.data.body(self._spec("left").hand_body).xpos.copy()
        right_start = self.data.body(self._spec("right").hand_body).xpos.copy()
        left_quat = self.data.body(self._spec("left").hand_body).xquat.copy()
        right_quat = self.data.body(self._spec("right").hand_body).xquat.copy()
        steps = max(1, int(duration_s * self.ctrl_hz))
        dt = 1.0 / float(self.ctrl_hz)

        for k in range(steps):
            t = (k + 1) / steps
            alpha = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
            self.target_pos["left"] = (1.0 - alpha) * left_start + alpha * left_target
            self.target_pos["right"] = (1.0 - alpha) * right_start + alpha * right_target
            self.target_quat["left"] = left_quat
            self.target_quat["right"] = right_quat
            time.sleep(dt)

        self._set_status("Moved both arms")

    def _settle_arm(self, arm: str, seconds: float = 0.18) -> None:
        dt = 1.0 / float(self.ctrl_hz)
        steps = max(1, int(seconds * self.ctrl_hz))
        for _ in range(steps):
            time.sleep(dt)

    def move_both_sequential(self, left_target: np.ndarray, right_target: np.ndarray, duration_s: float = 1.5) -> None:
        self._set_status("Moving left arm first")
        self.move_arm_linear("left", left_target, duration_s=duration_s, target_quat=self.home_quat["left"])
        self._set_status("Moving right arm second")
        self.move_arm_linear("right", right_target, duration_s=duration_s, target_quat=self.home_quat["right"])
        self._set_status("Moved both arms one by one")

    def park_arm_safely(self, arm: str, speed: str = "normal") -> None:
        duration = self._duration_for_speed(speed)
        safe_z = max(self.lift_height, 0.50)
        self.move_arm_linear(arm, self._safe_lift_pose(arm, safe_z), duration_s=max(0.5, duration * 0.6), target_quat=self.home_quat[arm])
        self.move_arm_linear(arm, self._park_pose(arm), duration_s=max(0.6, duration * 0.8), target_quat=self.home_quat[arm])
        self._settle_arm(arm, 0.25)

    def move_both_to_parking(self, speed: str = "normal") -> None:
        self.park_arm_safely("left", speed)
        self.park_arm_safely("right", speed)
        self._set_status("Parked both arms in safe standby")

    def return_home(self, arm: str, speed: str = "normal") -> None:
        spec = self._spec(arm)
        self.move_arm_linear(spec.name, self.home_pos[spec.name], duration_s=self._duration_for_speed(speed), target_quat=self.home_quat[spec.name])
        self.target_quat[spec.name] = self.home_quat[spec.name].copy()

    def return_home_both(self, speed: str = "normal") -> None:
        duration = self._duration_for_speed(speed)
        self.move_both_sequential(self.home_pos["left"], self.home_pos["right"], duration_s=duration)
        self.target_quat["left"] = self.home_quat["left"].copy()
        self.target_quat["right"] = self.home_quat["right"].copy()

    def move_arm_xyz(self, arm: str, x: float, y: float, z: float, speed: str = "normal") -> None:
        self.move_arm_linear(arm, np.array([x, y, z], dtype=float), self._duration_for_speed(speed))

    def _capture_block_impl(self, arm: str, obj_name: str, speed: str = "normal", clear_other: bool = True) -> None:
        if obj_name not in self.object_names():
            raise ValueError(f"Unknown object '{obj_name}'.")

        spec = self._spec(arm)
        other_arm = "right" if arm == "left" else "left"
        duration = self._duration_for_speed(speed)
        safe_z = max(self.lift_height, 0.50)
        origin_qpos = self._free_body_qpos(obj_name)
        carry_quat = self.home_quat[spec.name]

        if clear_other:
            self.park_arm_safely(other_arm, speed)

        self.park_arm_safely(spec.name, speed)

        xy = self._body_xy(obj_name)
        z_top = self._body_top_z(obj_name)
        block_center = self.data.body(obj_name).xpos.copy()
        lane = self._lane_pose(spec.name, obj_name, safe_z)
        approach = np.array([xy[0], lane[1], z_top + self.hover_clearance], dtype=float)
        hover = np.array([xy[0], xy[1], z_top + self.hover_clearance], dtype=float)
        self.target_quat[spec.name] = carry_quat.copy()
        self._settle_arm(spec.name, 0.12)
        hand = self.data.body(spec.hand_body)
        offset_world = np.asarray(hand.xmat, dtype=float).reshape(3, 3) @ self.held_block_offset
        grasp = block_center - offset_world
        grasp[2] = max(float(grasp[2]), z_top + self.grasp_clearance)
        lift = np.array([xy[0], xy[1], safe_z], dtype=float)
        retreat = self._lane_pose(spec.name, obj_name, safe_z)

        self._set_status(f"{spec.name} arm moving to {obj_name} via safe lane")
        self.gripper(True, spec.name)
        time.sleep(0.12)
        self.move_arm_linear(spec.name, lane, duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
        self.move_arm_linear(spec.name, approach, duration_s=max(0.5, duration * 0.7), target_quat=carry_quat)
        self.move_arm_linear(spec.name, hover, duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
        self._settle_arm(spec.name, 0.25)
        self.move_arm_linear(spec.name, grasp, duration_s=max(0.45, duration * 0.6), target_quat=carry_quat)
        self._settle_arm(spec.name, 0.20)
        self.gripper(False, spec.name)
        time.sleep(0.35)
        self.held_object[spec.name] = obj_name
        self.held_origin_qpos[spec.name] = origin_qpos
        self._sync_held_objects()
        self.move_arm_linear(spec.name, lift, duration_s=max(0.6, duration * 0.8), target_quat=carry_quat)
        self.move_arm_linear(spec.name, retreat, duration_s=max(0.5, duration * 0.7), target_quat=carry_quat)
        self._set_status(f"{spec.name} arm captured {obj_name}")

    def capture_block(self, arm: str, obj_name: str, speed: str = "normal") -> None:
        with self._motion_lock:
            self._capture_block_impl(arm, obj_name, speed, clear_other=True)

    def release_block(self, arm: str, speed: str = "normal") -> None:
        with self._motion_lock:
            spec = self._spec(arm)
            obj_name = self.held_object[spec.name]
            if obj_name is None:
                self._set_status(f"{spec.name} arm is not holding a block")
                return

            duration = self._duration_for_speed(speed)
            origin_qpos = self.held_origin_qpos[spec.name]
            carry_quat = self.home_quat[spec.name]
            if origin_qpos is None:
                origin_qpos = self._free_body_qpos(obj_name)
            if origin_qpos is None:
                origin_pos = self.data.body(obj_name).xpos.copy()
                origin_qpos = np.array([origin_pos[0], origin_pos[1], origin_pos[2], 1.0, 0.0, 0.0, 0.0], dtype=float)

            origin_pos = np.asarray(origin_qpos[:3], dtype=float)
            safe_z = max(self.lift_height, 0.50)
            lane_y = max(float(origin_pos[1]), 0.28) if spec.name == "left" else min(float(origin_pos[1]), -0.28)
            lane = np.array([origin_pos[0], lane_y, safe_z], dtype=float)
            hover = np.array([origin_pos[0], origin_pos[1], safe_z], dtype=float)
            hand = self.data.body(spec.hand_body)
            offset_world = np.asarray(hand.xmat, dtype=float).reshape(3, 3) @ self.held_block_offset
            place = origin_pos - offset_world
            place[2] = max(float(place[2]), 0.11)

            self.move_arm_linear(spec.name, lane, duration_s=max(0.5, duration * 0.7), target_quat=carry_quat)
            self.move_arm_linear(spec.name, hover, duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
            self.move_arm_linear(spec.name, place, duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
            self._settle_arm(spec.name, 0.18)
            self.gripper(True, spec.name)
            time.sleep(0.25)
            self._restore_free_body_qpos(obj_name, origin_qpos)
            self.held_object[spec.name] = None
            self.held_origin_qpos[spec.name] = None
            self.move_arm_linear(spec.name, hover, duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
            self.move_arm_linear(spec.name, self._park_pose(spec.name), duration_s=max(0.5, duration * 0.8), target_quat=carry_quat)
            self._set_status(f"{spec.name} arm returned {obj_name}")

    def capture_both(self, left_obj: str, right_obj: str, speed: str = "normal") -> None:
        if left_obj == right_obj:
            raise ValueError("Choose two different boxes for Capture Both.")
        with self._motion_lock:
            self.move_both_to_parking(speed)
            self._capture_block_impl("left", left_obj, speed, clear_other=False)
            self.park_arm_safely("left", speed)
            self._capture_block_impl("right", right_obj, speed, clear_other=False)
            self.park_arm_safely("right", speed)
            self._set_status(f"Captured both safely: left={left_obj}, right={right_obj}")

    def toggle_capture_release(self, arm: str, obj_name: str, speed: str = "normal") -> None:
        """
        Capture if empty, otherwise release
        """
        if self.held_object[arm] is None:
            self.capture_block(arm, obj_name, speed)
        else:
            self.release_block(arm, speed)

    def toggle_both_capture_release(self, left_obj: str, right_obj: str, speed: str = "normal") -> None:
        """
        Toggle capture/release for both arms
        """

        left_holding = self.held_object["left"] is not None
        right_holding = self.held_object["right"] is not None

        if not left_holding and not right_holding:
            self.capture_both(left_obj, right_obj, speed)
        else:
            if left_holding:
                self.release_block("left", speed)

            if right_holding:
                self.release_block("right", speed)

    def demo_motion(self) -> None:
        self.capture_both("red_box", "green_box", "normal")
        self.return_home_both("normal")
        self.gripper_both(True)

    def render(self) -> None:
        glfw.init()
        glfw.window_hint(glfw.SAMPLES, 8)
        window = glfw.create_window(self.width, self.height, "Dual Panda Control", None, None)
        glfw.make_context_current(window)

        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100)
        opt = mujoco.MjvOption()
        pert = mujoco.MjvPerturb()
        viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        while self.run:
            if glfw.window_should_close(window):
                if self.keep_viewer_alive:
                    glfw.set_window_should_close(window, False)
                    self._set_status("Viewer stays open for long session")
                else:
                    break

            w, h = glfw.get_framebuffer_size(window)
            viewport.width, viewport.height = w, h

            with self._data_lock:
                mujoco.mjv_updateScene(self.model, self.data, opt, pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
                mujoco.mjr_render(viewport, self.scene, context)
                left_holding = self.held_object["left"] or "-"
                right_holding = self.held_object["right"] or "-"

            with self._console_lock:
                status = self.console_status
            overlay = (
                f"Active arm: {self.active_arm}\n"
                f"Status: {status}\n"
                f"Left holding: {left_holding}\n"
                f"Right holding: {right_holding}"
            )
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                overlay,
                "",
                context,
            )

            time.sleep(1.0 / self.fps)
            glfw.swap_buffers(window)
            glfw.poll_events()

        glfw.terminate()

    def start(self) -> None:
        Thread(target=self._hold_loop, daemon=True).start()
        while self.run:
            try:
                self.render()
                break
            except Exception as e:
                self._set_status(f"Viewer recovered: {e}")
                if not self.keep_viewer_alive:
                    break
                time.sleep(0.5)
        self.run = False
        self._hold_running = False


def _run_async(action, root: tk.Tk) -> None:
    def runner() -> None:
        try:
            action()
        except Exception as e:
            root.after(0, lambda: messagebox.showerror("Error", str(e)))
    Thread(target=runner, daemon=True).start()


def launch_gui(demo: DualDemo) -> None:
    root = tk.Tk()
    root.title("Dual Panda Advanced Control")
    root.geometry("980x700")
    root.minsize(980, 660)

    def close_app() -> None:
        demo.run = False
        demo._hold_running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close_app)

    object_values = demo.object_names()
    speed_values = ["slow", "normal", "fast"]

    left_vars = {
        "x": tk.StringVar(value="0.55"),
        "y": tk.StringVar(value="0.18"),
        "z": tk.StringVar(value="0.25"),
        "speed": tk.StringVar(value="normal"),
        "obj": tk.StringVar(value="red_box"),
    }
    right_vars = {
        "x": tk.StringVar(value="0.55"),
        "y": tk.StringVar(value="-0.18"),
        "z": tk.StringVar(value="0.25"),
        "speed": tk.StringVar(value="normal"),
        "obj": tk.StringVar(value="green_box"),
    }
    both_vars = {
        "left_x": tk.StringVar(value="0.58"),
        "left_y": tk.StringVar(value="0.10"),
        "left_z": tk.StringVar(value="0.25"),
        "right_x": tk.StringVar(value="0.58"),
        "right_y": tk.StringVar(value="-0.10"),
        "right_z": tk.StringVar(value="0.25"),
        "speed": tk.StringVar(value="normal"),
        "left_obj": tk.StringVar(value="red_box"),
        "right_obj": tk.StringVar(value="green_box"),
    }

    main = ttk.Frame(root, padding=10)
    main.pack(fill="both", expand=True)
    main.columnconfigure(0, weight=1)
    main.columnconfigure(1, weight=1)

    def build_arm_panel(parent, title: str, arm: str, vars_dict: dict[str, tk.StringVar], column: int) -> None:
        frame = ttk.LabelFrame(parent, text=title, padding=10)
        frame.grid(row=0, column=column, sticky="nsew", padx=8, pady=8)

        ttk.Label(frame, text="Speed:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, width=12, textvariable=vars_dict["speed"], values=speed_values, state="readonly").grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="Target box:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        ttk.Combobox(frame, width=14, textvariable=vars_dict["obj"], values=object_values, state="readonly").grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(frame, text="X:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, width=10, textvariable=vars_dict["x"]).grid(row=2, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Y:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, width=10, textvariable=vars_dict["y"]).grid(row=3, column=1, sticky="w", padx=4, pady=4)
        ttk.Label(frame, text="Z:").grid(row=4, column=0, sticky="e", padx=4, pady=4)
        ttk.Entry(frame, width=10, textvariable=vars_dict["z"]).grid(row=4, column=1, sticky="w", padx=4, pady=4)

        ttk.Button(
            frame,
            text="Move",
            command=lambda: _run_async(lambda: demo.move_arm_xyz(arm, float(vars_dict["x"].get()), float(vars_dict["y"].get()), float(vars_dict["z"].get()), vars_dict["speed"].get()), root),
        ).grid(row=5, column=0, padx=4, pady=8, sticky="ew")
        ttk.Button(frame, text="Open", command=lambda: _run_async(lambda: demo.gripper(True, arm), root)).grid(row=5, column=1, padx=4, pady=8, sticky="ew")
        ttk.Button(frame, text="Close", command=lambda: _run_async(lambda: demo.gripper(False, arm), root)).grid(row=6, column=0, padx=4, pady=8, sticky="ew")
        ttk.Button(frame, text="Home", command=lambda: _run_async(lambda: demo.return_home(arm, vars_dict["speed"].get()), root)).grid(row=6, column=1, padx=4, pady=8, sticky="ew")
        ttk.Button(frame, text="Capture Block", command=lambda: _run_async(lambda: demo.capture_block(arm, vars_dict["obj"].get(), vars_dict["speed"].get()), root)).grid(row=7, column=0, padx=4, pady=8, sticky="ew")
        ttk.Button(frame, text="Release Block", command=lambda: _run_async(lambda: demo.release_block(arm, vars_dict["speed"].get()), root)).grid(row=7, column=1, padx=4, pady=8, sticky="ew")

        ttk.Button(
            frame,
            text="Capture / Return",
            command=lambda: _run_async(
                lambda: demo.toggle_capture_release(
                    arm,
                    vars_dict["obj"].get(),
                    vars_dict["speed"].get(),
                ),
                root,
            ),
        ).grid(row=8, column=0, columnspan=2, padx=4, pady=8, sticky="ew")

    build_arm_panel(main, "Left Hand Control", "left", left_vars, 0)
    build_arm_panel(main, "Right Hand Control", "right", right_vars, 1)

    both = ttk.LabelFrame(main, text="Both Hands Control", padding=10)
    both.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)

    ttk.Label(both, text="Speed:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, width=12, textvariable=both_vars["speed"], values=speed_values, state="readonly").grid(row=0, column=1, sticky="w", padx=4, pady=4)

    ttk.Label(both, text="Left box:").grid(row=1, column=0, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, width=14, textvariable=both_vars["left_obj"], values=object_values, state="readonly").grid(row=1, column=1, sticky="w", padx=4, pady=4)
    ttk.Label(both, text="Right box:").grid(row=1, column=2, sticky="e", padx=4, pady=4)
    ttk.Combobox(both, width=14, textvariable=both_vars["right_obj"], values=object_values, state="readonly").grid(row=1, column=3, sticky="w", padx=4, pady=4)

    ttk.Label(both, text="Left X/Y/Z:").grid(row=2, column=0, sticky="e", padx=4, pady=4)
    ttk.Entry(both, width=8, textvariable=both_vars["left_x"]).grid(row=2, column=1, sticky="w", padx=2)
    ttk.Entry(both, width=8, textvariable=both_vars["left_y"]).grid(row=2, column=2, sticky="w", padx=2)
    ttk.Entry(both, width=8, textvariable=both_vars["left_z"]).grid(row=2, column=3, sticky="w", padx=2)

    ttk.Label(both, text="Right X/Y/Z:").grid(row=3, column=0, sticky="e", padx=4, pady=4)
    ttk.Entry(both, width=8, textvariable=both_vars["right_x"]).grid(row=3, column=1, sticky="w", padx=2)
    ttk.Entry(both, width=8, textvariable=both_vars["right_y"]).grid(row=3, column=2, sticky="w", padx=2)
    ttk.Entry(both, width=8, textvariable=both_vars["right_z"]).grid(row=3, column=3, sticky="w", padx=2)

    ttk.Button(
        both,
        text="Pick One By One",
        command=lambda: _run_async(lambda: demo.capture_both(
            both_vars["left_obj"].get(),
            both_vars["right_obj"].get(),
            "normal",
        ), root),
    ).grid(row=4, column=0, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Open Both", command=lambda: _run_async(lambda: demo.gripper_both(True), root)).grid(row=4, column=1, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Close Both", command=lambda: _run_async(lambda: demo.gripper_both(False), root)).grid(row=4, column=2, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Home Both", command=lambda: _run_async(lambda: demo.return_home_both(both_vars["speed"].get()), root)).grid(row=4, column=3, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Capture Blocks", command=lambda: _run_async(lambda: demo.capture_both(both_vars["left_obj"].get(), both_vars["right_obj"].get(), "normal"), root)).grid(row=5, column=0, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Release Blocks", command=lambda: _run_async(lambda: (demo.release_block("left", both_vars["speed"].get()), demo.release_block("right", both_vars["speed"].get())), root)).grid(row=5, column=1, padx=4, pady=8, sticky="ew")

    ttk.Button(
        both,
        text="Capture / Return",
        command=lambda: _run_async(
            lambda: demo.toggle_both_capture_release(
                both_vars["left_obj"].get(),
                both_vars["right_obj"].get(),
                "normal",
            ),
            root,
        ),
    ).grid(row=5, column=2, padx=4, pady=8, sticky="ew")

    ttk.Button(both, text="Demo", command=lambda: _run_async(demo.demo_motion, root)).grid(row=5, column=3, padx=4, pady=8, sticky="ew")
    ttk.Button(both, text="Reset Blocks", command=lambda: _run_async(demo.reset_blocks, root)).grid(row=6, column=0, columnspan=4, padx=4, pady=8, sticky="ew")

    help_label = ttk.Label(
        main,
        text="Each hand has its own speed, target box, move, capture, and release controls. Keep this window open and use Reset Blocks to run the task again.",
        wraplength=900,
    )
    help_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=8)

    root.mainloop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open the advanced Tk control panel.")
    parser.add_argument("--arm", choices=["left", "right", "both"], default="left")
    parser.add_argument("--x", type=float, default=None)
    parser.add_argument("--y", type=float, default=None)
    parser.add_argument("--z", type=float, default=None)
    parser.add_argument("--left-x", type=float, default=None)
    parser.add_argument("--left-y", type=float, default=None)
    parser.add_argument("--left-z", type=float, default=None)
    parser.add_argument("--right-x", type=float, default=None)
    parser.add_argument("--right-y", type=float, default=None)
    parser.add_argument("--right-z", type=float, default=None)
    parser.add_argument("--speed", choices=["slow", "normal", "fast"], default="normal")
    parser.add_argument("--gripper", choices=["open", "close"], default=None)
    parser.add_argument("--home", action="store_true")
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--left-obj", type=str, default=None)
    parser.add_argument("--right-obj", type=str, default=None)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--no-viewer", action="store_true")
    args = parser.parse_args()

    demo = DualDemo()

    def run_sequence() -> None:
        if args.demo:
            demo.demo_motion()
            return

        if args.gripper is not None:
            if args.arm == "both":
                demo.gripper_both(args.gripper == "open")
            else:
                demo.gripper(args.gripper == "open", args.arm)

        if args.home:
            if args.arm == "both":
                demo.return_home_both(args.speed)
            else:
                demo.return_home(args.arm, args.speed)

        if args.capture:
            if args.arm == "both":
                demo.capture_both(args.left_obj or "red_box", args.right_obj or "green_box", args.speed)
            else:
                demo.capture_block(args.arm, args.obj or "box", args.speed)

        if args.release:
            if args.arm == "both":
                demo.release_block("left", args.speed)
                demo.release_block("right", args.speed)
            else:
                demo.release_block(args.arm, args.speed)

        if args.arm == "both" and None not in (args.left_x, args.left_y, args.left_z, args.right_x, args.right_y, args.right_z):
            demo.move_both_sequential(
                np.array([args.left_x, args.left_y, args.left_z], dtype=float),
                np.array([args.right_x, args.right_y, args.right_z], dtype=float),
                demo._duration_for_speed(args.speed),
            )
        elif args.arm in ("left", "right") and None not in (args.x, args.y, args.z):
            demo.move_arm_xyz(args.arm, float(args.x), float(args.y), float(args.z), args.speed)

    if args.no_viewer:
        Thread(target=demo._hold_loop, daemon=True).start()
        run_sequence()
        demo.run = False
        demo._hold_running = False
    else:
        if args.gui:
            demo.keep_viewer_alive = True
            Thread(target=launch_gui, args=(demo,), daemon=True).start()
        else:
            Thread(target=run_sequence, daemon=True).start()
        demo.start()
