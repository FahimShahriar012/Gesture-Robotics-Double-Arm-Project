# -*- coding: utf-8 -*-
"""
time_measurement.py
===================
PART 3 EXPERIMENTS - Exercise 1

Run 5-10 trials with different object positions.
Measures time and success for each trial.

Run:
    python time_measurement.py
"""

from __future__ import annotations

import csv
import threading
import time
import numpy as np

from pickandplace import Demo


# ================================================================
#  EXPERIMENT SETTINGS (EDIT THESE)
# ================================================================

OBJECT_NAME = "box"           # Object to pick
TARGET_X = 0.45               # Where to place (X)
TARGET_Y = 0.10               # Where to place (Y)

# Different start positions for the object (5 trials minimum)
# Each tuple is (x, y) position on table
START_POSITIONS = [
    (0.35, -0.25),   # Trial 1
    (0.50, -0.20),   # Trial 2
    (0.40, -0.15),   # Trial 3
    (0.55, -0.10),   # Trial 4
    (0.45, -0.30),   # Trial 5
    (0.38, -0.05),   # Trial 6 (optional)
    (0.52, -0.25),   # Trial 7 (optional)
    (0.42, -0.20),   # Trial 8 (optional)
    (0.48, -0.08),   # Trial 9 (optional)
    (0.35, -0.10),   # Trial 10 (optional)
]

# How many trials to run (change this to 5, 8, or 10)
NUM_TRIALS = 5   # <--- CHANGE THIS TO 5, 8, or 10

# Success tolerance (meters)
XY_TOLERANCE = 0.05   # 5cm from target is considered success

# ================================================================


def teleport_object(demo: Demo, obj: str, x: float, y: float, z: float = 0.03) -> bool:
    """Move object instantly to (x, y) position."""
    try:
        # Try to find the joint for the object
        try:
            jnt = demo.model.joint(f"{obj}_free")
        except:
            try:
                jnt = demo.model.joint(f"{obj}_joint")
            except:
                # If no free joint, just update body position
                demo.data.body(obj).xpos = [x, y, z]
                mujoco.mj_forward(demo.model, demo.data)
                return True
        
        # Move using joint
        qadr = demo.model.jnt_qposadr[jnt.id]
        vadr = demo.model.jnt_dofadr[jnt.id]
        
        demo.data.qpos[qadr] = x
        demo.data.qpos[qadr + 1] = y
        demo.data.qpos[qadr + 2] = z
        # Identity quaternion (no rotation)
        demo.data.qpos[qadr + 3] = 1.0
        demo.data.qpos[qadr + 4: qadr + 7] = [0.0, 0.0, 0.0]
        # Zero velocity
        demo.data.qvel[vadr: vadr + 6] = 0.0
        
        mujoco.mj_forward(demo.model, demo.data)
        return True
        
    except Exception as e:
        print(f"  Warning: Could not teleport - {e}")
        return False


def check_success(demo: Demo, obj: str, target_x: float, target_y: float) -> tuple[bool, float]:
    """
    Q10: Define success condition.
    
    Success = object is within 5cm of target XY AND robot released it.
    """
    try:
        body = demo.data.body(obj)
        obj_x, obj_y, obj_z = body.xpos
        
        # Calculate distance to target
        distance = np.linalg.norm([obj_x - target_x, obj_y - target_y])
        
        # Success conditions:
        # 1. Object is close to target (within XY_TOLERANCE)
        # 2. Robot is not holding the object anymore
        is_near_target = distance <= XY_TOLERANCE
        is_released = (demo.held_obj is None)
        
        success = is_near_target and is_released
        
        return success, distance
        
    except Exception as e:
        print(f"  Error checking success: {e}")
        return False, 999.0


def run_trial(demo: Demo, trial_num: int, start_x: float, start_y: float) -> dict:
    """
    Run one trial.
    Q9: Time measurement included.
    """
    print(f"\n  [Trial {trial_num}] Start: ({start_x:.2f}, {start_y:.2f}) -> Target: ({TARGET_X:.2f}, {TARGET_Y:.2f})")
    
    # Reset robot to home position
    demo.reset_home()
    time.sleep(0.2)
    
    # Teleport object to start position
    teleport_object(demo, OBJECT_NAME, start_x, start_y)
    time.sleep(0.3)
    
    # ========== Q9: TIME MEASUREMENT STARTS HERE ==========
    start_time = time.time()
    
    # Execute pick and place
    grasp_ok = demo.pick_only(target_hint=OBJECT_NAME, attempts=3)
    
    if grasp_ok:
        demo.place_xy(TARGET_X, TARGET_Y)
    
    # ========== TIME MEASUREMENT ENDS HERE ==========
    end_time = time.time()
    elapsed_time = round(end_time - start_time, 2)
    
    # Check success
    success, final_distance = check_success(demo, OBJECT_NAME, TARGET_X, TARGET_Y)
    
    # Return results
    result = {
        "trial": trial_num,
        "success": 1 if success else 0,
        "time": elapsed_time,
        "start_x": start_x,
        "start_y": start_y,
        "distance": round(final_distance, 3),
        "grasp_ok": 1 if grasp_ok else 0
    }
    
    # Print result immediately
    status = "✓ SUCCESS" if success else "✗ FAIL"
    print(f"    {status} | Time: {elapsed_time}s | Distance: {final_distance:.3f}m")
    
    return result


def print_results_table(results: list[dict]) -> None:
    """Q12: Print results in table format."""
    print("\n" + "="*50)
    print("  Q12: RESULTS TABLE")
    print("="*50)
    print(f"  {'Trial':<8} {'Success':<10} {'Time (s)':<10}")
    print("-"*35)
    
    for r in results:
        print(f"  {r['trial']:<8} {r['success']:<10} {r['time']:<10}")
    
    # Summary
    total_trials = len(results)
    successes = sum(r["success"] for r in results)
    success_rate = (successes / total_trials) * 100
    avg_time = sum(r["time"] for r in results) / total_trials
    
    print("-"*35)
    print(f"\n  Summary:")
    print(f"    Total trials: {total_trials}")
    print(f"    Successful:   {successes}")
    print(f"    Failed:       {total_trials - successes}")
    print(f"    Success rate: {success_rate:.1f}%")
    print(f"    Average time: {avg_time:.2f}s")
    print("="*50)


def save_to_csv(results: list[dict], filename: str = "trial_results.csv") -> None:
    """Save results to CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "success", "time"])
        for r in results:
            writer.writerow([r["trial"], r["success"], r["time"]])
    
    print(f"\n  Results saved to: {filename}")


def run_all_trials() -> list[dict]:
    """Run all trials based on START_POSITIONS and NUM_TRIALS."""
    
    # Use only first NUM_TRIALS positions
    positions_to_use = START_POSITIONS[:NUM_TRIALS]
    
    print("\n" + "="*60)
    print("  PART 3: EXPERIMENTS - Pick and Place")
    print("="*60)
    print(f"\n  Object: {OBJECT_NAME}")
    print(f"  Target position: ({TARGET_X}, {TARGET_Y})")
    print(f"  Number of trials: {NUM_TRIALS}")
    print(f"  Success tolerance: {XY_TOLERANCE}m")
    print("\n  Running trials...")
    
    # Create demo instance
    demo = Demo()
    demo.go_home_after_motion_task = True
    
    results = []
    
    # Run trials
    for i, (start_x, start_y) in enumerate(positions_to_use, start=1):
        result = run_trial(demo, i, start_x, start_y)
        results.append(result)
        
        # Small pause between trials
        time.sleep(0.5)
    
    # Stop demo
    demo.run = False
    demo._hold_running = False
    
    # Print table and save
    print_results_table(results)
    save_to_csv(results)
    
    return results


# ================================================================
#  MAIN
# ================================================================

if __name__ == "__main__":
    import sys
    
    # Ask user how many trials
    print("\n" + "="*50)
    print("  PART 3: TIME MEASUREMENT EXPERIMENT")
    print("="*50)
    
    try:
        num_input = input(f"\n  How many trials to run? (5-10, default=5): ").strip()
        if num_input:
            NUM_TRIALS = int(num_input)
            NUM_TRIALS = max(5, min(10, NUM_TRIALS))  # Clamp between 5-10
    except:
        NUM_TRIALS = 5
    
    print(f"\n  Running {NUM_TRIALS} trials...")
    print("  (The robot viewer window will open)")
    print("  Close the viewer window when done.\n")
    
    # Run experiments in a thread so viewer works
    demo = Demo()
    
    def run_experiment():
        positions_to_use = START_POSITIONS[:NUM_TRIALS]
        
        for i, (start_x, start_y) in enumerate(positions_to_use, start=1):
            result = run_trial(demo, i, start_x, start_y)
            # Store result (you can collect them)
            print(f"    Trial {i} complete")
            time.sleep(0.5)
        
        print_results_table([])  # Will be updated
    
    # Start viewer (main thread) and run trials
    t = threading.Thread(target=run_experiment, daemon=True)
    t.start()
    
    # Start the viewer
    demo.start()