#!/usr/bin/env python3
"""Simple MuJoCo viewer for go_bdx robot with keyboard control"""
import mujoco
import mujoco.viewer
import numpy as np
import time

# Load model
model = mujoco.MjModel.from_xml_path("go_bdx.xml")
data = mujoco.MjData(model)

# Print joint info
print("="*60)
print("GO_BDX Robot Simulation")
print("="*60)
print("\nJoints and actuators:")
for i in range(model.nu):
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    print(f"  {i:2d}: {act_name}")

print("\nControls:")
print("  Left/Right arrows: rotate view")
print("  Mouse drag: rotate camera")
print("  Scroll: zoom")
print("  Space: pause")
print("  Backspace: reset")
print("  ESC: quit")
print("="*60)

# Launch viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    # Set camera
    viewer.cam.distance = 1.2
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 135
    
    # Set initial pose - slightly bent knees for stability
    # Reset to known state
    mujoco.mj_resetData(model, data)
    
    # Simulation loop
    while viewer.is_running():
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Sync viewer
        viewer.sync()
        
        # Small sleep for real-time
        time.sleep(0.002)

print("Simulation ended.")
