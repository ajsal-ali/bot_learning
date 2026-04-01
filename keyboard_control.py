#!/usr/bin/env python3
"""Keyboard control for go_bdx robot - test all joints"""
import mujoco
import mujoco.viewer
import numpy as np
import time

model = mujoco.MjModel.from_xml_path("go_bdx.xml")
data = mujoco.MjData(model)

# Initial joint angles to start in a good pose
INITIAL_ANGLES = {
    'left_hip_roll': 0.0,      # Now holds at 0 with strong actuator
    'left_hip_pitch': 0.2,     
    'left_hip_yaw': 0.0,
    'left_shin': -0.4,         
    'left_foot': 0.2,
    'right_hip_roll': 0.0,     # Now holds at 0
    'right_hip_pitch': 0.2,
    'right_hip_yaw': 0.0,
    'right_shin': -0.4,
    'right_foot': 0.2,
}

def set_initial_pose():
    """Set initial joint angles"""
    mujoco.mj_resetData(model, data)
    
    # Set height
    data.qpos[2] = 0.25  # Z position
    
    # Set joint angles
    for jname, angle in INITIAL_ANGLES.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid != -1:
            qpos_addr = model.jnt_qposadr[jid]
            data.qpos[qpos_addr] = angle
    
    # Forward kinematics
    mujoco.mj_forward(model, data)

# Get all actuator names
actuators = {}
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    actuators[name] = i

# Current target positions - start from initial angles
targets = np.zeros(model.nu)
for jname, angle in INITIAL_ANGLES.items():
    motor_name = f"{jname}_motor"
    if motor_name in actuators:
        targets[actuators[motor_name]] = angle

# Joint groups for keyboard control
joint_groups = [
    ("Left Hip Roll/Pitch/Yaw", ['left_hip_roll_motor', 'left_hip_pitch_motor', 'left_hip_yaw_motor']),
    ("Right Hip Roll/Pitch/Yaw", ['right_hip_roll_motor', 'right_hip_pitch_motor', 'right_hip_yaw_motor']),
    ("Left Shin/Foot", ['left_shin_motor', 'left_foot_motor']),
    ("Right Shin/Foot", ['right_shin_motor', 'right_foot_motor']),
    ("Head/Neck", ['neck_motor', 'head_pitch_motor', 'head_yaw_motor']),
]
current_group = 0

print("="*70)
print("GO_BDX Keyboard Controller")
print("="*70)
print("\nKey bindings:")
print("  Tab         : Switch joint group")
print("  Q/A         : Joint 1 +/-")
print("  W/S         : Joint 2 +/-") 
print("  E/D         : Joint 3 +/-")
print("  R           : Reset to initial pose")
print("  ESC         : Quit")
print("="*70)

last_key_time = 0
def key_callback(key):
    global current_group, targets, last_key_time
    
    now = time.time()
    if now - last_key_time < 0.05:
        return
    last_key_time = now
    
    group_name, group_joints = joint_groups[current_group]
    step = 0.1
    
    if key == 258:  # Tab
        current_group = (current_group + 1) % len(joint_groups)
        print(f"\n>> {joint_groups[current_group][0]}")
    
    elif key == 81 and len(group_joints) > 0:  # Q
        aid = actuators.get(group_joints[0])
        if aid is not None:
            targets[aid] = min(1.5, targets[aid] + step)
            print(f"{group_joints[0].replace('_motor','')}: {targets[aid]:+.2f}")
    elif key == 65 and len(group_joints) > 0:  # A
        aid = actuators.get(group_joints[0])
        if aid is not None:
            targets[aid] = max(-1.5, targets[aid] - step)
            print(f"{group_joints[0].replace('_motor','')}: {targets[aid]:+.2f}")
    
    elif key == 87 and len(group_joints) > 1:  # W
        aid = actuators.get(group_joints[1])
        if aid is not None:
            targets[aid] = min(1.5, targets[aid] + step)
            print(f"{group_joints[1].replace('_motor','')}: {targets[aid]:+.2f}")
    elif key == 83 and len(group_joints) > 1:  # S
        aid = actuators.get(group_joints[1])
        if aid is not None:
            targets[aid] = max(-1.5, targets[aid] - step)
            print(f"{group_joints[1].replace('_motor','')}: {targets[aid]:+.2f}")
    
    elif key == 69 and len(group_joints) > 2:  # E
        aid = actuators.get(group_joints[2])
        if aid is not None:
            targets[aid] = min(1.5, targets[aid] + step)
            print(f"{group_joints[2].replace('_motor','')}: {targets[aid]:+.2f}")
    elif key == 68 and len(group_joints) > 2:  # D
        aid = actuators.get(group_joints[2])
        if aid is not None:
            targets[aid] = max(-1.5, targets[aid] - step)
            print(f"{group_joints[2].replace('_motor','')}: {targets[aid]:+.2f}")
    
    elif key == 82:  # R
        for jname, angle in INITIAL_ANGLES.items():
            motor_name = f"{jname}_motor"
            if motor_name in actuators:
                targets[actuators[motor_name]] = angle
        set_initial_pose()
        print("Reset to initial pose")

# Set initial pose
set_initial_pose()

print(f"\nStarting with: {joint_groups[current_group][0]}")
print("Press Tab to switch groups, Q/A W/S E/D to control joints")

with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    viewer.cam.distance = 1.2
    viewer.cam.elevation = -15
    viewer.cam.azimuth = 135
    
    while viewer.is_running():
        data.ctrl[:] = targets
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.002)

print("Done!")
