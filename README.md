# GO-BDX MuJoCo Simulation

MuJoCo physics simulation for the GO-BDX bipedal robot, ready for reinforcement learning.

## Requirements

```bash
pip install mujoco numpy
```

## Quick Start

### 1. Convert URDF to MuJoCo format
```bash
python3 convert_urdf.py
```
This generates `go_bdx.xml` from the URDF with proper physics settings.

### 2. Run simulation with keyboard control
```bash
python3 keyboard_control.py
```

### 3. Simple viewer (no control)
```bash
python3 simulate.py
```

## Keyboard Controls

| Key | Action |
|-----|--------|
| `Tab` | Switch joint group |
| `Q/A` | Joint 1 increase/decrease |
| `W/S` | Joint 2 increase/decrease |
| `E/D` | Joint 3 increase/decrease |
| `R` | Reset to initial pose |
| `ESC` | Quit |

**Joint Groups:**
1. Left Hip (Roll/Pitch/Yaw)
2. Right Hip (Roll/Pitch/Yaw)
3. Left Leg (Shin/Foot)
4. Right Leg (Shin/Foot)
5. Head (Neck/Pitch/Yaw)

## Files

| File | Description |
|------|-------------|
| `go_bdx.urdf` | Original robot description |
| `go_bdx.xml` | MuJoCo model (generated) |
| `convert_urdf.py` | URDF → MuJoCo converter |
| `keyboard_control.py` | Interactive keyboard controller |
| `simulate.py` | Basic viewer |
| `meshes/` | Robot mesh files (.obj) |

## Robot Specs

- **15 joints** (6 hip, 4 leg, 5 head/neck)
- **Floating base** with 6 DOF freejoint
- **Position-controlled actuators** with PD control

### Joint Limits

| Joint | Range |
|-------|-------|
| Hip Roll/Pitch/Yaw | ±90° |
| Shin | ±115° |
| Others | ±45° |

## For Reinforcement Learning

The simulation is set up for RL integration:

```python
import mujoco

# Load model
model = mujoco.MjModel.from_xml_path("go_bdx.xml")
data = mujoco.MjData(model)

# Control loop
data.ctrl[:] = action  # Set joint targets (15 values, range [-1.5, 1.5] rad)
mujoco.mj_step(model, data)

# Observations
joint_pos = data.qpos[7:]  # Skip freejoint (7 values)
joint_vel = data.qvel[6:]  # Skip freejoint (6 values)
```

## Troubleshooting

**Robot vibrates:** Increase damping in `convert_urdf.py`

**Joints don't move:** Check actuator `kp` values in `convert_urdf.py`

**Robot falls over:** Adjust initial pose in `keyboard_control.py`

## License

MIT
