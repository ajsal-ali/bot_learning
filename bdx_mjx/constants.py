"""Paths and model names for the GO-BDX MJX environment."""

import os

ROOT = os.path.dirname(os.path.abspath(__file__))
XML_DIR = os.path.join(ROOT, "assets", "xmls")
TRAIN_XML = os.path.join(XML_DIR, "go_bdx_train.xml")  # no meshes, for MJX
VIEW_XML = os.path.join(XML_DIR, "go_bdx_view.xml")    # same physics + meshes

# Action order == actuator order in the XML. See assets/build_model.py for
# what each CAD-named joint really does (hip_roll is yaw, hip_yaw is pitch...).
LEG_JOINTS = (
    "left_hip_roll", "left_hip_pitch", "left_hip_yaw", "left_shin", "left_foot",
    "right_hip_roll", "right_hip_pitch", "right_hip_yaw", "right_shin", "right_foot",
)
NUM_JOINTS = len(LEG_JOINTS)

# Indices into LEG_JOINTS grouped by real function.
HIP_YAW_IDS = (0, 5)    # *_hip_roll  (vertical axis)
HIP_ROLL_IDS = (1, 6)   # *_hip_pitch (forward axis)
HIP_PITCH_IDS = (2, 7)  # *_hip_yaw   (lateral axis)
KNEE_IDS = (3, 8)       # *_shin
ANKLE_IDS = (4, 9)      # *_foot

ROOT_BODY = "floating_base"
FLOOR_GEOM = "floor"
FEET_SITES = ("left_foot", "right_foot")
FEET_GEOMS = ("left_sole", "right_sole")

# Sensors (all in the IMU frame: x forward, y left, z up, unless "global").
GYRO_SENSOR = "gyro"
ACCELEROMETER_SENSOR = "accelerometer"
LOCAL_LINVEL_SENSOR = "local_linvel"
UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
FEET_POS_SENSORS = tuple(f"{s}_pos" for s in FEET_SITES)
FEET_LINVEL_SENSORS = tuple(f"{s}_global_linvel" for s in FEET_SITES)
FEET_CONTACT_SENSORS = tuple(f"{g}_floor_found" for g in FEET_GEOMS)
