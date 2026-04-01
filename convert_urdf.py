#!/usr/bin/env python3
"""
URDF to MuJoCo MJCF Converter
Converts go_bdx.urdf to go_bdx.xml with proper physics
"""

import mujoco
import re
import os

def fix_urdf_content(content):
    """Fix common URDF issues that break MuJoCo"""
    
    # Fix huge/invalid effort and velocity values
    def fix_limit(match):
        limit_str = match.group(0)
        # Extract values
        effort_match = re.search(r'effort="([^"]+)"', limit_str)
        velocity_match = re.search(r'velocity="([^"]+)"', limit_str)
        
        if effort_match:
            try:
                effort = float(effort_match.group(1))
                if effort > 100 or effort < 0:
                    limit_str = limit_str.replace(f'effort="{effort_match.group(1)}"', 'effort="100"')
            except:
                limit_str = limit_str.replace(f'effort="{effort_match.group(1)}"', 'effort="100"')
        
        if velocity_match:
            try:
                velocity = float(velocity_match.group(1))
                if velocity > 10 or velocity < 0:
                    limit_str = limit_str.replace(f'velocity="{velocity_match.group(1)}"', 'velocity="10"')
            except:
                limit_str = limit_str.replace(f'velocity="{velocity_match.group(1)}"', 'velocity="10"')
        
        return limit_str
    
    content = re.sub(r'<limit[^>]+>', fix_limit, content)
    
    # Fix zero or tiny mass values (MuJoCo needs mass > mjMINVAL)
    def fix_mass(match):
        mass_str = match.group(1)
        try:
            mass = float(mass_str)
            if mass < 0.01:
                return 'value="0.5"'  # Set reasonable default mass
        except:
            return 'value="0.5"'
        return match.group(0)
    
    content = re.sub(r'value="([^"]*)"(?=[^>]*</mass>)', fix_mass, content)
    
    # Fix zero inertia values - look for <inertia .../> within content
    def fix_inertia_line(match):
        line = match.group(0)
        for attr in ['ixx', 'iyy', 'izz']:
            # Diagonal elements must be positive
            pattern = f'{attr}="([^"]*)"'
            m = re.search(pattern, line)
            if m:
                try:
                    val = float(m.group(1))
                    if val < 0.0001:
                        line = line.replace(f'{attr}="{m.group(1)}"', f'{attr}="0.0001"')
                except:
                    line = line.replace(f'{attr}="{m.group(1)}"', f'{attr}="0.0001"')
        for attr in ['ixy', 'ixz', 'iyz']:
            # Off-diagonal can be zero but fix if weird
            pattern = f'{attr}="([^"]*)"'
            m = re.search(pattern, line)
            if m:
                try:
                    val = float(m.group(1))
                    # Keep as is unless it's causing issues
                except:
                    line = line.replace(f'{attr}="{m.group(1)}"', f'{attr}="0.0"')
        return line
    
    content = re.sub(r'<inertia [^>]+/>', fix_inertia_line, content)
    
    return content

def load_urdf_as_mjcf(urdf_path, meshes_dir):
    """Load URDF and convert to MJCF string"""
    
    # Read and fix URDF
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    urdf_content = fix_urdf_content(urdf_content)
    
    # Load mesh files as assets
    assets = {}
    if os.path.exists(meshes_dir):
        for fname in os.listdir(meshes_dir):
            if fname.endswith(('.obj', '.stl', '.OBJ', '.STL')):
                fpath = os.path.join(meshes_dir, fname)
                with open(fpath, 'rb') as f:
                    mesh_data = f.read()
                    assets[f"meshes/{fname}"] = mesh_data
    
    print(f"    Loaded {len(assets)//2} mesh files")
    
    # Load into MuJoCo
    model = mujoco.MjModel.from_xml_string(urdf_content, assets=assets)
    
    # Save as MJCF
    mjcf_path = urdf_path.replace('.urdf', '_temp.xml')
    mujoco.mj_saveLastXML(mjcf_path, model)
    
    with open(mjcf_path, 'r') as f:
        mjcf_content = f.read()
    
    os.remove(mjcf_path)
    
    return mjcf_content, model, assets

def add_ground_and_actuators(content, model, assets):
    """Add ground plane, lights, and actuators to MJCF"""
    
    # Get list of joints (excluding freejoint if any)
    joints = []
    for i in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if jnt_name and model.jnt_type[i] == 3:  # 3 = hinge joint
            joints.append(jnt_name)
    
    # Add compiler settings for mesh directory
    if '<compiler' in content:
        # Check if meshdir already exists
        if 'meshdir=' not in content:
            content = re.sub(r'<compiler([^/]*)/>', r'<compiler\1 meshdir="meshes"/>', content)
    else:
        content = content.replace('<mujoco', '<mujoco>\n  <compiler meshdir="meshes" angle="radian"/>\n<TEMP', 1)
        content = content.replace('<TEMP', '')
    
    # Add simulation options for stability
    if '<option' not in content:
        option = '''
  <option timestep="0.002" iterations="50" solver="Newton" tolerance="1e-10"/>
'''
        content = content.replace('</compiler>', '</compiler>' + option)
    
    # Wrap robot parts in a floating base body with freejoint
    # The URDF converter puts everything directly in worldbody - we need to wrap it
    # Find the first geom after worldbody opening (that's the robot body mesh)
    worldbody_match = re.search(r'(<worldbody[^>]*>)\s*(\n\s*<geom[^/]*/>\s*\n\s*<body)', content)
    if worldbody_match:
        # Insert floating_base wrapper
        old_part = worldbody_match.group(0)
        new_part = worldbody_match.group(1) + '''
    <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
    <body name="floating_base" pos="0 0 0.3">
      <freejoint name="root"/>
''' + worldbody_match.group(2).replace('<geom', '      <geom', 1).replace('<body', '      <body', 1)
        content = content.replace(old_part, new_part)
        
        # Need to close the floating_base body before worldbody closes
        content = content.replace('</worldbody>', '    </body>\n  </worldbody>')
    else:
        # Fallback: just add ground and light
        ground_and_light = '''
    <light name="light" pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.5 0.5 0.5 1"/>
'''
        content = re.sub(r'(<worldbody[^>]*>)', r'\1' + ground_and_light, content)
    
    # CRITICAL: Widen joint ranges for hip joints and add high damping
    # Hip roll needs to work against gravity when standing
    def fix_joint_range(match):
        joint_str = match.group(0)
        name_match = re.search(r'name="([^"]+)"', joint_str)
        if not name_match:
            return joint_str
        
        jnt_name = name_match.group(1)
        
        # REMOVE actuatorfrcrange from joint - let actuator control force
        joint_str = re.sub(r'\s*actuatorfrcrange="[^"]*"', '', joint_str)
        
        # Set appropriate range for each joint type
        if 'hip_roll' in jnt_name:
            # Hip roll: wider range, HIGH damping to resist body rotation
            joint_str = re.sub(r'range="[^"]+"', 'range="-1.57 1.57"', joint_str)
            joint_str = re.sub(r'damping="[^"]+"', 'damping="10.0"', joint_str)
            if 'damping=' not in joint_str:
                joint_str = joint_str.replace('/>', ' damping="10.0"/>')
            if 'armature=' not in joint_str:
                joint_str = joint_str.replace('/>', ' armature="0.1"/>')
            else:
                joint_str = re.sub(r'armature="[^"]+"', 'armature="0.1"', joint_str)
        elif 'hip_pitch' in jnt_name:
            joint_str = re.sub(r'range="[^"]+"', 'range="-1.57 1.57"', joint_str)
            joint_str = re.sub(r'damping="[^"]+"', 'damping="10.0"', joint_str)
            if 'damping=' not in joint_str:
                joint_str = joint_str.replace('/>', ' damping="10.0"/>')
            if 'armature=' not in joint_str:
                joint_str = joint_str.replace('/>', ' armature="0.1"/>')
        elif 'hip_yaw' in jnt_name:
            joint_str = re.sub(r'range="[^"]+"', 'range="-1.57 1.57"', joint_str)
            joint_str = re.sub(r'damping="[^"]+"', 'damping="10.0"', joint_str)
            if 'damping=' not in joint_str:
                joint_str = joint_str.replace('/>', ' damping="10.0"/>')
            if 'armature=' not in joint_str:
                joint_str = joint_str.replace('/>', ' armature="0.1"/>')
        elif 'shin' in jnt_name or 'knee' in jnt_name:
            joint_str = re.sub(r'range="[^"]+"', 'range="-2.0 2.0"', joint_str)
            if 'damping=' not in joint_str:
                joint_str = joint_str.replace('/>', ' damping="5.0" armature="0.05"/>')
        else:
            # Other joints: moderate settings
            if 'damping=' not in joint_str:
                joint_str = joint_str.replace('/>', ' damping="5.0" armature="0.05"/>')
        
        return joint_str
    
    content = re.sub(r'<joint[^/]*/>', fix_joint_range, content)
    
    # Add actuators - hip_roll needs very high force to hold against gravity
    actuators = '\n  <actuator>\n'
    for joint in joints:
        if 'hip_roll' in joint:
            # Hip roll: VERY high kp and NO force limit to hold position
            actuators += f'    <position name="{joint}_motor" joint="{joint}" kp="2000" kv="100" ctrlrange="-1.5 1.5"/>\n'
        elif 'hip_pitch' in joint:
            actuators += f'    <position name="{joint}_motor" joint="{joint}" kp="1000" kv="50" ctrlrange="-1.5 1.5"/>\n'
        elif 'hip_yaw' in joint:
            actuators += f'    <position name="{joint}_motor" joint="{joint}" kp="500" kv="30" ctrlrange="-1.5 1.5"/>\n'
        elif 'shin' in joint or 'foot' in joint:
            actuators += f'    <position name="{joint}_motor" joint="{joint}" kp="300" kv="20" ctrlrange="-1.5 1.5"/>\n'
        else:
            actuators += f'    <position name="{joint}_motor" joint="{joint}" kp="100" kv="10" ctrlrange="-1.5 1.5"/>\n'
    actuators += '  </actuator>\n'
    content = content.replace('</mujoco>', actuators + '</mujoco>')
    
    print(f"    Added ground plane and {len(joints)} actuators")
    
    return content

def main():
    print("=" * 50)
    print("URDF to MuJoCo Converter")
    print("=" * 50)
    
    urdf_path = "go_bdx.urdf"
    meshes_dir = "meshes"
    output_path = "go_bdx.xml"
    
    print(f"\n[1] Reading URDF...")
    
    print(f"[2] Fixing URDF issues...")
    
    print(f"[3] Loading mesh files...")
    
    print(f"[4] Converting URDF to MuJoCo...")
    mjcf_content, model, assets = load_urdf_as_mjcf(urdf_path, meshes_dir)
    
    print(f"[5] Saving MJCF...")
    print(f"\n    Model stats:")
    print(f"    - Bodies: {model.nbody}")
    print(f"    - Joints: {model.njnt}")
    print(f"    - DOFs: {model.nv}")
    
    print(f"\n[6] Adding ground and actuators...")
    mjcf_content = add_ground_and_actuators(mjcf_content, model, assets)
    
    # Save final XML
    with open(output_path, 'w') as f:
        f.write(mjcf_content)
    
    # Verify it loads
    try:
        final_model = mujoco.MjModel.from_xml_path(output_path)
        print(f"\n{'=' * 50}")
        print(f"DONE!")
        print(f"Output: {output_path}")
        print(f"Joints: {final_model.njnt - 1}")  # -1 for freejoint
        print(f"{'=' * 50}")
        
        # Print joint list
        print(f"\nJoint list:")
        for i in range(final_model.njnt):
            jnt_name = mujoco.mj_id2name(final_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            jnt_type = final_model.jnt_type[i]
            if jnt_type == 3:  # hinge
                print(f"  {i}: {jnt_name}")
    except Exception as e:
        print(f"\nERROR: Failed to load generated XML: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
