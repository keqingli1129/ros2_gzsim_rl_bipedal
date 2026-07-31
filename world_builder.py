import os
import subprocess
import xml.etree.ElementTree as ET

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XACRO_REL_PATH = "ros2_ws/src/robot_description/robot/cart_pole.urdf.xacro"

_PRIMITIVE_GEOMETRY = {
    "base_footprint": ("box", "0.4 0.4 0.6"),
    "cart_link": ("box", "0.3 0.3 0.15"),
}

# Distinct per-link colors purely so run_inference.py's live GUI is legible
# (headless training never renders, so these have zero effect there).
_VISUAL_COLOR = {
    "base_footprint": "0.25 0.25 0.3 1",
    "cart_link": "0.85 0.45 0.05 1",
    "pole_link": "0.75 0.1 0.1 1",
}

# The pole's collision cylinder. The real pole occupies pole_link-frame
# z in [-1, 0] (the xacro puts tip_link at "0 0 -1" relative to pole_link
# and the inertial CoM at z=-0.489554, i.e. the pole hangs off the -z side
# of its own frame; pole_joint's 180-degree-about-y rotation then points
# that -z direction *up* in the world). A cylinder element is centred on
# its own origin, so it needs a -length/2 pose offset to span that same
# extent instead of straddling the joint.
_POLE_CYLINDER_RADIUS = 0.02
_POLE_CYLINDER_LENGTH = 1.0
_POLE_COLLISION_POSE = f"0 0 {-_POLE_CYLINDER_LENGTH / 2.0:g} 0 0 0"

# base_footprint's collision box is centred on the model origin, so its
# bottom face sits half a box-height below that origin. Spawning the model
# at exactly that height puts the box flush on the ground plane with zero
# interpenetration and zero drop (measured below).
_BASE_BOX_HEIGHT = float(_PRIMITIVE_GEOMETRY["base_footprint"][1].split()[2])
SPAWN_Z = _BASE_BOX_HEIGHT / 2.0


def _run_in_ros_env(cmd: str) -> str:
    """Run a shell command with a colcon-built ros2_ws sourced and the venv
    stripped from PATH. Required because xacro (ament_index_python-based)
    fails under the venv's python3 and can't resolve the robot_description
    package unless ros2_ws/install/setup.bash has been sourced."""
    script = (
        'PATH=$(echo "$PATH" | tr ":" "\\n" | grep -v "\\.venv" | paste -sd:); '
        'unset VIRTUAL_ENV; '
        f'source {REPO_ROOT}/ros2_ws/install/setup.bash; '
        + cmd
    )
    result = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd!r}\n{result.stderr}")
    return result.stdout


def run_xacro() -> str:
    return _run_in_ros_env(f"xacro {XACRO_REL_PATH}")


def convert_urdf_to_sdf(urdf_text: str, scratch_dir: str) -> str:
    tmp_urdf = os.path.join(scratch_dir, "_generated.urdf")
    with open(tmp_urdf, "w") as f:
        f.write(urdf_text)
    return _run_in_ros_env(f"gz sdf -p {tmp_urdf}")


def _set_collision_pose(collision: ET.Element, pose_text: str) -> None:
    """Set (or insert) a <collision>'s <pose>, keeping it as the first child
    so the element order stays SDF-conventional."""
    pose = collision.find("pose")
    if pose is None:
        pose = ET.Element("pose")
        collision.insert(0, pose)
    pose.text = pose_text


def postprocess_model_sdf(model_sdf_text: str) -> str:
    """Strip mesh visuals and replace mesh collision with primitives sized
    to roughly match the real robot_description meshes, and drop tip_link
    (mass 0.0001, physically negligible mount point) - keeps the physics
    parameters (mass/inertia/joint limits) sourced live from the xacro
    while collision shape stays hand-simplified, since headless training
    never renders and shouldn't pay for mesh-based collision.

    Each replaced collision primitive gets a matching <visual> added back
    (same shape/size/pose) - training pays nothing extra for it (TestFixture
    never renders), but run_inference.py's live GUI loads this exact same
    generated world, and a model with collision geometry but zero <visual>
    elements exists in the ECM/scene graph (confirmed live: the GUI's
    Entity Tree correctly lists it) while rendering as a completely empty
    viewport - there's nothing to draw."""
    root = ET.fromstring(model_sdf_text)
    model = root if root.tag == "model" else root.find("model")

    for link in list(model.findall("link")):
        name = link.get("name")
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is None:
                continue
            geometry.remove(mesh)
            visual = ET.SubElement(link, "visual", {"name": f"{name}_visual"})
            vis_geometry = ET.SubElement(visual, "geometry")
            if name == "pole_link":
                cylinder = ET.SubElement(geometry, "cylinder")
                ET.SubElement(cylinder, "radius").text = str(_POLE_CYLINDER_RADIUS)
                ET.SubElement(cylinder, "length").text = str(_POLE_CYLINDER_LENGTH)
                _set_collision_pose(collision, _POLE_COLLISION_POSE)
                vis_cylinder = ET.SubElement(vis_geometry, "cylinder")
                ET.SubElement(vis_cylinder, "radius").text = str(_POLE_CYLINDER_RADIUS)
                ET.SubElement(vis_cylinder, "length").text = str(_POLE_CYLINDER_LENGTH)
                ET.SubElement(visual, "pose").text = _POLE_COLLISION_POSE
            else:
                shape, size = _PRIMITIVE_GEOMETRY.get(name, ("box", "0.2 0.2 0.2"))
                box = ET.SubElement(geometry, shape)
                ET.SubElement(box, "size").text = size
                vis_box = ET.SubElement(vis_geometry, shape)
                ET.SubElement(vis_box, "size").text = size

            color = _VISUAL_COLOR.get(name, "0.5 0.5 0.5 1")
            material = ET.SubElement(visual, "material")
            ET.SubElement(material, "ambient").text = color
            ET.SubElement(material, "diffuse").text = color

    for joint in list(model.findall("joint")):
        if joint.get("name") == "tip_joint":
            model.remove(joint)
    for link in list(model.findall("link")):
        if link.get("name") == "tip_link":
            model.remove(link)

    return ET.tostring(model, encoding="unicode")


def wrap_in_world(model_sdf_text: str) -> str:
    """Wrap the processed <model> in a full <world> TestFixture can load.

    The model spawns at z=SPAWN_Z (0.3), which is exactly half of
    base_footprint's 0.6m-tall collision box - i.e. the box's bottom face
    lands flush on the ground plane, resting height with no drop and no
    interpenetration. Measured (2s and 6s free settle with zero applied
    force, then 30N for 200ms):

      spawn z | pole collision | rest base_z | rest pole_pitch | cart accel
      --------|----------------|-------------|-----------------|-----------
      none    | offset (fixed) | 0.006       | 0.000           |  0.10 m/s^2
      2.0     | centred (bug)  | 0.300       | 1.700 (limit!)  |  0.03 m/s^2
      2.0     | offset (fixed) | 0.300       | 0.000           | 10.27 m/s^2
      0.3     | offset (fixed) | 0.300       | 0.000           | 10.27 m/s^2

    So the lift is genuinely necessary - at z=0 the base box starts half
    buried and the contact solver never pushes an 88kg link back out
    (base_z crawls to 0.006 over 6 simulated seconds), leaving cart_joint
    jammed at ~1% of its proper authority. But z=2 is far more than needed:
    it costs a ~590ms (~118 env step) free-fall at the start of every
    episode. z=0.3 is the smallest lift that spawns already at rest -
    base_z reads exactly 0.3000 at t=1ms and never moves, and the cart
    accelerates at 10.27 m/s^2, matching effort_limit/cart_mass
    (30N/2.7kg = 11.1 m/s^2) less joint damping and friction.

    Note both the lift and the pole-collision offset are required: the
    z=2 + centred-cylinder combination in the table above lands with the
    pole's lower half buried, which drags pole_joint straight to its
    +/-1.7rad limit and pins the cart - dynamics nothing like the real
    robot's.
    """
    old = '<model name="cart_pole">'
    new = f'<model name="cart_pole"><pose>0 0 {SPAWN_Z:g} 0 0 0</pose>'
    if old not in model_sdf_text:
        raise RuntimeError(
            "could not inject the spawn pose: the exact-match anchor "
            f"{old!r} is absent from the converted model SDF (did `gz sdf` "
            "reformat its output?). Refusing to emit a world that would "
            "silently spawn the model half-buried in the ground plane."
        )
    model_sdf_text = model_sdf_text.replace(old, new)
    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="cart_pole_train">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <gui fullscreen="0">
      <!-- A <gui> block in the SDF REPLACES gz-sim's entire default GUI
           config rather than adding to it - the first version of this
           block declared only the 3D View plugin below, which silently
           dropped World Control/World Stats/Entity Tree (confirmed live:
           the GUI rendered the robot correctly but had none of those
           panels). All four are declared explicitly here so the GUI keeps
           its normal controls plus the closer camera framing. -->
      <plugin name="3D View" filename="GzScene3D">
        <gz-gui>
          <title>3D View</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.8 0.8</background_color>
        <!-- Framed close to the robot's spawn point (origin, ~1.3m tall)
             so run_inference.py's GUI opens already looking at it, instead
             of the client's default far-away view. Purely cosmetic - has
             no effect on headless training. -->
        <camera_pose>-2 -2 1.4 0 0.15 0.785</camera_pose>
      </plugin>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <title>World control</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="double" key="width">121</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="left" target="left"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <title>World stats</title>
          <property type="bool" key="showTitleBar">false</property>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">110</property>
          <property type="double" key="width">290</property>
          <property type="double" key="z">1</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="right" target="right"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time>true</real_time>
        <real_time_factor>true</real_time_factor>
        <iterations>true</iterations>
      </plugin>
      <plugin filename="EntityTree" name="Entity tree"/>
    </gui>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
      </link>
    </model>
    {model_sdf_text}
  </world>
</sdf>
"""


def generate_training_world(output_path: str) -> str:
    scratch_dir = os.path.dirname(output_path)
    urdf = run_xacro()
    model_sdf = convert_urdf_to_sdf(urdf, scratch_dir)
    processed = postprocess_model_sdf(model_sdf)
    world = wrap_in_world(processed)
    with open(output_path, "w") as f:
        f.write(world)
    return world
