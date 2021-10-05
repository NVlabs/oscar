# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for OSCAR. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

"""
Set of utility functions for procedurally generating objects
"""

from oscar import ASSETS_ROOT
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import xml.etree.ElementTree as ET
from oscar.utils.torch_utils import *
import torch
import trimesh


def pretty_print_xml(current, parent=None, index=-1, depth=0, use_tabs=False):
    space = '\t' if use_tabs else ' ' * 4
    for i, node in enumerate(current):
        pretty_print_xml(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + (space * depth)
        else:
            parent[index - 1].tail = '\n' + (space * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + (space * (depth - 1))


def array_to_string(array):
    """
    Converts a numeric array into the string format in mujoco.
    Examples:
        [0, 1, 2] => "0 1 2"
    Args:
        array (n-array): Array to convert to a string
    Returns:
        str: String equivalent of @array
    """
    return " ".join(["{}".format(x) for x in array])


def convert_to_string(inp):
    """
    Converts any type of {bool, int, float, list, tuple, array, string, np.str_} into an mujoco-xml compatible string.
        Note that an input string / np.str_ results in a no-op action.
    Args:
        inp: Input to convert to string
    Returns:
        str: String equivalent of @inp
    """
    if type(inp) in {list, tuple, np.ndarray}:
        return array_to_string(inp)
    elif type(inp) in {int, float, bool, np.float32, np.float64, np.int32, np.int64}:
        return str(inp).lower()
    elif type(inp) in {str, np.str_}:
        return inp
    else:
        raise ValueError("Unsupported type received: got {}".format(type(inp)))


def create_collision_body(pos=(0, 0, 0), rpy=(0, 0, 0), shape="sphere", attribs=None):#, indent=0):
    """
    Generates XML collision body

    Args:
        pos (list or tuple or array): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or array): (r,p,y) offset rot values when creating the collision body
        shape (str): What shape to create. Must be one of {box, cylinder, sphere, mesh}
        attribs (None or dict): If specified, should contain keyword-mapped values to be used as additional attributes
            for the @shape tag

    Returns:
        ET.Element: Generated collision element
    """
    # Create the initial collision body
    col = ET.Element("collision")
    # Create origin subtag
    origin = ET.SubElement(col, "origin", attrib={"rpy": convert_to_string(rpy), "xyz": convert_to_string(pos)})
    # Create geometry subtag
    geo = ET.SubElement(col, "geometry")
    # Add shape info
    if attribs is not None:
        for k, v in attribs.items():
            attribs[k] = convert_to_string(v)
    shape = ET.SubElement(geo, shape, attrib=attribs)
    # Return this element
    return col


def create_visual_body(pos=(0, 0, 0), rpy=(0, 0, 0), shape="sphere", attribs=None):#, indent=0):
    """
    Generates XML visual body

    Args:
        pos (list or tuple or array): (x,y,z) offset pos values when creating the body
        rpy (list or tuple or array): (r,p,y) offset rot values when creating the body
        shape (str): What shape to create. Must be one of {box, cylinder, sphere, mesh}
        attribs (None or dict): If specified, should contain keyword-mapped values to be used as additional attributes
            for the @shape tag

    Returns:
        ET.Element: Generated visual element
    """
    # Create the initial body
    vis = ET.Element("visual")
    # Create origin subtag
    origin = ET.SubElement(vis, "origin", attrib={"rpy": convert_to_string(rpy), "xyz": convert_to_string(pos)})
    # Create geometry subtag
    geo = ET.SubElement(vis, "geometry")
    # Add shape info
    if attribs is not None:
        for k, v in attribs.items():
            attribs[k] = convert_to_string(v)
    shape = ET.SubElement(geo, shape, attrib=attribs)
    # Return this element
    return vis


def create_joint(name, parent, child, pos=(0, 0, 0), rpy=(0, 0, 0), joint_type="fixed",
                 axis=None, damping=None, friction=None, limits=None):
    """
    Generates XML joint

    Args:
        name (str): Name of this joint
        parent (str or ET.Element): Name of parent link or parent link element itself for this joint
        child (str or ET.Element): Name of child link or child link itself for this joint
        pos (list or tuple or array): (x,y,z) offset pos values when creating the collision body
        rpy (list or tuple or array): (r,p,y) offset rot values when creating the joint
        joint_type (str): What type of joint to create. Must be one of {fixed, revolute, prismatic}
        axis (None or 3-tuple): If specified, should be (x,y,z) axis corresponding to DOF
        damping (None or float): If specified, should be damping value to apply to joint
        friction (None or float): If specified, should be friction value to apply to joint
        limits (None or 2-tuple): If specified, should be min / max limits to the applied joint

    Returns:
        ET.Element: Generated joint element
    """
    # Create the initial joint
    jnt = ET.Element("joint", name=name, type=joint_type)
    # Create origin subtag
    origin = ET.SubElement(jnt, "origin", attrib={"rpy": convert_to_string(rpy), "xyz": convert_to_string(pos)})
    # Make sure parent and child are both names (str) -- if they're not str already, we assume it's the element ref
    if not isinstance(parent, str):
        parent = parent.get("name")
    if not isinstance(child, str):
        child = child.get("name")
    # Create parent and child subtags
    parent = ET.SubElement(jnt, "parent", link=parent)
    child = ET.SubElement(jnt, "child", link=child)
    # Add additional parameters if specified
    if axis is not None:
        ax = ET.SubElement(jnt, "axis", xyz=convert_to_string(axis))
    dynamic_params = {}
    if damping is not None:
        dynamic_params["damping"] = convert_to_string(damping)
    if friction is not None:
        dynamic_params["friction"] = convert_to_string(friction)
    if dynamic_params:
        dp = ET.SubElement(jnt, "dynamics", **dynamic_params)
    if limits is not None:
        lim = ET.SubElement(jnt, "limit", lower=limits[0], upper=limits[1])

    # Return this element
    return jnt


def create_link(name, subelements=None, mass=None):
    """
    Generates XML link element

    Args:
        name (str): Name of this link
        subelements (None or list): If specified, specifies all nested elements that should belong to this link
            (e.g.: visual, collision body elements)
        mass (None or float): If specified, will add an inertial tag with specified mass value

    Returns:
        ET.Element: Generated link
    """
    # Create the initial link
    link = ET.Element("link", name=name)
    # Add all subelements if specified
    if subelements is not None:
        for ele in subelements:
            link.append(ele)
    # Add mass subelement if requested
    if mass is not None:
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value=convert_to_string(mass))

    # Return this element
    return link


def generate_urdf_from_xmltree(elements, name, dirpath, unique_urdf=False):
    """
    Generates a URDF file corresponding to @xmltree at @dirpath with name @name.urdf.

    Args:
        elements (list of ET.Element): Elements that compose the URDF.
            NOTE: Should NOT include top-level XML tags (the `robot` and `?xml` tags)
        name (str): Name of this file (name assigned to robot tag)
        dirpath (str): Absolute path to the location / filename for the generated URDF
        unique_urdf (bool): Whether to use a unique identifier when naming urdf (uses current datetime)

    Returns:
        str: Path to newly created urdf (fpath/<name>.urdf)
    """
    # Create top level robot tag
    xml = ET.Element("robot", name=name)
    # Append each element to this xml
    for ele in elements:
        xml.append(ele)
    # Write to fpath, making sure the directory exists (if not, create it)
    Path(dirpath).mkdir(parents=True, exist_ok=True)
    # Get file
    date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
    fname = f"{name}_{date}.urdf" if unique_urdf else f"{name}.urdf"
    fpath = os.path.join(dirpath, fname)
    with open(fpath, 'w') as f:
        # Write top level header line first
        f.write('<?xml version="1.0" ?>\n')
        # Convert xml to string form and write to file
        pretty_print_xml(current=xml)
        xml_str = ET.tostring(xml, encoding="unicode")
        f.write(xml_str)

    # Return path to file
    return fpath


def create_box(name, size, mass, generate_urdf=True, unique_urdf_name=False, visual_top_site=False, from_mesh=False, asset_root_path=ASSETS_ROOT):
    """
    Procedurally generates a box whose geometry is given by @size, and with name @name.
    The object file will be generated in ASSETS_ROOT/urdf/procedural/box, and the relative path to this
    file will be returned (relative to ASSETS_ROOT).

    Note: Object state is taken to be the center of the box.

    Args:
        name (str): Name for this box object.
        size (3-array): (l, w, h) of box object.
        mass (float): Total mass of the cylinder object.
        generate_urdf (bool): If True, will generate URDF and return fpath to generated file. Otherwise, will
            directly return a list of the elements forming this box.
        unique_urdf_name (bool): Whether to create urdf that has a unique name or not
        visual_top_site (bool): If True, will add a separate body with only a visual site attached to it
        from_mesh (bool): If True, will generate box from triangular mesh
        asset_root_path (str): Absolute path to root asset path to save generated urdf

    Returns:
        str or list: If @generate_urdf is True, returns relative fpath (relative to ASSETS_ROOT) to generated URDF
            Else, returns list of elements forming this box
    """
    assert len(size) == 3, "Box size must be 3-array!"
    height = size[2]
    # Create bodies -- visual and collision is the same
    bodies = []
    body_kwargs = {
        "pos": (0, 0, 0),
    }
    if from_mesh:
        body_mesh = trimesh.creation.box(extents=size)
        date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
        suffix = date if unique_urdf_name else ""
        body_mesh_fpath = f'/tmp/box_mesh_{suffix}.stl'
        body_mesh.export(body_mesh_fpath)
        body_kwargs.update({
            "shape": "mesh",
            "attribs": {"filename": body_mesh_fpath},
        })
    else:
        body_kwargs.update({
            "shape": "box",
            "attribs": {"size": size},
        })

    for create_body in (create_collision_body, create_visual_body):
        bodies.append(create_body(**body_kwargs))

    # Create links from base bodies and wall bodies
    base_link = create_link(name=f"{name}", subelements=bodies, mass=mass)
    # Compose elements
    elements = [base_link]

    # Add visual top site if requested
    if visual_top_site:
        top_body = create_visual_body(
            pos=(0, 0, 0),
            shape="sphere",
            attribs={
                "radius": 0.001,
            }
        )
        top_link = create_link(name=f"{name}_top", subelements=[top_body])
        # Create joint linking bodies together
        jnt = create_joint(name=f"{name}_joint", parent=base_link, child=top_link,
                           pos=(0, 0, height / 1.2), joint_type="fixed")
        elements += [jnt, top_link]

    # Generate URDF if requested
    if generate_urdf:
        relative_fpath = generate_urdf_from_xmltree(
            elements=elements,
            name=name,
            dirpath=os.path.join(asset_root_path, "urdf/procedural/box"),
            unique_urdf=unique_urdf_name,
        )

        # Remove the initial ASSETPATH
        ret = relative_fpath.split(asset_root_path + "/")[-1]
    else:
        ret = elements

    # Return value
    return ret


def create_cylinder(name, size, mass, generate_urdf=True, unique_urdf_name=False, visual_top_site=False, from_mesh=False, hollow=False, asset_root_path=ASSETS_ROOT):
    """
    Procedurally generates a cylinder whose geometry is given by @size, and with name @name.
    The object file will be generated in ASSETS_ROOT/urdf/procedural/cylinder, and the relative path to this
    file will be returned (relative to ASSETS_ROOT).

    Note: Object state is taken to be the center of the cylinder.

    Args:
        name (str): Name for this cylinder object.
        size (2-array): (radius, height) of cylinder object.
        mass (float): Total mass of the cylinder object.
        generate_urdf (bool): If True, will generate URDF and return fpath to generated file. Otherwise, will
            directly return a list of the elements forming this cylinder.
        unique_urdf_name (bool): Whether to create urdf that has a unique name or not
        visual_top_site (bool): If True, will add a separate body with only a visual site attached to it
        from_mesh (bool): If True, will generate box from triangular mesh
        hollow (bool): If True, will create annulus instead of cylinder (MUST use mesh)
        asset_root_path (str): Absolute path to root asset path to save generated urdf

    Returns:
        str or list: If @generate_urdf is True, returns relative fpath (relative to ASSETS_ROOT) to generated URDF
            Else, returns list of elements forming this cylinder
    """
    radius, height = size
    # Create bodies -- visual and collision is the same
    bodies = []
    body_kwargs = {
        "pos": (0, 0, 0),
    }
    if hollow:
        assert from_mesh, "Hollow cylinder can only be generated from mesh!"

    if from_mesh:
        body_mesh = trimesh.creation.annulus(r_min=radius / 4, r_max=radius, height=height, sections=10) if hollow \
            else trimesh.creation.cylinder(radius=radius, height=height, sections=16)
        date = datetime.now().isoformat(timespec="microseconds").replace(".", "_").replace(":", "_").replace("-", "_")
        suffix = date if unique_urdf_name else ""
        body_mesh_fpath = f'/tmp/cylinder_mesh_{suffix}.stl'
        body_mesh.export(body_mesh_fpath)
        body_kwargs.update({
            "shape": "mesh",
            "attribs": {"filename": body_mesh_fpath},
        })
    else:
        body_kwargs.update({
            "shape": "cylinder",
            "attribs": {"radius": radius, "length": height},
        })

    bodies = []
    for create_body in (create_collision_body, create_visual_body):
        bodies.append(create_body(**body_kwargs))

    # Create links from base bodies and wall bodies
    base_link = create_link(name=f"{name}", subelements=bodies, mass=mass)
    # Compose elements
    elements = [base_link]

    # Add visual top site if requested
    if visual_top_site:
        top_body = create_visual_body(
            pos=(0, 0, 0),
            shape="sphere",
            attribs={
                "radius": 0.001,
            }
        )
        top_link = create_link(name=f"{name}_top", subelements=[top_body])
        # Create joint linking bodies together
        jnt = create_joint(name=f"{name}_joint", parent=base_link, child=top_link,
                           pos=(0, 0, height / 1.2), joint_type="fixed")
        elements += [jnt, top_link]

    # Generate URDF if requested
    if generate_urdf:
        relative_fpath = generate_urdf_from_xmltree(
            elements=elements,
            name=name,
            dirpath=os.path.join(asset_root_path, "urdf/procedural/cylinder"),
            unique_urdf=unique_urdf_name,
        )

        # Remove the initial ASSETPATH
        ret = relative_fpath.split(asset_root_path + "/")[-1]
    else:
        ret = elements

    # Return value
    return ret


def create_hollow_cylinder(name, size, thickness, mass, n_slices=8, shape="round", use_lid=False,
                           transparent_walls=False, generate_urdf=True,
                           unique_urdf_name=False, asset_root_path=ASSETS_ROOT):
    """
    Procedurally generates a hollow cylinder whose geometry is given by @size, and with name @name.
    The object file will be generated in ASSETS_ROOT/urdf/procedural/cylinder, and the relative path to this
    file will be returned (relative to ASSETS_ROOT).

    Note: Object state is taken to be the center of the BOTTOM surface of the cylinder.

    Args:
        name (str): Name for this cylinder object.
        size (2-array): (radius, height) of cylinder object.
        thickness (float): thickness of walls of cylinder object.
        mass (float): Total mass of the cylinder object.
        n_slices (int): Number of discrete "slices" to partition cylinder into.
        shape (str): Shape of generated cylinder. Options are "round" or "square"
        use_lid (bool): If True, will add lid geom to top of cylinder
        transparent_walls (bool): If True, will remove visual geoms for walls of cylinder
        generate_urdf (bool): If True, will generate URDF and return fpath to generated file. Otherwise, will
            directly return a list of the elements forming this cylinder.
        unique_urdf_name (bool): Whether to create urdf that has a unique name or not
        asset_root_path (str): Absolute path to root asset path to save generated urdf

    Returns:
        str or list: If @generate_urdf is True, returns relative fpath (relative to ASSETS_ROOT) to generated URDF
            Else, returns list of elements forming this cylinder
    """
    # Sanity check
    assert shape in {"round", "square"}, f"Invalid cylinder shape requested: {shape}!"
    if shape == "square":
        n_slices = 4

    radius, height = size
    if use_lid:
        base_mass, wall_mass, lid_mass = mass / (n_slices + 2), mass * n_slices / (n_slices + 2), mass / (n_slices + 2)
    else:
        base_mass, wall_mass, lid_mass = mass / (n_slices + 1), mass * n_slices / (n_slices + 1), None
    # Create initial element for the cylinder base -- visual and collision is the same
    base_bodies = []
    wall_bodies = []
    lid_bodies = []
    for create_body in (create_collision_body, create_visual_body):
        # Create base bodies
        if shape == "round":
            body_attribs = {
                "shape": "cylinder",
                "attribs": {
                    "radius": radius,
                    "length": thickness,
                }
            }
        else:       # square
            body_attribs = {
                "shape": "box",
                "attribs": {
                    "size": (radius * 2, radius * 2, thickness),
                }
            }
        base_bodies.append(create_body(pos=(0, 0, thickness / 2), **body_attribs))

        if (create_body != create_visual_body) or (not transparent_walls):
            # Create wall bodies
            wall_length = np.pi * 2 * (radius + thickness) / n_slices
            for i in range(n_slices):
                angle = np.pi * 2 * (i / n_slices)
                wall_bodies.append(
                    create_body(
                        pos=(radius * np.cos(angle), radius * np.sin(angle), 0),
                        rpy=(0, 0, angle),
                        shape="box",
                        attribs={
                            "size": (thickness, wall_length, height),
                        }
                    )
                )

        # Create lid bodies if requested
        if use_lid:
            lid_bodies.append(create_body(pos=(0, 0, -thickness / 2), **body_attribs))

    # Create links from base bodies and wall bodies
    base_link = create_link(name=f"{name}_base", subelements=base_bodies, mass=base_mass)
    wall_link = create_link(name=f"{name}_wall", subelements=wall_bodies, mass=wall_mass)
    # Create joint linking bodies together
    jnt = create_joint(name=f"{name}_joint", parent=base_link, child=wall_link,
                       pos=(0, 0, height / 2), joint_type="fixed")
    # Compose elements
    elements = [base_link, jnt, wall_link]

    # Add lid link if requested
    if use_lid:
        lid_link = create_link(name=f"{name}_lid", subelements=lid_bodies, mass=lid_mass)
        lid_jnt = create_joint(name=f"{name}_lid_joint", parent=wall_link, child=lid_link,
                           pos=(0, 0, height / 2), joint_type="fixed")
        elements += [lid_jnt, lid_link]

    # Generate URDF if requested
    if generate_urdf:
        relative_fpath = generate_urdf_from_xmltree(
            elements=elements,
            name=name,
            dirpath=os.path.join(asset_root_path, "urdf/procedural/cylinder"),
            unique_urdf=unique_urdf_name,
        )

        # Remove the initial ASSETPATH
        ret = relative_fpath.split(asset_root_path + "/")[-1]
    else:
        ret = elements

    # Return value
    return ret


@torch.jit.script
def get_sphere_positions_in_cylindrical_container(
    n_spheres,
    sphere_radius,
    tolerance,
    container_radius,
    container_height,
    container_pos,
    container_ori_vec,
):
    """
    Helper method to generate (x,y,z) positions to place N spheres deterministically within a cylindrical container

    Args:
        n_spheres (int): Number of spheres to place
        sphere_radius (float): Radius of spheres
        tolerance (float): Space to place beween spheres and container walls
        container_radius (float): Radius of cylindrical container
        container_height (float): Height of cylindrical container
        container_pos (Tensor): (n_envs, 3) (x,y,z) location of container in each env
            NOTE: This is assumed to be the BOTTOM SURFACE of the container
        container_ori_vec (Tensor): (n_envs, 3) (x,y,z) orientation vector of container representing the central
            axis of the cylinder in each env

    Returns:
        Tensor: (n_envs, n_spheres, 3) (x,y,z) location for each sphere in each env
    """
    # type: (int, float, float, float, float, Tensor, Tensor) -> Tensor
    # First create tensor that we will fill with sphere locations
    sphere_pos = torch.stack([torch.zeros_like(container_pos)] * n_spheres, dim=1)
    # For computational ease, we will simply pack the spheres in a cube-like structural form
    square_length = (container_radius - tolerance) * 1.4142135623730951
    # Find max num of spheres we can fit per row
    spheres_per_row = int(square_length // (2 * sphere_radius + tolerance))
    spheres_per_layer = spheres_per_row ** 2
    # Generate positions for each sphere -- location discrete steps of 2 * sphere_radius + tolerance
    # with initial offset of ((spheres_per_row - 1) * (2 * sphere_radius + tolerance)) / 2
    offset = -((spheres_per_row - 1) * (2 * sphere_radius + tolerance)) / 2
    z_offset = tolerance + sphere_radius
    step = 2 * sphere_radius + tolerance
    for i in range(n_spheres):
        # Get z, y, and x
        z = i // spheres_per_layer
        relative_i = i % spheres_per_layer
        y = relative_i // spheres_per_row
        x = relative_i % spheres_per_row
        sphere_pos[:, i, :] = torch.tensor([offset + step * x, offset + step * y, z_offset + step * z])

    # Transform the relative pos to the container's pos and orientation
    # Calculate necessary rotation
    init_ori_vec = torch.zeros_like(container_ori_vec)
    init_ori_vec[:, 2] = 1.0                        # initial orientation points upwards
    aa_rotation = torch.stack([axisangle_between_vec(vec1=init_ori_vec, vec2=container_ori_vec)] * n_spheres, dim=1)
    quat_rotation = axisangle2quat(aa_rotation)
    # Update sphere pos based on pitcher pose
    sphere_pos = rotate_vec_by_quat(sphere_pos, quat_rotation) + container_pos.unsqueeze(dim=1)

    # Return calculated sphere positions
    return sphere_pos
