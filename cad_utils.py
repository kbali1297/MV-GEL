import os, numpy as np
import sys
import gc
import pyvista as pv
import cv2
import math
# Uncomment below if using env other than LISA_multi_view
# freecad_base = "/data/1bali/miniforge3/envs/vtk_offscreen"
# # Append FreeCAD's Python library paths
# sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
# sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)
# Now you can import FreeCAD normally.
# Resolve FreeCAD's lib dir from the active environment (override with FREECAD_LIB).
freecad_lib = os.environ.get("FREECAD_LIB", os.path.join(sys.prefix, "lib"))
sys.path.append(freecad_lib)
import FreeCAD
import Part
import Import
import Mesh
import MeshPart
import trimesh
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def extract_el_az_from_view_desc(view_desc):
    """
    Extracts elevation and azimuth from a view description string.
    Example view description: ".../view_e30_a-60.png"
    """
    el,az = view_desc.split('_e')[1].split('_')[0], view_desc.split('_a')[1].split('.')[0].split('_')[0]
    return int(el), int(az)

def tessellate_entity_with_offset(
    shape,
    entity,
    entity_type,
    offset_dist,
    tessellation=0.05,
    samples=80,
):
    """
    Tessellate and offset a CAD entity (face or edge) along outward normals.

    Args:
        shape: FreeCAD shape
        entity: Part.Face or Part.Edge
        entity_type: 'face' or 'edge'
        offset_dist: absolute offset distance
        tessellation: face tessellation
        samples: edge discretization samples

    Returns:
        pv.PolyData or None
    """

    bbox = shape.BoundBox
    model_center = np.array([bbox.Center.x, bbox.Center.y, bbox.Center.z])

    # ==========================
    # FACE
    # ==========================
    if entity_type == "face":
        face = entity
        surface = face.Surface

        mesh = MeshPart.meshFromShape(
            Shape=face,
            LinearDeflection=tessellation,
            AngularDeflection=0.523,
            Relative=False,
        )

        raw_verts = [FreeCAD.Vector(v.x, v.y, v.z) for v in mesh.Topology[0]]
        displaced_verts = []

        for v in raw_verts:
            try:
                u, v_param = surface.parameter(v)
                n = surface.normal(u, v_param)

                if face.Orientation == "Reversed":
                    n.multiply(-1.0)

                n.normalize()
                dv = v.add(n.multiply(offset_dist))
                displaced_verts.append([dv.x, dv.y, dv.z])

            except Exception:
                displaced_verts.append([v.x, v.y, v.z])

        faces = []
        for tri in mesh.Topology[1]:
            if len(tri) == 3:
                faces.append([3, tri[0], tri[1], tri[2]])

        if len(faces) == 0:
            return None

        return pv.PolyData(
            np.array(displaced_verts),
            np.array(faces).flatten(),
        )

    # ==========================
    # EDGE
    # ==========================
    elif entity_type == "edge":
        edge = entity

        u0, u1 = edge.ParameterRange
        params = np.linspace(u0, u1, samples)

        pts = []
        for u in params:
            try:
                p = edge.valueAt(u)
                pts.append(np.array([p.x, p.y, p.z]))
            except Exception:
                continue

        if len(pts) < 2:
            return None

        pts = np.array(pts)

        faces = shape.ancestorsOfType(edge, Part.Face)
        displaced_pts = []

        for p in pts:
            best_normal = None
            best_dot = -np.inf

            for face in faces[:2]:
                try:
                    surface = face.Surface
                    u_face, v_face = surface.parameter(
                        FreeCAD.Vector(p[0], p[1], p[2])
                    )

                    n = surface.normal(u_face, v_face)

                    if face.Orientation == "Reversed":
                        n.multiply(-1.0)

                    n_vec = np.array([n.x, n.y, n.z])
                    n_vec /= np.linalg.norm(n_vec)

                    dot = np.dot(n_vec, p - model_center)
                    if dot > best_dot:
                        best_dot = dot
                        best_normal = n_vec

                except Exception:
                    continue

            if best_normal is None:
                fallback = p - model_center
                best_normal = fallback / np.linalg.norm(fallback)

            displaced_pts.append(p + offset_dist * best_normal)

        displaced_pts = np.array(displaced_pts)

        poly = pv.PolyData(displaced_pts)
        poly.lines = np.hstack(
            [[len(displaced_pts)], np.arange(len(displaced_pts))]
        )
        return poly

    else:
        raise ValueError(f"Unknown entity_type: {entity_type}")

def farthest_point_sampling(points, k, start_idx=None):
    """
    Farthest Point Sampling (FPS)

    Args:
        points: (N, D) numpy array of points
        k: number of points to sample
        start_idx: optional int, starting point index

    Returns:
        sampled_points: (k, D) numpy array
        sampled_indices: (k,) numpy array of indices
    """
    N, D = points.shape
    if k > N: k=N

    sampled_indices = np.zeros(k, dtype=np.int64)
    distances = np.full(N, np.inf)

    if start_idx is None:
        start_idx = np.random.randint(N)

    sampled_indices[0] = start_idx
    current_point = points[start_idx]

    for i in range(1, k):
        dist = np.linalg.norm(points - current_point, axis=1)
        distances = np.minimum(distances, dist)
        sampled_indices[i] = np.argmax(distances)
        current_point = points[sampled_indices[i]]

    return points[sampled_indices], sampled_indices

def highlight_topk_edges_offset(
    shape,
    k=5,
    samples=80,
    offset_ratio=0.001,
):
    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    edges_sorted = sorted(
        enumerate(shape.Edges),
        key=lambda x: x[1].Length,
        reverse=True
    )
    top_edges = edges_sorted[:k]

    edge_polys, edge_idxs = [], []

    for edge_idx, edge in top_edges:
        poly = tessellate_entity_with_offset(
            shape,
            edge,
            entity_type="edge",
            offset_dist=offset_dist,
            samples=samples,
        )
        if poly is not None:
            edge_polys.append(poly)
            edge_idxs.append(edge_idx)

    return edge_polys, edge_idxs


def highlight_cad_edge(
    shape,
    edge_idxs,
    samples=80,
    offset_ratio=0.001,
):
    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    edge_polys = []

    for edge_idx in edge_idxs:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Edges[edge_idx],
            entity_type="edge",
            offset_dist=offset_dist,
            samples=samples,
        )
        if poly is not None:
            edge_polys.append(poly)

    return edge_polys


def highlight_topk_cad_faces_offset(
    shape,
    k=5,
    tessellation=0.05,
    offset_ratio=0.003,
):
    """
    Highlight top-k largest CAD faces.
    """

    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    face_areas = [(i, f.Area) for i, f in enumerate(shape.Faces)]
    face_areas.sort(key=lambda x: x[1], reverse=True)
    top_ids = [i for i, _ in face_areas[:k]]

    face_polys = []

    for fid in top_ids:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Faces[fid],
            "face",
            offset_dist,
            tessellation,
        )
        if poly is not None:
            face_polys.append(poly)

    return face_polys, top_ids

def highlight_cad_face(
    shape,
    face_idxs,
    tessellation=0.05,
    offset_ratio=0.003,
):
    """
    Highlight specific CAD faces.
    """

    bbox = shape.BoundBox
    diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
    offset_dist = diag * offset_ratio

    face_polys = []

    for fid in face_idxs:
        poly = tessellate_entity_with_offset(
            shape,
            shape.Faces[fid],
            "face",
            offset_dist,
            tessellation,
        )
        if poly is not None:
            face_polys.append(poly)

    return face_polys

def extract_cad_edges(shape, step_file=None):
    if step_file is not None: # Override the part input and read part from the step file
        shape = Part.read(step_file)

    cad_edges = []

    for edge in shape.Edges:
        faces = shape.ancestorsOfType(edge, Part.Face)

        # true CAD edge = exactly two faces meet
        if len(faces) == 2:
            cad_edges.append(edge)

    return cad_edges

def discretize_edge(edge, n=50):
    params = np.linspace(edge.FirstParameter, edge.LastParameter, n)
    return np.array([edge.valueAt(u) for u in params], dtype=float)

def cad_edges_to_polydata(edges, samples=50):
    polylines = []

    for edge in edges:
        pts = discretize_edge(edge, samples)

        line = pv.PolyData(pts)
        cells = np.hstack([[len(pts)], np.arange(len(pts))])
        line.lines = cells

        polylines.append(line)

    return polylines

def freecad_shape_to_pyvista(shape, linear_deflection=0.1, angular_deflection=0.523):
    """
    Convert a FreeCAD Part.Shape to PyVista PolyData (faces only).
    """
    mesh = MeshPart.meshFromShape(
        Shape=shape,
        LinearDeflection=linear_deflection,
        AngularDeflection=angular_deflection,
        Relative=False
    )

    # mesh.Topology[0] -> list[Base.Vector]
    verts = np.array([[v.x, v.y, v.z] for v in mesh.Topology[0]])

    faces = []
    for f in mesh.Topology[1]:
        if len(f) == 3:
            faces.append([3, f[0], f[1], f[2]])

    faces = np.array(faces).flatten()
    return pv.PolyData(verts, faces)


def freecad_edges_to_pyvista(shape, edge_samples=80):
    """
    Robust extraction of CAD edges using FreeCAD's discretize().
    Works for all curve types (lines, arcs, splines, trimmed curves).
    """
    polylines = []

    for edge in shape.Edges:
        try:
            pts_fc = edge.discretize(Num=edge_samples)
        except Exception:
            # Fallback: skip pathological edges
            continue

        if len(pts_fc) < 2:
            continue

        pts = np.array([[p.x, p.y, p.z] for p in pts_fc])

        lines = np.hstack([[len(pts)], np.arange(len(pts))])
        poly = pv.PolyData(pts)
        poly.lines = lines
        polylines.append(poly)

    return polylines


def render_cad_views(
    cad_file,
    output_dir=None,
    n_azimuth=12,
    n_elevation=3,
    orthographic=False,
    points_3d=None,
    add_axes=True,
    verbose=True,
    axes_size="normal",
    tessellation=0.1,
    highlight_edge_idxs=None,
    highlight_face_idxs=None,
    use_cad=True,
    save_img=True,
    retain_compound_shape=False
):
    """
    Fast multi-view CAD renderer using FreeCAD + PyVista.

    Optimizations:
    - Single Plotter instance
    - Static geometry added once
    - Camera-only updates per view
    - screenshot() instead of show()
    """

    if output_dir is None:
        output_dir = os.path.dirname(cad_file)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load CAD
    # --------------------------------------------------
    if use_cad:
        doc = FreeCAD.newDocument()
        Import.insert(cad_file, doc.Name)
        doc.recompute()

        shapes = [
            obj.Shape
            for obj in doc.Objects
            if hasattr(obj, "Shape") and not obj.Shape.isNull()
        ]

        if len(shapes) == 0:
            raise ValueError(f"No valid shapes found in {cad_file}")

        if retain_compound_shape:
            shape = Part.makeCompound(shapes)
        else:
            shape = shapes[0]

        # --------------------------------------------------
        # Convert to PyVista
        # --------------------------------------------------
        cad_mesh = freecad_shape_to_pyvista(
            shape, linear_deflection=tessellation
        )

        cad_edges = extract_cad_edges(shape)
        cad_polylines = cad_edges_to_polydata(cad_edges)

        FreeCAD.closeDocument(doc.Name)
    else:
        cad_mesh = pv.read(cad_file.replace(".step", ".obj") if cad_file.endswith(".step") else cad_file)
        if cad_mesh.n_points == 0:
            raise ValueError(f"Mesh file {cad_mesh} is empty or invalid")

        # --- Strong edge overlay ---
        # step_file_path = mesh_file.replace('.obj', '.step').replace('.stl', '.step')
        # cad_edges = extract_cad_edges(step_file_path)
        # cad_polylines = cad_edges_to_polydata(cad_edges)

        mesh_clean = (
            cad_mesh
            .clean(tolerance=1e-6)
            .merge_points()
            .compute_normals(auto_orient_normals=True, split_vertices=False)
        )
        cad_polylines = [
            mesh_clean.extract_feature_edges(
                boundary_edges=True,
                feature_edges=True,
                manifold_edges=False,
                non_manifold_edges=False,
                feature_angle=60.0
            )]
        cad_polylines = []
        

    if highlight_edge_idxs is not None:
        assert use_cad, "CAD must be loaded to highlight edges"
        chosen_edges = highlight_cad_edge(
            shape, edge_idxs=highlight_edge_idxs
        )

    if highlight_face_idxs is not None:
        assert use_cad, "CAD must be loaded to highlight faces"
        chosen_faces = highlight_cad_face(
            shape, face_idxs=highlight_face_idxs
        )

    # --------------------------------------------------
    # Camera setup
    # --------------------------------------------------
    bounds = cad_mesh.bounds
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
    )

    diag = np.linalg.norm(
        [
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ]
    )

    zoom = 1.2 if orthographic else 4.0
    edge_mark_factor = 0.003 if orthographic else 0.001
    radius = diag / 2 * zoom

    elevations = (
        n_elevation
        if isinstance(n_elevation, list)
        else [
            -90 + (180 / (n_elevation + 1)) * e
            for e in range(1, n_elevation + 1)
        ]
    )

    azimuths = (
        n_azimuth
        if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )

    # --------------------------------------------------
    # Plotter (ONE instance)
    # --------------------------------------------------
    pv.start_xvfb()

    plotter = pv.Plotter(
        off_screen=True, window_size=(1024, 1024)
    )
    plotter.set_background("white")

    if orthographic:
        plotter.enable_parallel_projection()

    # --------------------------------------------------
    # Static geometry (added ONCE)
    # --------------------------------------------------
    plotter.add_mesh(
        cad_mesh,
        color="#cccccc",
        opacity=1.0,
        lighting=True,
        specular=0.2,
        ambient=0.3,
        diffuse=0.9,
    )

    # Precompute edge tubes once
    edge_tubes = [
        line.tube(
            radius=edge_mark_factor * radius,
            n_sides=6,
        )
        for line in cad_polylines
    ]

    for tube in edge_tubes:
        plotter.add_mesh(
            tube, color="black", lighting=False
        )

    if highlight_edge_idxs is not None:
        for edge in chosen_edges:
            plotter.add_mesh(
                edge.tube(
                    radius=edge_mark_factor * 3 * radius,
                    n_sides=8,
                ),
                color="red",
                lighting=False,
            )

    if highlight_face_idxs is not None:
        for face in chosen_faces:
            plotter.add_mesh(
                face,
                color="blue",
                opacity=1.0,
                smooth_shading=True,
            )

    if points_3d is not None:
        plotter.add_points(
            pv.PolyData(points_3d),
            color="red",
            point_size=15,
            render_points_as_spheres=True,
        )

    if add_axes:
        plotter.add_axes(
            interactive=False,
            line_width=5,
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(0.0, 0.0, 0.32, 0.35),
        )

    # --------------------------------------------------
    # Camera sweep + screenshots
    # --------------------------------------------------
    # Add a dictionary to store images in memory
    rendered_images = {}
    world_up = (0.0, 0.0, 1.0)

    for elevation in elevations:
        for azimuth in azimuths:

            cam_x = center[0] + radius * np.cos(
                np.radians(-elevation)
            ) * np.sin(np.radians(azimuth))
            cam_y = center[1] - radius * np.cos(
                np.radians(-elevation)
            ) * np.cos(np.radians(azimuth))
            cam_z = center[2] - radius * np.sin(
                np.radians(-elevation)
            )

            plotter.camera_position = [
                (cam_x, cam_y, cam_z),
                center,
                world_up,
            ]

            if orthographic:
                plotter.camera.parallel_scale = radius * 0.8

            if highlight_edge_idxs is not None and highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}&face{highlight_face_idxs}.png"
                )
            elif highlight_edge_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}.png"
                )
            elif highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_face{highlight_face_idxs}.png"
                )
            else:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}.png"
                )

            plotter.render()

            filename = os.path.join(output_dir, image_name)
            if save_img:
                img_array = plotter.screenshot(filename, return_img=True)
                if verbose:
                    print(f"✅ Saved {filename}")
            else:
                img_array = plotter.screenshot(None, return_img=True)
            
            rendered_images[filename] = img_array
            
    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    # VTK holds render buffers/meshes at the C level that plotter.close() alone
    # does not always free. close_all() + an explicit gc pass keeps host RAM flat
    # across the thousands of render calls in a long evaluation run (prevents the
    # slow climb that previously triggered the Linux OOM killer).
    plotter.deep_clean()
    plotter.close()
    pv.close_all()
    del plotter
    gc.collect()

    if verbose and save_img:
        print(
            f"✅ Finished rendering CAD views → {output_dir}"
        )
    
    return rendered_images

def normalize_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    center = vertices.mean(axis=0)
    vertices -= center
    scale = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= scale
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def create_camera_positions(center, n_elevation, n_azimuth, radius):
    elevations = (
        n_elevation
        if isinstance(n_elevation, list)
        else [-90 + (180 / (n_elevation + 1)) * e for e in range(1, n_elevation + 1)]
    )

    azimuths = (
        n_azimuth
        if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )

    poses = []
    for elevation in elevations:
        for azimuth in azimuths:

            cam_x = center[0] -radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
            cam_y = center[1] -radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
            cam_z = center[2] -radius * np.sin(np.radians(-elevation))

            cam_pos = np.array([cam_x, cam_y, cam_z])
            poses.append((elevation, azimuth, cam_pos))

    return poses


def render_depth_maps(
    mesh_path,
    output_dir,
    image_size=(1024, 1024),
    n_elevation=5,
    n_azimuth=8,
    fov=30.0,
    zoom=4.0,
):
    """
    Precompute depth maps for a mesh and save them.

    Saves:
        depth_e{e}_a{a}.npy
    """

    H, W = image_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + normalize mesh ----
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = normalize_mesh(mesh)

    # Compute radius from bounding box
    bounds = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bounds.get_extent())
    radius = diag / 2 * zoom

    # ---- Create renderer ONCE ----
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"

    renderer.scene.add_geometry("mesh", mesh, material)

    poses = create_camera_positions(n_elevation, n_azimuth, radius)

    #print(f"Rendering {len(poses)} depth maps...")

    for elevation, azimuth, cam_pos in poses:

        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])

        renderer.setup_camera(
            fov,
            center,
            cam_pos,
            up,
        )

        depth = renderer.render_to_depth_image()
        depth = np.asarray(depth).astype(np.float32)

        # Optional: replace inf with 0
        depth[np.isinf(depth)] = 0

        # ---- Save ----
        filename = output_dir / f"depth_e{int(elevation)}_a{int(azimuth)}.npy"
        np.save(filename, depth)

        # # ---- Render RGB ----
        # color = renderer.render_to_image()
        # color = np.asarray(color)  # uint8 HxWx3

        # # ---- Save RGB ----
        # rgb_filename = output_dir / f"rgb_e{int(elevation)}_a{int(azimuth)}.png"
        # o3d.io.write_image(str(rgb_filename), o3d.geometry.Image(color))

        # # For visualization only (not training)
        # depth_vis = depth.copy()
        # mask = depth_vis > 0
        # depth_vis[mask] = (depth_vis[mask] - depth_vis[mask].min()) / (
        #     depth_vis[mask].max() - depth_vis[mask].min()
        # )

        # depth_vis = (depth_vis * 255).astype(np.uint8)

        # depth_vis_filename = output_dir / f"depth_vis_e{int(elevation)}_a{int(azimuth)}.png"
        # o3d.io.write_image(
        #     str(depth_vis_filename),
        #     o3d.geometry.Image(depth_vis)
        # )

    # Clean up once
    renderer.scene.clear_geometry()
    del renderer

    print(f"Saved all at {output_dir}")

def render_depth_normal_maps(
    mesh_path,
    output_dir,
    image_size=(1024, 1024),
    n_elevation=5,
    n_azimuth=8,
    fov=30.0,
    zoom=4.0,
    use_compression=False,
):
    """
    Saves per view:
        DN_e{e}_a{a}.npz

    Each file contains:
        depth  -> (H, W) float16
        normal -> (H, W, 3) uint8  in [0,255]
    """

    import open3d as o3d
    import numpy as np
    from pathlib import Path

    H, W = image_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + normalize mesh ----
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = normalize_mesh(mesh)
    
    aabb = mesh.get_axis_aligned_bounding_box()

    min_bound = aabb.get_min_bound()  # [xmin, ymin, zmin]
    max_bound = aabb.get_max_bound()  # [xmax, ymax, zmax]

    # ---- center exactly like your CAD version ----
    center = np.array([
        (min_bound[0] + max_bound[0]) / 2,
        (min_bound[1] + max_bound[1]) / 2,
        (min_bound[2] + max_bound[2]) / 2,
    ])

    # ---- diagonal exactly like your CAD version ----
    diag = np.linalg.norm([
        max_bound[0] - min_bound[0],
        max_bound[1] - min_bound[1],
        max_bound[2] - min_bound[2],
    ])

    radius = diag / 2 * zoom

    # ---- Renderer ----
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

    material_depth = o3d.visualization.rendering.MaterialRecord()
    material_depth.shader = "defaultUnlit"

    material_normal = o3d.visualization.rendering.MaterialRecord()
    material_normal.shader = "normals"

    renderer.scene.add_geometry("mesh_depth", mesh, material_depth)
    renderer.scene.add_geometry("mesh_normal", mesh, material_normal)

    poses = create_camera_positions(center, n_elevation, n_azimuth, radius)

    for elevation, azimuth, cam_pos in poses:

        up = np.array([0.0, 0.0, 1.0])
        renderer.setup_camera(fov, center, cam_pos, up)

        # ==========================
        # DEPTH
        # ==========================
        renderer.scene.show_geometry("mesh_depth", True)
        renderer.scene.show_geometry("mesh_normal", False)

        depth = renderer.render_to_depth_image()
        depth = np.asarray(depth)
        depth[np.isinf(depth)] = 0
        depth = depth.astype(np.float16)  # <-- KEY CHANGE

        # ==========================
        # NORMAL
        # ==========================
        renderer.scene.show_geometry("mesh_depth", False)
        renderer.scene.show_geometry("mesh_normal", True)

        normal_img = renderer.render_to_image()
        normal_img = np.asarray(normal_img)

        # normal_img is already uint8 in [0,255]
        normal_img = normal_img.astype(np.uint8)

        # ==========================
        # SAVE
        # ==========================
        filename = output_dir / f"DN_e{int(elevation)}_a{int(azimuth)}.npz"

        if use_compression:
            np.savez_compressed(
                filename,
                depth=depth,
                normal=normal_img,
            )
        else:
            np.savez(
                filename,
                depth=depth,
                normal=normal_img,
            )

    renderer.scene.clear_geometry()
    del renderer

    print(f"Saved all maps at {output_dir}")

def render_sobel_edge_maps(
    mesh_path,
    output_dir,
    image_size=(1024, 1024),
    n_elevation=5,
    n_azimuth=8,
    fov=30.0,
    zoom=4.0,
):
    """
    Saves per view:
        edge_depth_e{e}_a{a}.png
        edge_normal_e{e}_a{a}.png
    """

    import open3d as o3d
    import numpy as np
    import cv2
    from pathlib import Path

    H, W = image_size
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load + normalize mesh ----
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()
    mesh = normalize_mesh(mesh)

    bounds = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bounds.get_extent())
    radius = diag / 2 * zoom

    # ---- Renderer ----
    renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)

    material_depth = o3d.visualization.rendering.MaterialRecord()
    material_depth.shader = "defaultUnlit"

    material_normal = o3d.visualization.rendering.MaterialRecord()
    material_normal.shader = "normals"

    renderer.scene.add_geometry("mesh_depth", mesh, material_depth)
    renderer.scene.add_geometry("mesh_normal", mesh, material_normal)

    poses = create_camera_positions(n_elevation, n_azimuth, radius)

    for elevation, azimuth, cam_pos in poses:

        center = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        renderer.setup_camera(fov, center, cam_pos, up)

        # ==========================
        # DEPTH → Sobel
        # ==========================
        renderer.scene.show_geometry("mesh_depth", True)
        renderer.scene.show_geometry("mesh_normal", False)

        depth = renderer.render_to_depth_image()
        depth = np.asarray(depth).astype(np.float32)
        depth[np.isinf(depth)] = 0

        # dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        # dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        # edge_depth = np.sqrt(dx**2 + dy**2)

        # # Normalize for visualization
        # edge_depth = edge_depth / (edge_depth.max() + 1e-8)
        # edge_depth = (edge_depth * 255).astype(np.uint8)

        # ---- Log depth (VERY IMPORTANT) ----
        depth = np.log(depth + 1e-6)

        # ---- Sobel ----
        dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge_depth = np.sqrt(dx**2 + dy**2)

        # ---- Percentile normalization (prevents silhouette domination) ----
        p = np.percentile(edge_depth, 99)
        edge_depth = np.clip(edge_depth / (p + 1e-8), 0, 1)

        edge_depth = (edge_depth * 255).astype(np.uint8)

        cv2.imwrite(
            str(output_dir / f"edge_depth_e{int(elevation)}_a{int(azimuth)}.png"),
            edge_depth,
        )

        # ==========================
        # NORMAL → Sobel
        # ==========================
        renderer.scene.show_geometry("mesh_depth", False)
        renderer.scene.show_geometry("mesh_normal", True)

        normal_img = renderer.render_to_image()
        normal_img = np.asarray(normal_img).astype(np.float32)

        normal_img = normal_img / 255.0
        normal_img = normal_img * 2.0 - 1.0

        grad_mag = np.zeros((H, W), dtype=np.float32)

        for c in range(3):
            dx = cv2.Sobel(normal_img[..., c], cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(normal_img[..., c], cv2.CV_32F, 0, 1, ksize=3)
            grad_mag += dx**2 + dy**2

        edge_normal = np.sqrt(grad_mag)
        edge_normal = edge_normal / (edge_normal.max() + 1e-8)
        edge_normal = (edge_normal * 255).astype(np.uint8)

        cv2.imwrite(
            str(output_dir / f"edge_normal_e{int(elevation)}_a{int(azimuth)}.png"),
            edge_normal,
        )

    renderer.scene.clear_geometry()
    del renderer

    print(f"Sobel edge images saved at {output_dir}")

def viewNMS(view_list, scores, angle_threshold=30):
    """
    Non-Maximum Suppression for CAD views based on camera angle similarity.

    Args:
        view_list: list of view metadata (elevation, azimuth)
        scores: list of view scores
        angle_threshold: minimum angle difference (in degrees) to consider views as distinct
    Returns:
        selected_views: list of selected view metadata
    """
    selected_views = []

    views_sorted = sorted(zip(view_list, scores), key=lambda x: x[1], reverse=True)
    for view, _ in views_sorted:

        el,az = extract_el_az_from_view_desc(view)
        el,az = float(el), float(az)

        keep = True
        v1 = np.array([
            np.cos(np.radians(el)) * np.cos(np.radians(az)),
            np.cos(np.radians(el)) * np.sin(np.radians(az)),
            np.sin(np.radians(el)) 
        ])
        if selected_views == []:
            selected_views.append(view)
            continue
        
        for sel_view in selected_views:
            sel_el, sel_az = extract_el_az_from_view_desc(sel_view)
            sel_el, sel_az = float(sel_el), float(sel_az)

            v2 = np.array([
                np.cos(np.radians(sel_el)) * np.cos(np.radians(sel_az)),
                np.cos(np.radians(sel_el)) * np.sin(np.radians(sel_az)),
                np.sin(np.radians(sel_el)) 
            ])
            dot_product = np.dot(v1, v2)
            angle_diff = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

            if angle_diff < angle_threshold:
                keep = False
                break

        if keep:
            selected_views.append(view)

    return selected_views

import numpy as np

def view_diversified_topK(view_list, scores, K, lambda_div=0.5):
    """
    Diversity-aware top-K view selection.

    Args:
        view_list: list of view descriptors
        scores: list of scalar scores (same order)
        K: number of views to select
        lambda_div: weight for angular diversity term

    Returns:
        selected_views: list of K selected view descriptors
    """

    assert len(view_list) == len(scores)
    assert K <= len(view_list)

    # Convert all views to unit direction vectors once
    view_dirs = []
    for view in view_list:
        el, az = extract_el_az_from_view_desc(view)
        el, az = float(el), float(az)

        v = np.array([
            np.cos(np.radians(el)) * np.cos(np.radians(az)),
            np.cos(np.radians(el)) * np.sin(np.radians(az)),
            np.sin(np.radians(el))
        ])
        v = v / np.linalg.norm(v)
        view_dirs.append(v)

    view_dirs = np.stack(view_dirs)

    # Convert scores to numpy
    scores = np.array(scores)

    # Step 1: select highest scoring view first
    selected_indices = [np.argmax(scores)]

    # Step 2: greedy selection
    while len(selected_indices) < K:

        best_value = -np.inf
        best_idx = None

        for i in range(len(view_list)):
            if i in selected_indices:
                continue

            # Compute minimum angular distance to already selected views
            divs = []
            for sel_idx in selected_indices:
                dot = np.clip(np.dot(view_dirs[i], view_dirs[sel_idx]), -1.0, 1.0)
                # angle = np.degrees(np.arccos(dot))
                divs.append(1-dot)

            min_div = min(divs)

            # Diversity-aware objective
            value = scores[i] + lambda_div * min_div

            if value > best_value:
                best_value = value
                best_idx = i

        selected_indices.append(best_idx)

    selected_views = [view_list[i] for i in selected_indices]
    return selected_views

def convert_to_obj(input_file, output_file, consider_first_part_idx=None, linear_deflection=0.1, angular_deflection=0.5):
    ext = os.path.splitext(input_file)[1].lower()

    doc = FreeCAD.newDocument("Doc")

    try:
        if ext in [".step", ".stp", ".iges", ".igs"]:
            Part.open(input_file)  # directly loads B-rep shapes
        elif ext in [".fcstd"]:
            FreeCAD.open(input_file)
        else:
            Mesh.open(input_file)  # STL, OBJ, etc.
    except Exception as e:
        print(f"❌ Failed to open {input_file}: {e}")
        return

    meshes = []
    for obj_idx, obj in enumerate(FreeCAD.ActiveDocument.Objects):
        if consider_first_part_idx is not None:
            if consider_first_part_idx==obj_idx:
                try:
                    if obj.TypeId == "Mesh::Feature":
                        meshes.append(obj.Mesh)
                    elif hasattr(obj, "Shape") and not obj.Shape.isNull():
                        mesh = MeshPart.meshFromShape(
                            Shape=obj.Shape,
                            LinearDeflection=linear_deflection,
                            AngularDeflection=angular_deflection,
                            Relative=False
                        )
                        meshes.append(mesh)
                except Exception as e:
                    print(f"⚠️ Skipping {obj.Label}: {e}")
                break
        else: #plot all objects
            try:
                if obj.TypeId == "Mesh::Feature":
                    meshes.append(obj.Mesh)
                elif hasattr(obj, "Shape") and not obj.Shape.isNull():
                    mesh = MeshPart.meshFromShape(
                        Shape=obj.Shape,
                        LinearDeflection=linear_deflection,
                        AngularDeflection=angular_deflection,
                        Relative=False
                    )
                    meshes.append(mesh)
            except Exception as e:
                print(f"⚠️ Skipping {obj.Label}: {e}")
        
    if meshes:
        merged = Mesh.Mesh()  # ✅ new container
        for m in meshes:
            merged.addMesh(m.copy())  # ✅ avoid immutability
        merged.write(output_file)
        print(f"✅ Exported {input_file} → {output_file}")
    else:
        print(f"⚠️ No meshable objects found in {input_file}")

    return output_file

def compute_optimal_views(cad_file,
    n_azimuth=12,
    n_elevation=9,
    num_top_edges=1,
    num_top_faces=1,
    orthographic=True):

    doc = FreeCAD.newDocument()
    Import.insert(cad_file, doc.Name)
    doc.recompute()

    shapes = []

    for obj in doc.Objects:
        if hasattr(obj, "Shape") and not obj.Shape.isNull():
            shapes.append(obj.Shape)

    if len(shapes) == 0:
        raise ValueError(f"No valid shapes found in {cad_file}")


    # Create a compound (non-boolean, preserves topology)
    shape = shapes[0] #Part.makeCompound(shapes)

    cad_edges, cad_edge_idxs = highlight_topk_edges_offset(shape=shape, k=num_top_edges)
    cad_faces, cad_face_idxs = highlight_topk_cad_faces_offset(shape=shape, k=num_top_faces)

    
    mesh_file = convert_to_obj(cad_file, output_file=cad_file.replace(".step", ".obj"), consider_first_part_idx=0)

    #mesh = pv.read(mesh_file)
    mesh = trimesh.load(mesh_file)

    elevations = (
        n_elevation if isinstance(n_elevation, list)
        else [-90 + (180 / (n_elevation + 1)) * e for e in range(1, n_elevation + 1)]
    )

    azimuths = (
        n_azimuth if isinstance(n_azimuth, list)
        else [(360 / n_azimuth) * a for a in range(n_azimuth)]
    )
    
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    if orthographic: 
        zoom = 1.2
    else: 
        zoom = 4.0
    

    radius = np.linalg.norm(bounds[0,:] - bounds[1,:])/2 * zoom
    pts_edge_list, face_centroids_fps_list = [], []
    for cad_edge in cad_edges:
        cad_edge_pts_fps, _ = farthest_point_sampling(cad_edge.points, k=30)
        # pts.extend(cad_edge_pts_fps)
        ## For multiple edges seperate lists of edge_pts
        pts_edge_list.append(cad_edge_pts_fps)

    for cad_face in cad_faces:
        mesh_faces = cad_face.faces.reshape(-1, 4)[:, 1:]
        pts_faces = cad_face.points

        face_centroids = pts_faces[mesh_faces].mean(axis=1)
        face_centroids_fps, _ = farthest_point_sampling(points=face_centroids, k=30)
        face_centroids_fps_list.append(face_centroids_fps)

    # ---------------------------
    # Views inspection loop for each cad edge and cad face, which views maximize visibility 
    # ---------------------------
    
    edge_viewpaths_list, face_viewpaths_list = [],[]
    for pts, face_centroids_fps in zip(pts_edge_list, face_centroids_fps_list):
        edge_occlude, face_occlude = {}, {}
        for elevation in elevations:
            for azimuth in azimuths:

                cam_x = center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
                cam_y = center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
                cam_z = center[2] - radius * np.sin(np.radians(-elevation))

                cam_pos = np.array([cam_x, cam_y, cam_z])

                ## Check if occluded, currently mechanism works for orthographic views
                # Create a ray to the points
                if orthographic:
                    z_cam = center - cam_pos
                    ray_dir = z_cam / np.linalg.norm(z_cam)
                #ray_dir = z_cam
                
                key = f'view_e{int(elevation)}_a{int(azimuth)}'
                edge_occlude[key] = 0
                face_occlude[key] = 0

                for pt in pts:
                    if not orthographic:
                        ray_dir = (pt - cam_pos) / np.linalg.norm(pt - cam_pos)
                    ray_origin = np.array(pt) - ray_dir * 10000.0
                    locations, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_dir])
                    )

                    try:
                        if np.linalg.norm(locations[0] - ray_origin) < np.linalg.norm(pt - ray_origin):
                            edge_occlude[key] += 1
                    except: edge_occlude[key] += 2

                for centroid in face_centroids_fps:
                    if not orthographic:
                        ray_dir = (centroid - cam_pos) / np.linalg.norm(centroid - cam_pos)
                    ray_origin = centroid - ray_dir * 10000.0
                    locations, index_ray, index_tri = mesh.ray.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_dir])
                    )

                    try:
                        if np.linalg.norm(locations[0] - ray_origin) < np.linalg.norm(centroid - ray_origin):
                            face_occlude[key] += 1
                    except: face_occlude[key] += 2


        ## Sorting views according to occluded edge and face scores
        edge_occlude_ = [(key, value) for key, value in edge_occlude.items()]
        face_occlude_ = [(key, value) for key, value in face_occlude.items()]

        sorted_views_edge = sorted(edge_occlude_, key= lambda x: x[1])
        sorted_views_face = sorted(face_occlude_, key= lambda x: x[1])
        
        cad_dirpath = os.path.dirname(cad_file)
        edge_viewpaths = [f'{cad_dirpath}/{edge_view[0]}_marked_edge.png' for edge_view in sorted_views_edge] 
        face_viewpaths = [f'{cad_dirpath}/{face_view[0]}_marked_face.png' for face_view in sorted_views_face]

        edge_viewpaths_list.append(edge_viewpaths)
        face_viewpaths_list.append(face_viewpaths)
    
    
    return edge_viewpaths_list, face_viewpaths_list, cad_edge_idxs, cad_face_idxs


def pixel_to_mesh(cam_pos,
    F_pos,
    u, v,
    up_cam_vec,
    img_H, img_W,
    fov_in_degrees=75,
    parallel_scale=None,   # REQUIRED if orthographic=True
    mesh=None,
    orthographic=False,
    return_all_hits=False,
    return_no_location=False):
    """
    Convert pixel (u, v) to 3D world coordinate on mesh or view plane.
    Supports both perspective and orthographic projections.
    """
    # -------------------------------
    # Camera coordinate system
    # -------------------------------
    z_cam = F_pos - cam_pos
    z_cam = z_cam / np.linalg.norm(z_cam)

    x_cam = np.cross(up_cam_vec, -z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)

    y_cam = np.cross(-z_cam, x_cam)

    cx = img_W / 2.0
    cy = img_H / 2.0

    plane_depth = np.linalg.norm(F_pos - cam_pos)
    # -------------------------------
    # PERSPECTIVE PROJECTION
    # -------------------------------
    if not orthographic:
        plane_depth = np.linalg.norm(F_pos - cam_pos)

        fov_y_rad = math.radians(fov_in_degrees)
        fy = 0.5 * img_H / math.tan(0.5 * fov_y_rad)
        fx = fy * img_W / img_H

        x_plane = (u - cx) / fx * plane_depth
        y_plane = -(v - cy) / fy * plane_depth

        ray_origin = cam_pos + x_plane * x_cam + y_plane * y_cam
        ray_dir = z_cam

    # -------------------------------
    # ORTHOGRAPHIC PROJECTION
    # -------------------------------
    else:
        if parallel_scale is None:
            raise ValueError("parallel_scale must be provided for orthographic projection")

        world_height = 2.0 * parallel_scale
        world_width = world_height * img_W / img_H

        sx = world_width / img_W
        sy = world_height / img_H

        dx = (u - cx) * sx
        dy = -(v - cy) * sy

        # Projection plane through focal point
        P_plane = F_pos + dx * x_cam + dy * y_cam

        ray_origin = P_plane + 1000000.0 * z_cam
        ray_dir = -z_cam

    # -------------------------------
    # Ray–mesh intersection
    # -------------------------------
    if mesh is None:
        return ray_origin + plane_depth * z_cam

    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=np.array([ray_origin]),
        ray_directions=np.array([ray_dir])
    )

    if len(locations) == 0:
        if return_no_location:
            return None
        return np.array([ray_origin + plane_depth * z_cam])

    # sort hits by distance to camera
    dists = np.linalg.norm(locations - cam_pos, axis=1)
    order = np.argsort(dists)
    locations = locations[order]

    if return_all_hits:
        if len(locations.shape)>1:
            return locations
        return locations[None,:]
    else:
        return locations[0]   # closest hit

def build_mesh_edges(faces):
    """
    Deterministic unique edge list.
    Order is stable across calls.
    """
    import numpy as np

    # Collect all edges
    e0 = faces[:, [0, 1]]
    e1 = faces[:, [1, 2]]
    e2 = faces[:, [2, 0]]

    all_edges = np.vstack([e0, e1, e2])

    # Sort each edge (v_min, v_max)
    all_edges = np.sort(all_edges, axis=1)

    # Unique with stable ordering
    edges = np.unique(all_edges, axis=0)

    return edges
    
def mark_spots_in_image(img_path, spot_radius=10, spot_color=(0, 0, 255), spot_positions=[], output_path=None):
    """
    Marks spots on the image at specified positions.

    Parameters:
    - img_path: str, path to the input image.
    - spot_radius: int, radius of the spots to be drawn.
    - spot_color: tuple, BGR color of the spots.
    - spot_positions: list of tuples, each tuple contains (x, y) coordinates for a spot.

    Returns:
    - output_img_path: str, path to the output image with spots marked.
    """
    # Read the image
    img = cv2.imread(img_path)

    # Draw spots on the image
    for pos in spot_positions:
        try:
            if pos.shape[0] !=2: pos = pos.squeeze(0)
        except: pass
        x, y = pos
        x,y = int(round(x)), int(round(y))
        cv2.circle(img, (x,y), spot_radius, spot_color, -1)  # -1 fills the circle
        
    # Save the output image
    if output_path is not None:
        output_img_path = output_path
    elif not img_path.endswith('_with_spots.png'):
        output_img_path = img_path.replace('.png', '_with_spots.png').replace('.jpg', '_with_spots.jpg')
    else:
        output_img_path = img_path
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, img)

    return output_img_path

import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

import os
import numpy as np
import pyvista as pv

def render_cad_views_gt_mesh(
    cad_file,
    output_dir=None,
    n_azimuth=12,
    n_elevation=3,
    orthographic=False,
    points_3d=None,
    add_axes=True,
    verbose=True,
    axes_size="normal",
    tessellation=0.1,
    highlight_edge_idxs=None,
    highlight_face_idxs=None,
    use_cad=True,
    save_img=True,
    retain_compound_shape=False,
    use_prediction_style=False  # <--- NEW TOGGLE ADDED HERE
):
    """
    Fast multi-view CAD renderer using FreeCAD + PyVista.

    Optimizations:
    - Single Plotter instance
    - Static geometry added once
    - Camera-only updates per view
    - screenshot() instead of show()
    """

    if output_dir is None:
        output_dir = os.path.dirname(cad_file)

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Load CAD
    # --------------------------------------------------
    if use_cad:
        import FreeCAD
        import Part
        import Import
        # Make sure freecad_shape_to_pyvista, extract_cad_edges, cad_edges_to_polydata 
        # highlight_cad_edge, highlight_cad_face are imported/defined in your file!
        
        doc = FreeCAD.newDocument()
        Import.insert(cad_file, doc.Name)
        doc.recompute()

        shapes =[
            obj.Shape
            for obj in doc.Objects
            if hasattr(obj, "Shape") and not obj.Shape.isNull()
        ]

        if len(shapes) == 0:
            raise ValueError(f"No valid shapes found in {cad_file}")

        if retain_compound_shape:
            shape = Part.makeCompound(shapes)
        else:
            shape = shapes[0]

        # --------------------------------------------------
        # Convert to PyVista
        # --------------------------------------------------
        cad_mesh = freecad_shape_to_pyvista(
            shape, linear_deflection=tessellation
        )

        cad_edges = extract_cad_edges(shape)
        cad_polylines = cad_edges_to_polydata(cad_edges)

        FreeCAD.closeDocument(doc.Name)
    else:
        cad_mesh = pv.read(cad_file.replace(".step", ".obj") if cad_file.endswith(".step") else cad_file)
        if cad_mesh.n_points == 0:
            raise ValueError(f"Mesh file {cad_mesh} is empty or invalid")

        mesh_clean = (
            cad_mesh
            .clean(tolerance=1e-6)
            .merge_points()
            .compute_normals(auto_orient_normals=True, split_vertices=False)
        )
        cad_polylines =[]

    if highlight_edge_idxs is not None:
        assert use_cad, "CAD must be loaded to highlight edges"
        chosen_edges = highlight_cad_edge(
            shape, edge_idxs=highlight_edge_idxs
        )

    if highlight_face_idxs is not None:
        assert use_cad, "CAD must be loaded to highlight faces"
        chosen_faces = highlight_cad_face(
            shape, face_idxs=highlight_face_idxs
        )

    # --------------------------------------------------
    # Camera setup
    # --------------------------------------------------
    bounds = cad_mesh.bounds
    center = np.array(
        [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2,
        ]
    )

    diag = np.linalg.norm([
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        ]
    )

    zoom = 1.2 if orthographic else 4.0
    edge_mark_factor = 0.003 if orthographic else 0.001
    radius = diag / 2 * zoom

    elevations = (
        n_elevation
        if isinstance(n_elevation, list)
        else[
            -90 + (180 / (n_elevation + 1)) * e
            for e in range(1, n_elevation + 1)
        ]
    )

    azimuths = (
        n_azimuth
        if isinstance(n_azimuth, list)
        else[(360 / n_azimuth) * a for a in range(n_azimuth)]
    )

    # --------------------------------------------------
    # Plotter (ONE instance)
    # --------------------------------------------------
    pv.start_xvfb()

    plotter = pv.Plotter(
        off_screen=True, window_size=(1024, 1024)
    )
    plotter.set_background("white")

    if orthographic:
        plotter.enable_parallel_projection()

    # --------------------------------------------------
    # Static geometry (added ONCE)
    # --------------------------------------------------
    
    # 1. ADD BASE MESH WITH CHOSEN STYLE
    if use_prediction_style:
        if highlight_face_idxs is not None:
            # Match visualize_entity_predictions "face" mode (solid, with edges)
            plotter.add_mesh(
                cad_mesh,
                color="#D2D2D2",  # Equivalent to RGB 210, 210, 210
                lighting=True,
                show_edges=True,
                edge_color="black",
                line_width=0.3,
                specular=0.2,
                ambient=0.3,
                diffuse=0.9,
            )
        else:
            # Match visualize_entity_predictions "edge" mode (translucent + wireframe)
            plotter.add_mesh(
                cad_mesh,
                color="#cccccc",
                opacity=0.4,
                lighting=True,
                specular=0.2,
                ambient=0.3,
                diffuse=0.9,
            )
            plotter.add_mesh(
                cad_mesh,
                style='wireframe',
                color='black',
                line_width=0.2,
                opacity=0.15,
                lighting=False,
            )
    else:
        # Original default base mesh
        plotter.add_mesh(
            cad_mesh,
            color="#cccccc",
            opacity=1.0,
            lighting=True,
            specular=0.2,
            ambient=0.3,
            diffuse=0.9,
        )

    # 2. ADD STANDARD CAD EDGES (Turned off in prediction style to prevent clutter)
    if not use_prediction_style:
        edge_tubes =[
            line.tube(radius=edge_mark_factor * radius, n_sides=6)
            for line in cad_polylines
        ]
        for tube in edge_tubes:
            plotter.add_mesh(tube, color="black", lighting=False)

    # 3. HIGHLIGHT EDGES
    if highlight_edge_idxs is not None:
        for edge in chosen_edges:
            plotter.add_mesh(
                edge.tube(
                    radius=edge_mark_factor * 3 * radius,
                    n_sides=8,
                ),
                color="red",
                lighting=True if use_prediction_style else False, # Prediction edges have lighting
            )

    # 4. HIGHLIGHT FACES
    if highlight_face_idxs is not None:
        for face in chosen_faces:
            if use_prediction_style:
                plotter.add_mesh(
                    face,
                    color="blue",
                    lighting=True,
                    show_edges=True,
                    edge_color="black",
                    line_width=0.3,
                    specular=0.2,
                    ambient=0.3,
                    diffuse=0.9,
                )
            else:
                plotter.add_mesh(
                    face,
                    color="blue",
                    opacity=1.0,
                    smooth_shading=True,
                )

    if points_3d is not None:
        plotter.add_points(
            pv.PolyData(points_3d),
            color="red",
            point_size=15,
            render_points_as_spheres=True,
        )

    if add_axes:
        plotter.add_axes(
            interactive=False,
            line_width=5,
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(0.0, 0.0, 0.32, 0.35),
        )

    # --------------------------------------------------
    # Camera sweep + screenshots
    # --------------------------------------------------
    rendered_images = {}
    world_up = (0.0, 0.0, 1.0)

    for elevation in elevations:
        for azimuth in azimuths:

            cam_x = center[0] - radius * np.cos(
                np.radians(-elevation)
            ) * np.sin(np.radians(azimuth))
            cam_y = center[1] - radius * np.cos(
                np.radians(-elevation)
            ) * np.cos(np.radians(azimuth))
            cam_z = center[2] - radius * np.sin(
                np.radians(-elevation)
            )

            plotter.camera_position =[
                (cam_x, cam_y, cam_z),
                center,
                world_up,
            ]

            if orthographic:
                plotter.camera.parallel_scale = radius * 0.8

            if highlight_edge_idxs is not None and highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}&face{highlight_face_idxs}.png"
                )
            elif highlight_edge_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_edge{highlight_edge_idxs}.png"
                )
            elif highlight_face_idxs is not None:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}"
                    f"_marked_face{highlight_face_idxs}.png"
                )
            else:
                image_name = (
                    f"view_e{elevation:.0f}_a{azimuth:.0f}.png"
                )

            plotter.render()

            filename = os.path.join(output_dir, image_name)
            if save_img:
                img_array = plotter.screenshot(filename, return_img=True)
                if verbose:
                    print(f"✅ Saved {filename}")
            else:
                img_array = plotter.screenshot(None, return_img=True)
            
            rendered_images[filename] = img_array
            
    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    plotter.deep_clean()
    plotter.close()
    pv.close_all()
    del plotter
    gc.collect()

    if verbose and save_img:
        print(
            f"✅ Finished rendering CAD views → {output_dir}"
        )
    
    return rendered_images

# def inspect_masks_(pred_mask, gt_mask, image_path, save_dir=None, marked_image_path=None, prompt=None, append_str=""):
    
#     save_dir = "debug_masks_cad_GLoc3" if save_dir is None else save_dir
    
#     os.makedirs(save_dir, exist_ok=True)

#     pred_mask = (pred_mask * 255).cpu().numpy()
#     pred_uint8 = pred_mask.astype(np.uint8)

#     cad_name = os.path.dirname(image_path).split('/')[-2]
#     image_id = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
#     target_img_id = os.path.basename(marked_image_path).replace('.png', '').replace('.jpg', '') if marked_image_path is not None else None
#     # Image.fromarray(pred_uint8).save(
#     #     os.path.join(save_dir, f"{image_id}_pred.png")
#     # )

#     gt_mask = (gt_mask * 255).cpu().numpy()
#     gt_uint8 = gt_mask.astype(np.uint8)

#     # Image.fromarray(gt_uint8).save(
#     #     os.path.join(save_dir, f"{image_id}_gt.png")
#     # )

#     import matplotlib.pyplot as plt

#     img = Image.open(image_path).convert("RGB")
#     target_img = Image.open(marked_image_path).convert("RGB") if marked_image_path is not None else None

#     # 2️⃣ Convert to tensor (C,H,W)
#     transform = transforms.ToTensor()
#     image = transform(img)
#     target_img = transform(target_img) if target_img is not None else None

#     # 3️⃣ Add batch dimension → (1,C,H,W)
#     image = image.unsqueeze(0)
#     target_img = target_img.unsqueeze(0) if target_img is not None else None

#     image_np = image[0].permute(1,2,0).cpu().numpy()
#     target_img_np = target_img[0].permute(1,2,0).cpu().numpy() if target_img is not None else None

#     # plt.figure(figsize=(6,6))
#     # plt.imshow(image_np)
#     # plt.imshow(pred_mask, alpha=0.5)
#     # plt.axis("off")

#     # # plt.savefig(os.path.join(save_dir, f"{image_id}_pred_overlay.png"),
#     # #             bbox_inches="tight",
#     # #             pad_inches=0)
#     # plt.close()

#     fig, ax = plt.subplots(1, 4, figsize=(15, 5))

#     ax[0].imshow(image_np)
#     ax[0].set_title(f"Chosen View {image_id}")

#     ax[1].imshow(gt_mask, cmap="gray")
#     ax[1].set_title("Ground Truth on chosen view")

#     ax[2].imshow(pred_mask, cmap="gray")
#     ax[2].set_title("Prediction on chosen view")

#     ax[3].imshow(target_img_np)
#     ax[3].set_title(f"Target: {target_img_id}")

#     for a in ax:
#         a.axis("off")

#     import textwrap

#     if prompt is not None:
#         wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=90))
#         fig.suptitle(wrapped_prompt, fontsize=11)
#         plt.tight_layout(rect=[0, 0, 1, 0.90])  # reserve space for title
#     else:
#         plt.tight_layout()

#     plt.savefig(
#         os.path.join(save_dir, f"{cad_name}_{image_id}{append_str}.png"),
#         bbox_inches="tight"
#     )
#     plt.close()

#     intersection = (pred_mask * gt_mask).sum()
#     union = ((pred_mask + gt_mask) > 0).sum()

#     iou = intersection / (union + 1e-6)

#     print(f"{save_dir}/{cad_name}_{image_id}{append_str}.png IoU: {iou:.4f}")

#     print(prompt) if prompt is not None else print("")



if __name__ == '__main__':
    cad_file_path = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset/abc_0000_step_v00/00000001/00000001_1ffb81a71e5b402e966b9341_step_000.step'
    render_save_path = '/'.join(cad_file_path.split('/')[:-1])
    render_cad_views(cad_file=cad_file_path, output_dir=render_save_path, orthographic=True)


# def highlight_topk_cad_faces_offset(
#     shape,
#     k=5,
#     tessellation=0.05,
#     offset_ratio=0.003,
# ):
#     """
#     Highlight top-k largest CAD faces and push them outward along
#     their local normals so they are always visible (prevent z-fighting).
#     """

#     # --- Compute model scale for relative offset ---
#     bbox = shape.BoundBox
#     diag = np.linalg.norm([bbox.XLength, bbox.YLength, bbox.ZLength])
#     offset_dist = diag * offset_ratio

#     # --- Sort faces by area ---
#     face_areas = [(i, f.Area) for i, f in enumerate(shape.Faces)]
#     face_areas.sort(key=lambda x: x[1], reverse=True)
#     top_ids = [i for i, _ in face_areas[:k]]

#     all_verts = []
#     all_faces = []
#     vert_offset = 0

#     for fid in top_ids:
#         face = shape.Faces[fid]
#         surface = face.Surface

#         # --- Tessellate the face ---
#         # Note: Depending on your FreeCAD version, Standard might be preferred over meshFromShape
#         # for better control, but sticking to your current logic:
#         mesh = MeshPart.meshFromShape(
#             Shape=face,
#             LinearDeflection=tessellation,
#             AngularDeflection=0.523,
#             Relative=False,
#         )

#         # Extract raw vertices
#         raw_verts = [FreeCAD.Vector(v.x, v.y, v.z) for v in mesh.Topology[0]]
#         new_verts = []

#         # --- Per-Vertex Normal Calculation ---
#         for v in raw_verts:
#             try:
#                 # 1. Get UV coordinates on the surface for this 3D point
#                 # This projects the mesh vertex onto the mathematical surface
#                 u, v_param = surface.parameter(v)
                
#                 # 2. Compute the normal at these UV coordinates
#                 n = surface.normal(u, v_param)
                
#                 # 3. Handle orientation (Face normal vs Surface normal)
#                 # FreeCAD Faces can be reversed relative to their geometric Surface
#                 if face.Orientation == 'Reversed':
#                     n.multiply(-1.0)
                
#                 # 4. Normalize (just to be safe, though .normal() usually is)
#                 n.normalize()

#                 # 5. Displace vertex
#                 displaced_v = v.add(n.multiply(offset_dist))
#                 new_verts.append([displaced_v.x, displaced_v.y, displaced_v.z])

#             except Exception as e:
#                 # Fallback if projection fails: keep original position
#                 new_verts.append([v.x, v.y, v.z])

#         # Convert to numpy
#         verts_np = np.array(new_verts)

#         # --- Collect Faces ---
#         for tri in mesh.Topology[1]:
#             # Filter for triangles (FreeCAD mesh usually guarantees this, but safety first)
#             if len(tri) == 3:
#                 all_faces.append([
#                     3,
#                     tri[0] + vert_offset,
#                     tri[1] + vert_offset,
#                     tri[2] + vert_offset,
#                 ])

#         all_verts.append(verts_np)
#         vert_offset += verts_np.shape[0]

#     # Combine all into one PolyData object
#     if not all_verts:
#         return pv.PolyData(), top_ids

#     poly = pv.PolyData(np.vstack(all_verts), np.array(all_faces).flatten())
#     return poly, top_ids



# def visualize_cad_entity(
#     step_file,
#     feature,            # "face" or "edge"
#     entity_index,       # index of the entity to highlight
#     save_prefix="cad_vis",
#     elev_azim_list=[(30, 45), (30, 135), (60, 45), (0, 0)]
# ):
#     """
#     Visualize a CAD entity from a STEP file.

#     feature = "face" or "edge"
#     entity_index = index of that face/edge in shape.Faces or shape.Edges

#     Colors:
#         Highlighted entity : Yellow
#         Background geometry: Light gray
#     """

#     import FreeCAD
#     import Part
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

#     # ---------------------------------------------------------
#     # Load STEP
#     # ---------------------------------------------------------
#     doc = FreeCAD.newDocument()
#     Part.insert(step_file, doc.Name)

#     shape_objs = [obj for obj in doc.Objects if hasattr(obj, "Shape")]
#     if not shape_objs:
#         raise RuntimeError("No valid shape found in STEP file")

#     obj = shape_objs[0]

#     # Apply placement correctly
#     shape = obj.Shape.copy()
#     shape = shape.transformGeometry(obj.Placement.toMatrix())

#     # ---------------------------------------------------------
#     # Tessellate entire shape for visualization
#     # ---------------------------------------------------------
#     tess_tol = 0.5  # visual resolution only
#     vertices = []
#     faces = []

#     for face in shape.Faces:
#         tri = face.tessellate(tess_tol)
#         v = np.array(tri[0])
#         f = np.array(tri[1])

#         offset = len(vertices)
#         vertices.extend(v)
#         faces.extend(f + offset)

#     vertices = np.array(vertices)
#     faces = np.array(faces)

#     # ---------------------------------------------------------
#     # Extract entity geometry
#     # ---------------------------------------------------------
#     highlight_triangles = []
#     highlight_edges = []

#     if feature == "face":
#         if entity_index >= len(shape.Faces):
#             raise IndexError("Face index out of range")

#         target_face = shape.Faces[entity_index]
#         tri = target_face.tessellate(tess_tol)

#         hv = np.array(tri[0])
#         hf = np.array(tri[1])

#         highlight_triangles = hv[hf]

#     elif feature == "edge":
#         if entity_index >= len(shape.Edges):
#             raise IndexError("Edge index out of range")

#         target_edge = shape.Edges[entity_index]

#         # Discretize edge
#         pts = target_edge.discretize(50)
#         highlight_edges = np.array(pts)

#     else:
#         raise ValueError("feature must be 'face' or 'edge'")

#     # ---------------------------------------------------------
#     # Render views
#     # ---------------------------------------------------------
#     for i, (elev, azim) in enumerate(elev_azim_list):

#         fig = plt.figure(figsize=(6, 6))
#         ax = fig.add_subplot(111, projection="3d")

#         # Draw full CAD shape (light gray)
#         mesh_poly = Poly3DCollection(
#             vertices[faces],
#             facecolors=[0.85, 0.85, 0.85, 1.0],
#             linewidths=0.1,
#             edgecolors="k",
#             alpha=0.5 if feature == "edge" else 1.0
#         )
#         ax.add_collection3d(mesh_poly)

#         # Highlight face
#         if feature == "face":
#             poly = Poly3DCollection(
#                 highlight_triangles,
#                 facecolors=[1.0, 1.0, 0.0, 1.0],  # yellow
#                 edgecolors="k",
#                 linewidths=0.5
#             )
#             ax.add_collection3d(poly)

#         # Highlight edge
#         if feature == "edge":
#             lc = Line3DCollection(
#                 [highlight_edges],
#                 colors=[[1.0, 1.0, 0.0, 1.0]],
#                 linewidths=3.0
#             )
#             ax.add_collection3d(lc)

#         # Scaling
#         scale = vertices.flatten()
#         ax.auto_scale_xyz(scale, scale, scale)

#         ax.view_init(elev=elev, azim=azim)
#         ax.set_axis_off()

#         plt.tight_layout()
#         plt.savefig(f"{save_prefix}_{feature}_{entity_index}_view{i}.png", dpi=300)
#         plt.close()

#     FreeCAD.closeDocument(doc.Name)

#     print(f"Saved CAD visualization for {feature} {entity_index}")