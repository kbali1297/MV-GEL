import os, numpy as np
import sys
import pyvista as pv
import cv2
import math
# Uncomment below if using env other than LISA_multi_view
# freecad_base = "/data/1bali/miniforge3/envs/vtk_offscreen"
# # Append FreeCAD's Python library paths
# sys.path.append(os.path.join(freecad_base, "lib"))   # core FreeCAD libraries
# sys.path.append(os.path.join(freecad_base, "Mod"))   # FreeCAD Python modules (Part, Mesh, etc.)
# Now you can import FreeCAD normally
freecad_lib = os.path.join("/data/1bali/miniforge3/envs/LISA_multi_view", "lib")
if freecad_lib not in sys.path:
    sys.path.append(freecad_lib)
# FreeCAD/Part/Import/Mesh/MeshPart are imported lazily inside the functions
# that need them (e.g. cad_entity_to_mesh_faces). Importing them at module
# load makes this file unusable from envs without a compatible libstdc++/Qt6.
try:
    import FreeCAD  # noqa: F401
    import Part  # noqa: F401
    import Import  # noqa: F401
    import Mesh  # noqa: F401
    import MeshPart  # noqa: F401
except ImportError as _freecad_err:
    print(f"[eval_utils] FreeCAD not importable in this env: {_freecad_err}. "
          "Functions needing FreeCAD will fail when called.")
import trimesh
import open3d as o3d
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def stitch_mesh_topology(vertices, faces, tol=1e-6):
    import numpy as np

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # Round vertices to tolerance
    rounded = np.round(vertices / tol) * tol

    # Find unique vertices
    unique_vertices, inverse = np.unique(
        rounded, axis=0, return_inverse=True
    )

    # Reindex faces
    new_faces = inverse[faces]

    return unique_vertices, new_faces

#----------------------eval-------------------------------------------------------------
import numpy as np
try:
    import Mesh  # noqa: F401  (FreeCAD's Mesh module; only needed by some funcs)
except ImportError:
    Mesh = None


# ---------------------------------------------------------
# Core Geometry Utilities
# ---------------------------------------------------------

def triangle_area(v0, v1, v2):
    v0_np = np.array([v0.x, v0.y, v0.z])
    v1_np = np.array([v1.x, v1.y, v1.z])
    v2_np = np.array([v2.x, v2.y, v2.z])
    return 0.5 * np.linalg.norm(np.cross(v1_np - v0_np, v2_np - v0_np))


def get_mesh_data(mesh_file):
    mesh = Mesh.Mesh(mesh_file)
    vertices = np.array(mesh.Points)
    faces = np.array([facet.PointIndices for facet in mesh.Facets])
    return vertices, faces


def compute_triangle_areas(vertices, faces):
    areas = []
    for tri in faces:
        v0, v1, v2 = vertices[tri]
        areas.append(triangle_area(v0, v1, v2))
    return np.array(areas)


# ---------------------------------------------------------
# 1. Surface IoU + Precision + Recall + F1
# ---------------------------------------------------------

import numpy as np

def compute_triangle_areas(vertices, faces):
    """
    Vectorized and MUCH faster version of triangle area computation.
    Replaces the old loop-based function.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Vectorized cross product and norm
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    
    return areas

from PIL import Image

def generate_masked_view(pred_mask, image_path, save_dir):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torchvision.transforms as transforms

    # 1. Ensure pred_mask is 2D (H, W)
    # If pred_mask is (1, H, W) or (H, W, 1), .squeeze() will make it (H, W)
    # If it is (H, W, H), that is likely an error in your data loading.
    pred_mask = pred_mask.squeeze()
    if pred_mask.ndim > 2:
        # If it's still 3D (e.g., from a multi-channel output), take the first channel
        pred_mask = pred_mask[0]
        
    # Convert prediction mask to float and ensure range 0-1
    pred_mask = pred_mask.cpu().numpy()
    if pred_mask.max() > 1.0:
        pred_mask = pred_mask / 255.0
        
    # Load and process image
    img = Image.open(image_path).convert("RGB")
    transform = transforms.ToTensor()
    image = transform(img) # Shape: (3, H, W)
    
    # Convert image to (H, W, 3)
    image_np = image.permute(1, 2, 0).cpu().numpy()
    
    # ----- Create translucent overlay -----
    overlay = image_np.copy()
    
    # Ensure pred_bin is 2D (H, W)
    pred_bin = (pred_mask > 0.5).astype(np.float32)

    alpha = 0.4

    # Apply overlay only where mask is 1
    # overlay channels: 0=R, 1=G, 2=B
    # We only modify pixels where pred_bin is 1
    
    # Red channel: reduce intensity
    overlay[..., 0] = np.where(pred_bin == 1, overlay[..., 0] * (1 - alpha), overlay[..., 0])
    # Green channel: add green
    overlay[..., 1] = np.where(pred_bin == 1, overlay[..., 1] * (1 - alpha) + alpha * 1.0, overlay[..., 1])
    # Blue channel: reduce intensity
    overlay[..., 2] = np.where(pred_bin == 1, overlay[..., 2] * (1 - alpha), overlay[..., 2])

    overlay = np.clip(overlay, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    #ax.set_title("Overlay (Pred=Green)")
    ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)

    return save_path


def inspect_masks(pred_mask, gt_mask, image_path, save_dir=None, marked_image_path=None, prompt=None, append_str="", dpi=300, remove_ground_truth=False):
    
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import textwrap
    import torchvision.transforms as transforms

    save_dir = "debug_masks_cad_GLoc3" if save_dir is None else save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Convert prediction mask
    pred_mask = (pred_mask * 255).cpu().numpy()
    pred_uint8 = pred_mask.astype(np.uint8)

    # Convert GT mask
    gt_mask = (gt_mask * 255).cpu().numpy()
    gt_uint8 = gt_mask.astype(np.uint8)

    cad_name = os.path.dirname(image_path).split('/')[-2]
    image_id = os.path.basename(image_path).replace('.png', '').replace('.jpg', '')
    target_img_id = os.path.basename(marked_image_path).replace('.png', '').replace('.jpg', '') if marked_image_path is not None else None

    # Load images
    img = Image.open(image_path).convert("RGB")
    target_img = Image.open(marked_image_path).convert("RGB") if marked_image_path is not None else None

    transform = transforms.ToTensor()

    image = transform(img).unsqueeze(0)
    target_img = transform(target_img).unsqueeze(0) if target_img is not None else None

    image_np = image[0].permute(1, 2, 0).cpu().numpy()
    target_img_np = target_img[0].permute(1, 2, 0).cpu().numpy() if target_img is not None else None

    # ----- Create translucent overlay -----
    overlay = image_np.copy()

    gt_bin = (gt_mask > 127).astype(np.float32)
    pred_bin = (pred_mask > 127).astype(np.float32)

    alpha = 0.4

    # Red overlay for GT
    if not remove_ground_truth:
        overlay[..., 0] = overlay[..., 0] * (1 - alpha * gt_bin) + alpha * gt_bin * 1.0
        overlay[..., 1] = overlay[..., 1] * (1 - alpha * gt_bin)
        overlay[..., 2] = overlay[..., 2] * (1 - alpha * gt_bin)

    # Green overlay for Prediction
    overlay[..., 0] = overlay[..., 0] * (1 - alpha * pred_bin)
    overlay[..., 1] = overlay[..., 1] * (1 - alpha * pred_bin) + alpha * pred_bin * 1.0
    overlay[..., 2] = overlay[..., 2] * (1 - alpha * pred_bin)

    overlay = np.clip(overlay, 0, 1)

    # ----- Plot -----
    fig, ax = plt.subplots(1, 5, figsize=(30, 8))

    ax[0].imshow(image_np)
    ax[0].set_title(f"Chosen View {image_id}")

    ax[1].imshow(gt_mask, cmap="gray")
    ax[1].set_title("Ground Truth on chosen view")

    ax[2].imshow(pred_mask, cmap="gray")
    ax[2].set_title("Prediction on chosen view")

    ax[3].imshow(overlay)
    if remove_ground_truth:
        ax[3].set_title("Overlay (Pred=Green)")
    else:
        ax[3].set_title("Overlay (GT=Red, Pred=Green)")

    if target_img_np is not None:
        ax[4].imshow(target_img_np)
        ax[4].set_title(f"Target: {target_img_id}")
    else:
        ax[4].axis("off")

    for a in ax:
        a.axis("off")

    if prompt is not None:
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=90))
        fig.suptitle(wrapped_prompt, fontsize=11)
        plt.tight_layout(rect=[0, 0, 1, 0.90])
    else:
        plt.tight_layout()

    save_path = os.path.join(save_dir, f"{cad_name}_{image_id}{append_str}.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close()

    # ----- IoU -----
    intersection = ((pred_mask > 127) & (gt_mask > 127)).sum()
    union = ((pred_mask > 127) | (gt_mask > 127)).sum()
    iou = intersection / (union + 1e-6)

    print(f"{save_path} IoU: {iou:.4f}")
    print(prompt if prompt is not None else "")

def return_localization_metrics(
    mesh_file,
    vertices,
    faces,
    pred_entities,
    gt_entities,
    feature,        # "face" or "edge"
    feature_idx
):
    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # ---------------------------------------------------------
    # FACE METRICS (area weighted)
    # ---------------------------------------------------------
    if feature == "face":

        areas = compute_triangle_areas(vertices, faces)
        n_faces = len(faces)

        pred_mask = np.zeros(n_faces, dtype=bool)
        gt_mask = np.zeros(n_faces, dtype=bool)

        # Safely mark valid indices to avoid IndexError
        valid_pred =[int(p) for p in pred_entities if 0 <= int(p) < n_faces]
        valid_gt =[int(g) for g in gt_entities if 0 <= int(g) < n_faces]

        pred_mask[valid_pred] = True
        gt_mask[valid_gt] = True

        intersection_mask = pred_mask & gt_mask
        union_mask = pred_mask | gt_mask

        area_intersection = areas[intersection_mask].sum()
        area_pred = areas[pred_mask].sum()
        area_gt = areas[gt_mask].sum()
        area_union = areas[union_mask].sum()

        iou = area_intersection / area_union if area_union > 0 else 0.0
        precision = area_intersection / area_pred if area_pred > 0 else 0.0
        recall = area_intersection / area_gt if area_gt > 0 else 0.0

    # ---------------------------------------------------------
    # EDGE METRICS (length weighted)
    # ---------------------------------------------------------
    elif feature == "edge":

        # Build unique mesh edges
        edge_set = set()
        for f in faces:
            edge_set.add(tuple(sorted((f[0], f[1]))))
            edge_set.add(tuple(sorted((f[1], f[2]))))
            edge_set.add(tuple(sorted((f[2], f[0]))))

        edges = np.array(list(edge_set), dtype=np.int64)
        n_edges = len(edges)

        # Map edge tuples to integer indices for masking
        edge_to_idx = {tuple(e): i for i, e in enumerate(edges)}

        # Compute edge lengths
        v0 = vertices[edges[:, 0]]
        v1 = vertices[edges[:, 1]]
        edge_lengths = np.linalg.norm(v1 - v0, axis=1)

        pred_mask = np.zeros(n_edges, dtype=bool)
        gt_mask = np.zeros(n_edges, dtype=bool)

        # Safely map tuple pairs to boolean mask
        for e in pred_entities:
            e_sorted = tuple(sorted(e))
            if e_sorted in edge_to_idx:
                pred_mask[edge_to_idx[e_sorted]] = True
                
        for e in gt_entities:
            e_sorted = tuple(sorted(e))
            if e_sorted in edge_to_idx:
                gt_mask[edge_to_idx[e_sorted]] = True

        intersection_mask = pred_mask & gt_mask
        union_mask = pred_mask | gt_mask

        length_intersection = edge_lengths[intersection_mask].sum()
        length_pred = edge_lengths[pred_mask].sum()
        length_gt = edge_lengths[gt_mask].sum()
        length_union = edge_lengths[union_mask].sum()

        iou = length_intersection / length_union if length_union > 0 else 0.0
        precision = length_intersection / length_pred if length_pred > 0 else 0.0
        recall = length_intersection / length_gt if length_gt > 0 else 0.0

    else:
        raise ValueError("feature must be 'face' or 'edge'")

    # ---------------------------------------------------------
    # F1 (shared)
    # ---------------------------------------------------------
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "mesh_file": mesh_file,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "feature": feature,
        "feature_idx": feature_idx
    }


def visualize_entity_predictions(
    vertices,
    faces,
    pred_entities,
    target_entities,
    feature,
    save_prefix="mesh_vis",
    add_axes=True,
    elev_azim_list=[(36, 60), (36, 300), (-36, 60), (-36, 300)],
    orthographic=False,
    filter_normals=False,       # NEW: Toggle for normal filtering
    angle_threshold=15.0        # NEW: Threshold in degrees
):
    """
    feature = "face" or "edge"
    filter_normals: If True, prunes face entities that are not parallel to the target geometry
    """
    import numpy as np
    import pyvista as pv
    import os

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    # --------------------------------------------------
    # Create Mesh and Compute Normals
    # --------------------------------------------------
    n_faces = len(faces)
    face_padding = np.full((n_faces, 1), 3, dtype=faces.dtype)
    pv_faces = np.hstack([face_padding, faces]).flatten()
    mesh = pv.PolyData(vertices, pv_faces)
    
    # Compute normals for filtering
    mesh = mesh.compute_normals(cell_normals=True, point_normals=False)
    cell_normals = mesh.cell_data["Normals"]

    # =========================================================
    # LOGIC (Correct, Pred-only, Target-only)
    # =========================================================
    if feature == "face":
        # 1. Standard Logic
        pred_set = set(pred_entities)
        target_set = set(target_entities)

        # 2. Add "Islands" Filtering (Prune noise)
        if filter_normals: # Reusing your toggle
            import networkx as nx
            
            def prune_islands(indices, mesh):
                if not indices: return set()
                # Build adjacency for these specific faces
                submesh = mesh.extract_cells(list(indices))
                # Get adjacency (faces that share edges)
                edges = submesh.extract_feature_edges(manifold_edges=False, boundary_edges=False)
                # This is a bit complex, simplified: just get connected components via PyVista
                # Actually, simpler: only keep indices that are part of the largest cluster
                clusters = submesh.connectivity(largest=False) 
                # This returns a mesh with a 'RegionId' scalar
                region_ids = clusters.cell_data['RegionId']
                counts = np.bincount(region_ids)
                # Keep only faces in the largest cluster
                largest_cluster_id = np.argmax(counts)
                return {idx for i, idx in enumerate(indices) if region_ids[i] == largest_cluster_id}

            pred_set = prune_islands(pred_set, mesh)
            # We usually keep target_set as is, or prune it too if it's messy

        correct = pred_set & target_set
        pred_only = pred_set - target_set
        target_only = target_set - pred_set

        face_colors = np.ones((len(faces), 3), dtype=np.uint8) * 210
        for idx in target_only: face_colors[idx] = [0, 255, 0]
        for idx in pred_only: face_colors[idx] = [255, 0, 0]
        for idx in correct: face_colors[idx] = [255, 255, 0]

        mesh.cell_data["Colors"] = face_colors

    elif feature == "edge":
        pred_set = {tuple(sorted(e)) for e in pred_entities}
        target_set = {tuple(sorted(e)) for e in target_entities}
        correct = pred_set & target_set
        pred_only = pred_set - target_set
        target_only = target_set - pred_set

    else:
        raise ValueError("feature must be 'face' or 'edge'")

    # =========================================================
    # CAMERA AND SCENE MATH
    # =========================================================
    bounds = mesh.bounds
    center = np.array([(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2])
    diag = np.linalg.norm([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]])
    radius = (diag / 2) * (1.2 if orthographic else 4.0)
    edge_mark_factor = 0.003 if orthographic else 0.001
    world_up = (0.0, 0.0, 1.0)

    # =========================================================
    # INITIALIZE PLOTTER
    # =========================================================
    pv.start_xvfb() if hasattr(pv, "start_xvfb") else None
    plotter = pv.Plotter(off_screen=True, window_size=(1024, 1024))
    plotter.set_background("white")
    if orthographic: plotter.enable_parallel_projection()

    if feature == "face":
        plotter.add_mesh(mesh, scalars="Colors", rgb=True, lighting=True, 
                         show_edges=True, edge_color="black", line_width=0.3, 
                         specular=0.2, ambient=0.3, diffuse=0.9)
    else:
        plotter.add_mesh(mesh, color="#cccccc", opacity=0.4, lighting=True, 
                         specular=0.2, ambient=0.3, diffuse=0.9)
        plotter.add_mesh(mesh, style='wireframe', color='black', line_width=0.2, 
                         opacity=0.15, lighting=False)

        def add_edge_tubes(edge_set, color):
            if not edge_set: return
            edges_array = np.array(list(edge_set))
            n_edges = len(edges_array)
            pv_lines = np.hstack([np.full((n_edges, 1), 2, dtype=edges_array.dtype), edges_array]).flatten()
            line_mesh = pv.PolyData(vertices, lines=pv_lines)
            plotter.add_mesh(line_mesh.tube(radius=edge_mark_factor * 3 * radius, n_sides=8), 
                             color=color, lighting=True)

        add_edge_tubes(target_only, "green")
        add_edge_tubes(pred_only, "red")
        add_edge_tubes(correct, "yellow")

    if add_axes:
        plotter.add_axes(
            interactive=False,
            line_width=5,
            x_color="red",
            y_color="green",
            z_color="blue",
            viewport=(0.0, 0.0, 0.32, 0.35),
        )

    save_paths = []
    for i, (elevation, azimuth) in enumerate(elev_azim_list):
        cam_x = center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
        cam_y = center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
        cam_z = center[2] - radius * np.sin(np.radians(-elevation))
        plotter.camera_position = [(cam_x, cam_y, cam_z), center, world_up]
        if orthographic: plotter.camera.parallel_scale = radius * 0.8
        plotter.render()
        save_path = f"{save_prefix}_{feature}_view_e{elevation}_a{azimuth}.png"
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plotter.screenshot(save_path)
        save_paths.append(save_path)

    plotter.close()
    print(f"Saved rendered PyVista views (filter_normals={filter_normals}).")
    return save_paths

def return_mesh_entities_from_2D_masks(
    vertices,
    faces,
    binary_mask,
    elevation,
    azimuth,
    feature,
    orthographic=False,
    dilation_iters=1,
    use_proximity_filter=False,
    proximity_threshold=0.15
):
    import numpy as np
    import open3d as o3d
    from scipy.ndimage import binary_dilation
    from scipy.spatial import cKDTree

    # 1. Prepare Mask
    if "torch" in str(type(binary_mask)):
        binary_mask = binary_mask.detach().cpu().numpy()
    if len(binary_mask.shape) == 3:
        binary_mask = np.squeeze(binary_mask, axis=0)
    if feature == "edge" and dilation_iters > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilation_iters)

    width, height = 1024, 1024
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # 2. Setup Mesh
    device = o3d.core.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex.positions = o3d.core.Tensor(vertices, o3d.core.float32, device)
    mesh.triangle.indices = o3d.core.Tensor(faces, o3d.core.int32, device)

    # 3. Build edge topology
    edge_set = set()
    for f in faces:
        edge_set.add(tuple(sorted((f[0], f[1]))))
        edge_set.add(tuple(sorted((f[1], f[2]))))
        edge_set.add(tuple(sorted((f[2], f[0]))))
    edges = np.array(list(edge_set), dtype=np.int64)
    edge_dict = {tuple(e): i for i, e in enumerate(edges)}
    tri_to_edges = np.array([[edge_dict[tuple(sorted((f[0], f[1])))], 
                              edge_dict[tuple(sorted((f[1], f[2])))], 
                              edge_dict[tuple(sorted((f[2], f[0])))]] for f in faces])

    # 4. Camera Setup
    bounds_min, bounds_max = vertices.min(axis=0), vertices.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    radius = (np.linalg.norm(bounds_max - bounds_min) / 2.0) * (1.2 if orthographic else 4.0)
    
    eye = np.array([center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth)),
                    center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth)),
                    center[2] - radius * np.sin(np.radians(-elevation))])
    
    forward = (center - eye) / (np.linalg.norm(center - eye) + 1e-9)
    right = np.cross(forward, [0, 0, 1]); right /= (np.linalg.norm(right) + 1e-9)
    true_up = np.cross(right, forward)
    cx, cy = width / 2.0, height / 2.0
    fx = fy = (height / 2.0) / np.tan(np.radians(15.0)) # 30 deg FOV

    # 5. Raycasting
    scene_raycast = o3d.t.geometry.RaycastingScene()
    scene_raycast.add_triangles(mesh)
    v_coords, u_coords = np.nonzero(binary_mask)
    if len(v_coords) == 0: return []

    rays = []
    for v, u in zip(v_coords, u_coords):
        x, y = (u - cx) / fx, (v - cy) / fy
        dir_w = (x * right - y * true_up + forward); dir_w /= np.linalg.norm(dir_w)
        rays.append(np.hstack([eye, dir_w]))

    rays_np = np.array(rays, dtype=np.float32)
    ans = scene_raycast.cast_rays(o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32))
    
    # 6. Filter Hits
    valid_mask = ans["primitive_ids"].numpy() != o3d.t.geometry.RaycastingScene.INVALID_ID
    hit_triangles_all = ans["primitive_ids"].numpy()[valid_mask]
    
    if use_proximity_filter:
        t_hit = ans["t_hit"].numpy()[valid_mask]
        hit_points = rays_np[valid_mask, 0:3] + (t_hit[:, np.newaxis] * rays_np[valid_mask, 3:6])
        face_centroids = vertices[faces].mean(axis=1)
        tree = cKDTree(hit_points)
        dist, _ = tree.query(face_centroids[hit_triangles_all])
        proximity_threshold *= radius/223 # proximity_thresh relative scaling, lets see if it works
        hit_triangles = np.unique(hit_triangles_all[dist < proximity_threshold])
    else:
        hit_triangles = np.unique(hit_triangles_all)

    if feature == "face":
        return hit_triangles.tolist()

    # 7. Edge Mode
    hit_edges = set()
    for tri_id in hit_triangles:
        for edge_id in tri_to_edges[tri_id]:
            v0, v1 = edges[edge_id]
            midpoint = (vertices[v0] + vertices[v1]) / 2.0
            vec_w = midpoint - eye
            Z, X, Y = np.dot(vec_w, forward), np.dot(vec_w, right), np.dot(vec_w, -true_up)
            if Z > 0:
                px, py = int(round(fx * (X / Z) + cx)), int(round(fy * (Y / Z) + cy))
                if 0 <= px < width and 0 <= py < height and binary_mask[py, px]:
                    hit_edges.add(edge_id)

    return [tuple(int(v) for v in edges[e]) for e in hit_edges]

def return_mesh_entities_from_2D_masks_(
    vertices,
    faces,
    binary_mask,
    elevation,
    azimuth,
    feature,  # "face" or "edge"
    orthographic=False,
    dilation_iters=1,
):
    import numpy as np
    import open3d as o3d
    from scipy.ndimage import binary_dilation

    if "torch" in str(type(binary_mask)):
        binary_mask = binary_mask.detach().cpu().numpy()

    if len(binary_mask.shape) == 3:
        binary_mask = np.squeeze(binary_mask, axis=0)

    if feature == "edge" and dilation_iters > 0:
        binary_mask = binary_dilation(binary_mask, iterations=dilation_iters)

    width = 1024
    height = 1024

    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    # -------------------------
    # Create Open3D Tensor Mesh directly (Super Fast, No Legacy Conversion)
    # -------------------------
    device = o3d.core.Device("CPU:0")
    mesh = o3d.t.geometry.TriangleMesh(device)
    mesh.vertex.positions = o3d.core.Tensor(vertices, o3d.core.float32, device)
    mesh.triangle.indices = o3d.core.Tensor(faces, o3d.core.int32, device)

    # -------------------------
    # Build unique mesh edges
    # -------------------------
    edge_set = set()
    for f in faces:
        edge_set.add(tuple(sorted((f[0], f[1]))))
        edge_set.add(tuple(sorted((f[1], f[2]))))
        edge_set.add(tuple(sorted((f[2], f[0]))))

    edges = np.array(list(edge_set), dtype=np.int64)
    edge_dict = {tuple(e): i for i, e in enumerate(edges)}

    tri_to_edges = []
    for f in faces:
        tri_edges =[
            edge_dict[tuple(sorted((f[0], f[1])))],
            edge_dict[tuple(sorted((f[1], f[2])))],
            edge_dict[tuple(sorted((f[2], f[0])))],
        ]
        tri_to_edges.append(tri_edges)

    tri_to_edges = np.array(tri_to_edges)

    # -------------------------
    # Camera setup
    # -------------------------
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = (bounds_min + bounds_max) / 2.0
    diag = np.linalg.norm(bounds_max - bounds_min)

    zoom = 1.2 if orthographic else 4.0
    radius = diag / 2.0 * zoom

    cam_x = center[0] - radius * np.cos(np.radians(-elevation)) * np.sin(np.radians(azimuth))
    cam_y = center[1] - radius * np.cos(np.radians(-elevation)) * np.cos(np.radians(azimuth))
    cam_z = center[2] - radius * np.sin(np.radians(-elevation))

    eye = np.array([cam_x, cam_y, cam_z])
    lookat = center
    up = np.array([0.0, 0.0, 1.0])

    # -------------------------
    # Proper Camera Frame Math (Renderer Removed!)
    # -------------------------
    # Build a right-handed orthonormal basis for the camera in World space
    forward = lookat - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, forward)  # Points UP relative to camera

    cx = width / 2.0
    cy = height / 2.0

    if not orthographic:
        fov = np.radians(30.0)
        fy = (height / 2.0) / np.tan(fov / 2.0)
        fx = fy  # Square pixels
    else:
        top = radius * 0.8
        world_height = 2.0 * top
        world_width = world_height * (width / height)

    # -------------------------
    # Raycasting
    # -------------------------
    scene_raycast = o3d.t.geometry.RaycastingScene()
    scene_raycast.add_triangles(mesh)

    v_coords, u_coords = np.nonzero(binary_mask)
    if len(v_coords) == 0:
        return[]

    rays =[]
    for v, u in zip(v_coords, u_coords):
        if not orthographic:
            x = (u - cx) / fx
            y = (v - cy) / fy
            # Direction in world space (note: -y * true_up because image Y goes DOWN)
            dir_w = x * right - y * true_up + forward
            dir_w = dir_w / np.linalg.norm(dir_w)
            orig_w = eye
        else:
            x_ndc = (u - cx) / cx
            y_ndc = (v - cy) / cy
            orig_w = eye + right * (x_ndc * world_width / 2.0) - true_up * (y_ndc * world_height / 2.0)
            dir_w = forward

        rays.append(np.hstack([orig_w, dir_w]))

    rays = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)
    ans = scene_raycast.cast_rays(rays)

    hit_triangles = ans["primitive_ids"].numpy()
    hit_triangles = hit_triangles[
        hit_triangles != o3d.t.geometry.RaycastingScene.INVALID_ID
    ]
    hit_triangles = np.unique(hit_triangles)
    

    if feature == "face":
        return hit_triangles.tolist()

    # -------------------------
    # EDGE MODE (filtered)
    # -------------------------
    hit_edges = set()

    for tri_id in hit_triangles:
        for edge_id in tri_to_edges[tri_id]:
            v0, v1 = edges[edge_id]
            midpoint = (vertices[v0] + vertices[v1]) / 2.0

            # Project midpoint to Camera Space utilizing our orthonormal basis
            vec_w = midpoint - eye
            Z = np.dot(vec_w, forward)
            X = np.dot(vec_w, right)
            Y = np.dot(vec_w, -true_up)  # -true_up corresponds to image Y going DOWN

            if Z > 0:  # Ensures the geometry is in front of the camera
                if not orthographic:
                    # Pinhole projection
                    px = fx * (X / Z) + cx
                    py = fy * (Y / Z) + cy
                else:
                    # Orthographic projection mapping
                    px = (X / (world_width / 2.0)) * cx + cx
                    py = (Y / (world_height / 2.0)) * cy + cy

                px = int(round(px))
                py = int(round(py))

                if 0 <= px < width and 0 <= py < height:
                    if binary_mask[py, px]:
                        hit_edges.add(edge_id)

    # Return actual (v0, v1) integer tuples instead of flat edge IDs
    return[tuple(int(v) for v in edges[e]) for e in hit_edges]


def cad_entity_to_mesh_faces(
    cad_file,
    vertices,
    faces,
    entity_type,
    entity_index,
    tol=1e-3
):
    """
    Returns:
        face mode -> list of face indices
        edge mode -> list of (v0, v1) vertex index pairs
    """
    import FreeCAD
    import Part
    import numpy as np

    vertices = np.asarray(vertices)
    faces = np.asarray(faces)

    # -----------------------------
    # Load CAD
    # -----------------------------
    doc = FreeCAD.newDocument("Recovery")
    Part.open(cad_file)

    shape_objs =[
        obj for obj in FreeCAD.ActiveDocument.Objects
        if hasattr(obj, "Shape") and not obj.Shape.isNull()
    ]

    if not shape_objs:
        FreeCAD.closeDocument(doc.Name)
        raise ValueError("No valid CAD shapes found.")

    obj = shape_objs[0]
    shape = obj.Shape.copy()
    # Apply the placement rigidly without rebuilding the underlying geometry
    shape.Placement = obj.Placement
    #shape = shape.transformGeometry(obj.Placement.toMatrix())

    if entity_type == "face":
        cad_entity = shape.Faces[entity_index]
    elif entity_type == "edge":
        cad_entity = shape.Edges[entity_index]
    else:
        FreeCAD.closeDocument(doc.Name)
        raise ValueError("entity_type must be 'face' or 'edge'")

    tri_vertices = vertices[faces]
    
    # Expand the BBox filter generously to account for chordal deviation 
    # (centroids of flat triangles on curved surfaces deviate from the surface).
    filter_tol = tol + 0.1

    # =========================================================
    # FACE MODE
    # =========================================================
    if entity_type == "face":

        centroids = tri_vertices.mean(axis=1)
        bbox = cad_entity.BoundBox

        in_bbox = (
            (centroids[:, 0] >= bbox.XMin - filter_tol) &
            (centroids[:, 0] <= bbox.XMax + filter_tol) &
            (centroids[:, 1] >= bbox.YMin - filter_tol) &
            (centroids[:, 1] <= bbox.YMax + filter_tol) &
            (centroids[:, 2] >= bbox.ZMin - filter_tol) &
            (centroids[:, 2] <= bbox.ZMax + filter_tol)
        )

        candidate_indices = np.where(in_bbox)[0]

        matching_faces =[]

        for i in candidate_indices:
            # FIX: Check if the 3 vertices lie on the face, robust against chordal deviation
            v0 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 0]))
            v1 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 1]))
            v2 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 2]))
            
            d0, _, _ = cad_entity.distToShape(v0)
            if d0 > tol: continue
            
            d1, _, _ = cad_entity.distToShape(v1)
            if d1 > tol: continue
            
            d2, _, _ = cad_entity.distToShape(v2)
            if d2 > tol: continue

            matching_faces.append(int(i))

        FreeCAD.closeDocument(doc.Name)
        return matching_faces

    # =========================================================
    # EDGE MODE (FAST + TOPOLOGICALLY CORRECT)
    # =========================================================
    else:

        # 1. Find adjacent CAD faces
        adj_faces =[]
        for f in shape.Faces:
            for e in f.Edges:
                if e.isSame(cad_entity):
                    adj_faces.append(f)
                    break

        if len(adj_faces) == 0:
            FreeCAD.closeDocument(doc.Name)
            return[]

        # -----------------------------------
        # 2. Map CAD faces -> mesh face indices
        # -----------------------------------
        mesh_face_sets =[]

        for f in adj_faces:
            centroids = tri_vertices.mean(axis=1)
            bbox = f.BoundBox

            in_bbox = (
                (centroids[:, 0] >= bbox.XMin - filter_tol) &
                (centroids[:, 0] <= bbox.XMax + filter_tol) &
                (centroids[:, 1] >= bbox.YMin - filter_tol) &
                (centroids[:, 1] <= bbox.YMax + filter_tol) &
                (centroids[:, 2] >= bbox.ZMin - filter_tol) &
                (centroids[:, 2] <= bbox.ZMax + filter_tol)
            )

            candidate_indices = np.where(in_bbox)[0]

            matched = set()
            for i in candidate_indices:
                # FIX: Check if the 3 vertices lie on the face
                v0 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 0]))
                v1 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 1]))
                v2 = Part.Vertex(FreeCAD.Vector(*tri_vertices[i, 2]))
                
                d0, _, _ = f.distToShape(v0)
                if d0 > tol: continue
                d1, _, _ = f.distToShape(v1)
                if d1 > tol: continue
                d2, _, _ = f.distToShape(v2)
                if d2 > tol: continue
                
                matched.add(int(i))

            mesh_face_sets.append(matched)

        if len(mesh_face_sets) == 1:
            faceA = mesh_face_sets[0]
            faceB = set()
        else:
            faceA, faceB = mesh_face_sets[:2]

        # -----------------------------------
        # 3. Build edge -> face adjacency
        # -----------------------------------
        edge_to_faces = {}

        for fi in faceA | faceB:
            tri = faces[fi]
            edges_local =[
                tuple(sorted((tri[0], tri[1]))),
                tuple(sorted((tri[1], tri[2]))),
                tuple(sorted((tri[2], tri[0]))),
            ]

            for e in edges_local:
                edge_to_faces.setdefault(e, set()).add(fi)

        # -----------------------------------
        # 4. Keep edges shared between faceA and faceB
        # -----------------------------------
        matching_edges =[]

        if len(mesh_face_sets) == 1:
            # FIX: Handle Open Shells / Boundary Edges properly
            for e, adj in edge_to_faces.items():
                if len(adj) == 1: # verify it is a structural boundary edge in the mesh
                    v0 = Part.Vertex(FreeCAD.Vector(*vertices[e[0]]))
                    v1 = Part.Vertex(FreeCAD.Vector(*vertices[e[1]]))
                    d0, _, _ = cad_entity.distToShape(v0)
                    if d0 > tol: continue
                    d1, _, _ = cad_entity.distToShape(v1)
                    if d1 > tol: continue
                    matching_edges.append(e)
        else:
            # Shared Edges
            for e, adj in edge_to_faces.items():
                inA = any(fi in faceA for fi in adj)
                inB = any(fi in faceB for fi in adj)

                if inA and inB:
                    matching_edges.append(e)

        FreeCAD.closeDocument(doc.Name)
        return matching_edges



if __name__ == '__main__':


    cad_file_path = '/data/1bali/Other_LLM_projects/ECCV_2026/ABC_CAD_Dataset_small2/00460032/00460032_588014cd814ad91039d47561_step_001.step'
    mesh_path = cad_file_path.replace(".step",".obj")
    mesh = trimesh.load(mesh_path)
    mesh.vertices, mesh.faces = stitch_mesh_topology(mesh.vertices, mesh.faces)
    feature, feature_idx = 'edge', 48
    gt_edges = cad_entity_to_mesh_faces(cad_file_path, mesh.vertices, mesh.faces, entity_type=feature, entity_index=feature_idx)
    
    print('Lets See')