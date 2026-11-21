"""
Multi-Format 3D Mesh Exporter (.OBJ, .GLB, .STL, .PLY) & Computational Geometry Surface Fairing Engine.
Includes:
  - Stage 1: Topological Outlier Despiking & Voxel-Uniformized Surface Cloud
  - Stage 2: Bi-Directional Elastic Shrink-Wrap & Bilateral Symmetry Regularization
  - Stage 3: Taubin lambda|mu Volume-Preserving Spectral Fairing & Tangential Vertex Equalization
  - Stage 4: Loop-Taubin Surface Subdivision (642v / 1280f -> 2562v / 5120f)
"""
import json
import struct
import numpy as np

# Module-level cache for precomputed topological symmetry & subdivision operators
_TOPO_CACHE = {}


def _get_topological_operators(tpl_verts, tpl_faces):
    """Precomputes symmetry permutation and sparse Loop-Taubin subdivision operators once."""
    key = (len(tpl_verts), len(tpl_faces))
    if key in _TOPO_CACHE:
        return _TOPO_CACHE[key]

    tv = np.asarray(tpl_verts, dtype=np.float32)
    tf = np.asarray(tpl_faces, dtype=np.int64)
    tv_unit = (tv / (np.linalg.norm(tv, axis=1, keepdims=True) + 1e-8)).astype(np.float32)

    tv_mirror = tv_unit.copy()
    tv_mirror[:, 0] = -tv_mirror[:, 0]
    sym_perm = np.argmin(np.linalg.norm(tv_unit[:, None, :] - tv_mirror[None, :, :], axis=-1), axis=1)

    edge_to_new_vert = {}
    edge_pairs = []
    new_faces = []
    next_idx = int(tf.max()) + 1

    def get_edge_vert(u, v):
        nonlocal next_idx
        k = (min(int(u), int(v)), max(int(u), int(v)))
        if k not in edge_to_new_vert:
            edge_to_new_vert[k] = next_idx
            edge_pairs.append([k[0], k[1]])
            next_idx += 1
        return edge_to_new_vert[k]

    for tri in tf:
        v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
        a = get_edge_vert(v0, v1)
        b = get_edge_vert(v1, v2)
        c = get_edge_vert(v2, v0)
        new_faces.extend([
            [v0, a, c],
            [v1, b, a],
            [v2, c, b],
            [a, b, c],
        ])

    edge_pairs_arr = np.array(edge_pairs, dtype=np.int64)
    new_faces_arr = np.array(new_faces, dtype=np.int64)

    # Precompute 1-ring neighbor index list padded to max_deg=6 for fast O(N) smoothing on 2562 vertices
    n_sub = next_idx
    neigh_sets = [set() for _ in range(n_sub)]
    for f in new_faces_arr:
        u, v, w = int(f[0]), int(f[1]), int(f[2])
        neigh_sets[u].update((v, w))
        neigh_sets[v].update((u, w))
        neigh_sets[w].update((u, v))

    sub_neigh_idx = np.zeros((n_sub, 6), dtype=np.int64)
    sub_neigh_w = np.zeros((n_sub, 6), dtype=np.float32)
    for i in range(n_sub):
        nbs = sorted(list(neigh_sets[i]))[:6]
        deg = len(nbs)
        if deg > 0:
            sub_neigh_idx[i, :deg] = nbs
            sub_neigh_w[i, :deg] = 1.0 / float(deg)
        else:
            sub_neigh_idx[i, 0] = i
            sub_neigh_w[i, 0] = 1.0

    ops = {
        "tv_unit": tv_unit,
        "sym_perm": sym_perm,
        "edge_pairs": edge_pairs_arr,
        "sub_faces": new_faces_arr,
        "sub_neigh_idx": sub_neigh_idx,
        "sub_neigh_w": sub_neigh_w,
    }
    _TOPO_CACHE[key] = ops
    return ops


def _voxel_downsample_point_cloud(pts, voxel_size=0.045):
    """Uniformly downsamples a 3D point cloud via voxel grid averaging to remove vertex-clustering density bias."""
    q = np.floor(pts / float(voxel_size)).astype(np.int32)
    _, inv, counts = np.unique(q, axis=0, return_inverse=True, return_counts=True)
    vox_pts = np.zeros((len(counts), 3), dtype=np.float32)
    np.add.at(vox_pts, inv, pts)
    vox_pts /= counts[:, None].astype(np.float32)
    return vox_pts


def compute_vertex_and_face_normals(verts, faces):
    """
    Computes smooth area-weighted vertex normals (Nv, 3) and unit face normals (Nf, 3)
    from mesh vertices (Nv, 3) and triangular faces (Nf, 3).
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    raw_fn = np.cross(v1 - v0, v2 - v0)
    fn_len = np.linalg.norm(raw_fn, axis=1, keepdims=True) + 1e-8
    face_normals = (raw_fn / fn_len).astype(np.float32)

    v_normals = np.zeros_like(verts, dtype=np.float32)
    np.add.at(v_normals, faces[:, 0], raw_fn)
    np.add.at(v_normals, faces[:, 1], raw_fn)
    np.add.at(v_normals, faces[:, 2], raw_fn)

    vn_len = np.linalg.norm(v_normals, axis=1, keepdims=True)
    fallback_radial = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-8)
    v_normals = np.where(vn_len > 1e-8, v_normals / np.maximum(vn_len, 1e-8), fallback_radial).astype(np.float32)
    return v_normals, face_normals


def apply_adaptive_edge_pruning(verts, faces, stretch_ratio=2.35):
    """
    Adaptive Topology Edge Pruning: Removes abnormally stretched triangles spanning across topological holes.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    e01 = np.linalg.norm(v0 - v1, axis=-1)
    e12 = np.linalg.norm(v1 - v2, axis=-1)
    e20 = np.linalg.norm(v2 - v0, axis=-1)
    max_edge = np.maximum(np.maximum(e01, e12), e20)
    median_edge = np.median(max_edge)
    keep = max_edge < (median_edge * float(stretch_ratio))
    return faces[keep] if np.sum(keep) >= 64 else faces


def refine_and_fair_3d_mesh(v_raw, tpl_verts, tpl_faces, adj_norm, refine_mode="cad_smooth", use_pruning=False):
    """
    4-Stage Computational Geometry Mesh Refinement Pipeline (~8-12 ms in pure float32 BLAS):
      - 'cad_smooth' (Default):
          1. Outlier Despiking & Voxel-Downsampled Uniform Point Cloud Target
          2. Bi-Directional Elastic Surface Shrink-Wrap + Bilateral Symmetry Regularization
          3. Taubin lambda|mu Volume-Preserving Spectral Fairing + Tangential Relaxation
          4. Loop-Taubin Surface Subdivision (642v / 1280f -> 2562v / 5120f)
      - 'taubin_smooth':
          Despiking + Taubin lambda|mu Spectral Fairing + Tangential Relaxation on 642v / 1280f.
      - 'raw':
          Unmodified raw neural network output (642v / 1280f).
    """
    pts = np.asarray(v_raw, dtype=np.float32).copy()
    faces_base = np.asarray(tpl_faces, dtype=np.int64).copy()

    if refine_mode == "raw":
        if use_pruning:
            faces_base = apply_adaptive_edge_pruning(pts, faces_base)
        return pts, faces_base

    ops = _get_topological_operators(tpl_verts, tpl_faces)
    sym_perm = ops["sym_perm"]
    tv_unit = ops["tv_unit"]

    pts[:, 0] -= np.median(pts[:, 0])

    if refine_mode == "taubin_smooth":
        v = pts.copy()
        for _ in range(3):
            neigh_c = adj_norm @ v
            delta = v - neigh_c
            d_len = np.linalg.norm(delta, axis=1, keepdims=True)
            q75 = float(np.percentile(d_len, 75)) + 1e-6
            max_allow = 1.35 * q75
            v = neigh_c + delta * np.where(d_len > max_allow, max_allow / (d_len + 1e-8), 1.0).astype(np.float32)

        v_refl = v[sym_perm].copy()
        v_refl[:, 0] = -v_refl[:, 0]
        v = (0.68 * v + 0.32 * v_refl).astype(np.float32)

        for _ in range(5):
            v = v + np.float32(0.46) * (adj_norm @ v - v)
            v = v - np.float32(0.485) * (adj_norm @ v - v)

        for _ in range(2):
            vn, _ = compute_vertex_and_face_normals(v, faces_base)
            delta = (adj_norm @ v) - v
            v = v + np.float32(0.45) * (delta - np.sum(delta * vn, axis=1, keepdims=True) * vn)

        if use_pruning:
            faces_base = apply_adaptive_edge_pruning(v, faces_base)
        return v.astype(np.float32), faces_base

    # Full 'cad_smooth' Pipeline: Voxel-Uniformized Bi-Directional Elastic Shrink-Wrap + Taubin + Loop Subdivision
    neigh_raw = adj_norm @ pts
    d_raw = np.linalg.norm(pts - neigh_raw, axis=1)
    q80 = float(np.percentile(d_raw, 80)) + 1e-5
    clean_pts = pts[d_raw <= 2.0 * q80]
    if len(clean_pts) < 100:
        clean_pts = pts

    pts_sym = clean_pts.copy()
    pts_sym[:, 0] = -pts_sym[:, 0]
    raw_cloud = np.concatenate([clean_pts, pts_sym], axis=0)
    target_cloud = _voxel_downsample_point_cloud(raw_cloud, voxel_size=0.045)
    tc_norm2 = np.sum(target_cloud ** 2, axis=1, keepdims=True).T  # (1, M) float32

    p_min = np.percentile(target_cloud, 3, axis=0).astype(np.float32)
    p_max = np.percentile(target_cloud, 97, axis=0).astype(np.float32)
    center = 0.5 * (p_min + p_max)
    half_ext = 0.5 * (p_max - p_min) + np.float32(0.025)

    v = (tv_unit * half_ext[None, :] + center[None, :]).astype(np.float32)

    sigmas = [0.15, 0.11, 0.085, 0.065, 0.05]
    alphas = [0.55, 0.50, 0.46,  0.42,  0.38]

    for sigma, alpha in zip(sigmas, alphas):
        inv_2s2 = np.float32(-1.0 / (2.0 * sigma * sigma))
        v_norm2 = np.sum(v ** 2, axis=1, keepdims=True)  # (642, 1)
        # Fast BLAS float32 squared distance matrix: ||v - c||^2
        dist2 = np.maximum(v_norm2 + tc_norm2 - np.float32(2.0) * (v @ target_cloud.T), np.float32(0.0))

        min_v2c = np.min(dist2, axis=1, keepdims=True)
        w_v2c = np.exp((dist2 - min_v2c) * inv_2s2)
        w_v2c /= (np.sum(w_v2c, axis=1, keepdims=True) + np.float32(1e-8))
        p_forward = w_v2c @ target_cloud

        min_c2v = np.min(dist2, axis=0, keepdims=True)
        w_c2v = np.exp((dist2 - min_c2v) * inv_2s2)
        w_c2v_sum = np.sum(w_c2v, axis=1, keepdims=True)
        p_reverse = np.where(w_c2v_sum > 1e-5, (w_c2v @ target_cloud) / np.maximum(w_c2v_sum, np.float32(1e-5)), p_forward)

        p_target = np.float32(0.58) * p_forward + np.float32(0.42) * p_reverse
        v = np.float32(1.0 - alpha) * v + np.float32(alpha) * p_target

        v_refl = v[sym_perm].copy()
        v_refl[:, 0] = -v_refl[:, 0]
        v = np.float32(0.70) * v + np.float32(0.30) * v_refl

        for _ in range(2):
            v = v + np.float32(0.46) * (adj_norm @ v - v)
            v = v - np.float32(0.485) * (adj_norm @ v - v)

    for _ in range(3):
        vn, _ = compute_vertex_and_face_normals(v, faces_base)
        delta = (adj_norm @ v) - v
        v = v + np.float32(0.52) * (delta - np.sum(delta * vn, axis=1, keepdims=True) * vn)

    # Stage 4: Loop-Taubin Surface Subdivision (642 -> 2562 vertices, 1280 -> 5120 faces)
    edge_pairs = ops["edge_pairs"]
    sub_faces = ops["sub_faces"].copy()
    sub_n_idx = ops["sub_neigh_idx"]
    sub_n_w = ops["sub_neigh_w"][:, :, None]  # (2562, 6, 1)

    mid_verts = np.float32(0.5) * (v[edge_pairs[:, 0]] + v[edge_pairs[:, 1]])
    v_sub = np.concatenate([v, mid_verts], axis=0)

    # Fast O(N) sparse 1-ring Taubin fairing on the 2562 subdivided vertices
    for _ in range(2):
        neigh_avg = np.sum(v_sub[sub_n_idx] * sub_n_w, axis=1)
        v_sub = v_sub + np.float32(0.42) * (neigh_avg - v_sub)
        neigh_avg2 = np.sum(v_sub[sub_n_idx] * sub_n_w, axis=1)
        v_sub = v_sub - np.float32(0.44) * (neigh_avg2 - v_sub)

    if use_pruning:
        sub_faces = apply_adaptive_edge_pruning(v_sub, sub_faces)

    return v_sub.astype(np.float32), sub_faces


def export_to_obj_bytes(verts, faces, vertex_normals=None, model_name="GeoAttMesh_3D"):
    """
    Exports 3D mesh to Wavefront .OBJ format with vertex normals (v, vn, f v//vn).
    Compatible with: Blender, Maya, 3ds Max, Cinema 4D, ZBrush, Rhino.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertex_normals is None:
        vertex_normals, _ = compute_vertex_and_face_normals(verts, faces)

    lines = [
        f"# Wavefront OBJ exported by Image-to-3D Mesh Reconstruction Studio",
        f"# Model: {model_name}",
        f"# Vertices: {len(verts)} | Faces: {len(faces)}",
        f"o {model_name}",
    ]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for vn in vertex_normals:
        lines.append(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}")
    lines.append("s 1")
    for f in faces:
        i1, i2, i3 = int(f[0]) + 1, int(f[1]) + 1, int(f[2]) + 1
        lines.append(f"f {i1}//{i1} {i2}//{i2} {i3}//{i3}")

    return ("\n".join(lines) + "\n").encode("utf-8")


def export_to_stl_bytes(verts, faces, model_name="GeoAttMesh_3D"):
    """
    Exports 3D mesh to standard Binary .STL format.
    Compatible with: 3D Printing Slicers (Cura, PrusaSlicer, Bambu Studio), SolidWorks, AutoCAD, Fusion 360.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    _, face_normals = compute_vertex_and_face_normals(verts, faces)

    header_str = f"Binary STL - {model_name} (Image-to-3D Studio)"[:80].encode("ascii", errors="replace")
    header = header_str.ljust(80, b"\x00")
    num_tris = len(faces)

    chunks = [header, struct.pack("<I", num_tris)]
    for idx in range(num_tris):
        fn = face_normals[idx]
        v0 = verts[faces[idx, 0]]
        v1 = verts[faces[idx, 1]]
        v2 = verts[faces[idx, 2]]
        chunks.append(struct.pack(
            "<12fH",
            float(fn[0]), float(fn[1]), float(fn[2]),
            float(v0[0]), float(v0[1]), float(v0[2]),
            float(v1[0]), float(v1[1]), float(v1[2]),
            float(v2[0]), float(v2[1]), float(v2[2]),
            0
        ))
    return b"".join(chunks)


def export_to_ply_bytes(verts, faces, vertex_normals=None, model_name="GeoAttMesh_3D"):
    """
    Exports 3D mesh to Stanford .PLY format with vertex positions, normals, and height-mapped RGB colors.
    Compatible with: MeshLab, CloudCompare, Open3D, Blender.
    """
    verts = np.asarray(verts, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if vertex_normals is None:
        vertex_normals, _ = compute_vertex_and_face_normals(verts, faces)

    y_norm = (verts[:, 1] - verts[:, 1].min()) / (np.ptp(verts[:, 1]) + 1e-6)
    r_col = np.clip(95 + 130 * y_norm, 0, 255).astype(np.uint8)
    g_col = np.clip(145 + 95 * y_norm, 0, 255).astype(np.uint8)
    b_col = np.clip(225 + 25 * y_norm, 0, 255).astype(np.uint8)

    lines = [
        "ply",
        "format ascii 1.0",
        f"comment Exported by Image-to-3D Studio ({model_name})",
        f"element vertex {len(verts)}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        f"element face {len(faces)}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    for i in range(len(verts)):
        v = verts[i]
        vn = vertex_normals[i]
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f} {int(r_col[i])} {int(g_col[i])} {int(b_col[i])}")
    for f in faces:
        lines.append(f"3 {int(f[0])} {int(f[1])} {int(f[2])}")

    return ("\n".join(lines) + "\n").encode("utf-8")


def export_to_glb_bytes(verts, faces, vertex_normals=None, model_name="GeoAttMesh_3D"):
    """
    Exports 3D mesh to Khronos Binary glTF 2.0 (.GLB) format with double-sided PBR material.
    Compatible with: Blender, Unity, Unreal Engine, Godot, Three.js, Windows/macOS 3D Viewer.
    """
    verts = np.ascontiguousarray(verts, dtype=np.float32)
    faces = np.ascontiguousarray(faces, dtype=np.uint32)
    if vertex_normals is None:
        vertex_normals, _ = compute_vertex_and_face_normals(verts, faces)
    vertex_normals = np.ascontiguousarray(vertex_normals, dtype=np.float32)

    pos_bytes = verts.tobytes()
    nrm_bytes = vertex_normals.tobytes()
    idx_bytes = faces.flatten().tobytes()

    def pad4(b, pad_char=b"\x00"):
        rem = len(b) % 4
        return b if rem == 0 else b + pad_char * (4 - rem)

    pos_bytes_p = pad4(pos_bytes)
    nrm_bytes_p = pad4(nrm_bytes)
    idx_bytes_p = pad4(idx_bytes)

    bin_chunk = pos_bytes_p + nrm_bytes_p + idx_bytes_p

    v_min = verts.min(axis=0).tolist()
    v_max = verts.max(axis=0).tolist()

    gltf_dict = {
        "asset": {
            "version": "2.0",
            "generator": "Image-to-3D Mesh Reconstruction Studio"
        },
        "scene": 0,
        "scenes": [{"name": "Scene", "nodes": [0]}],
        "nodes": [{"name": model_name, "mesh": 0}],
        "materials": [{
            "name": "StudioClayPBR",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorFactor": [0.38, 0.65, 0.98, 1.0],
                "metallicFactor": 0.15,
                "roughnessFactor": 0.45
            }
        }],
        "meshes": [{
            "name": model_name,
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1
                },
                "indices": 2,
                "material": 0,
                "mode": 4
            }]
        }],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(verts)),
                "type": "VEC3",
                "min": [float(x) for x in v_min],
                "max": [float(x) for x in v_max]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": int(len(vertex_normals)),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,
                "count": int(faces.size),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(pos_bytes),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes_p),
                "byteLength": len(nrm_bytes),
                "target": 34962
            },
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes_p) + len(nrm_bytes_p),
                "byteLength": len(idx_bytes),
                "target": 34963
            }
        ],
        "buffers": [{
            "byteLength": len(bin_chunk)
        }]
    }

    json_bytes = pad4(json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8"), pad_char=b" ")
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_chunk)

    glb_header = struct.pack("<III", 0x46546C67, 2, total_len)
    json_header = struct.pack("<II", len(json_bytes), 0x4E4F534A)
    bin_header = struct.pack("<II", len(bin_chunk), 0x004E4942)

    return glb_header + json_header + json_bytes + bin_header + bin_chunk
