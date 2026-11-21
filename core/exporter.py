"""
Multi-Format 3D Mesh Exporter (.OBJ, .GLB, .STL, .PLY) & Geometry Post-Processing.
Designed for seamless import into Blender, Autodesk Maya, 3ds Max, Unity, Unreal Engine,
Cinema 4D, ZBrush, MeshLab, CloudCompare, SolidWorks, and 3D Printing Slicers (Cura, PrusaSlicer).
"""
import json
import struct
import numpy as np


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

    # Generate subtle studio height/normal shading vertex colors
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

    # Align each bufferView to 4-byte boundaries
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
                "componentType": 5126,  # FLOAT
                "count": int(len(verts)),
                "type": "VEC3",
                "min": [float(x) for x in v_min],
                "max": [float(x) for x in v_max]
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,  # FLOAT
                "count": int(len(vertex_normals)),
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": int(faces.size),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(pos_bytes),
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes_p),
                "byteLength": len(nrm_bytes),
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(pos_bytes_p) + len(nrm_bytes_p),
                "byteLength": len(idx_bytes),
                "target": 34963  # ELEMENT_ARRAY_BUFFER
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
