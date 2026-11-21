"""
Icosphere Template Loader & Pure-NumPy Subdivision Generator.
Ensures 100% compatibility with trained Session 1 & Session 2 checkpoints (642 vertices, 1280 faces, 1920 edges).
"""
import os
import math
import numpy as np


def generate_icosphere_numpy(subdivisions=3, radius=0.65):
    """
    Pure-NumPy Subdivision Icosphere Generator (fallback if icosphere_template.npz is not found).
    subdivisions=3 -> 642 vertices, 1280 triangular faces, 1920 unique edges.
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1,  t,  0], [ 1,  t,  0], [-1, -t,  0], [ 1, -t,  0],
        [ 0, -1,  t], [ 0,  1,  t], [ 0, -1, -t], [ 0,  1, -t],
        [ t,  0, -1], [ t,  0,  1], [-t,  0, -1], [-t,  0,  1],
    ], dtype=np.float64)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)

    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    v_list = verts.tolist()
    for _ in range(subdivisions):
        midpoint_cache = {}
        new_faces = []

        def get_midpoint(i1, i2):
            key = (min(i1, i2), max(i1, i2))
            if key in midpoint_cache:
                return midpoint_cache[key]
            p1 = np.array(v_list[i1])
            p2 = np.array(v_list[i2])
            pm = (p1 + p2) * 0.5
            pm /= (np.linalg.norm(pm) + 1e-12)
            idx = len(v_list)
            v_list.append(pm.tolist())
            midpoint_cache[key] = idx
            return idx

        for tri in faces:
            v0, v1, v2 = int(tri[0]), int(tri[1]), int(tri[2])
            a = get_midpoint(v0, v1)
            b = get_midpoint(v1, v2)
            c = get_midpoint(v2, v0)
            new_faces.extend([
                [v0, a, c],
                [v1, b, a],
                [v2, c, b],
                [a, b, c],
            ])
        faces = np.array(new_faces, dtype=np.int64)

    verts_arr = (np.array(v_list, dtype=np.float32) * float(radius)).astype(np.float32)
    edge_set = set()
    for f in faces:
        for u, v in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            edge_set.add((min(int(u), int(v)), max(int(u), int(v))))
    edges_arr = np.array(sorted(list(edge_set)), dtype=np.int64)

    n_v = len(verts_arr)
    adj = np.eye(n_v, dtype=np.float32)
    for u, v in edges_arr:
        adj[u, v] = 1.0
        adj[v, u] = 1.0
    deg = np.sum(adj, axis=1, keepdims=True)
    adj_norm = (adj / np.maximum(deg, 1.0)).astype(np.float32)
    return verts_arr, faces, edges_arr, adj_norm


def load_icosphere_template(base_dir=None):
    """
    Loads the canonical Icosphere template from assets/icosphere_template.npz,
    or generates and caches it if missing.
    """
    candidates = []
    if base_dir:
        candidates.append(os.path.join(base_dir, "assets", "icosphere_template.npz"))
    candidates.extend([
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icosphere_template.npz"),
        r"G:\My Drive\Colab Notebooks\ChuyenDe_TGMT\Bài tập nhóm 2\notebook\dataset\preparation_canonical_mesh_s1_s2\icosphere_template.npz",
    ])

    for p in candidates:
        if os.path.exists(p):
            data = np.load(p)
            return (
                data["vertices"].astype(np.float32),
                data["faces"].astype(np.int64),
                data["edges"].astype(np.int64),
                data["adj_norm"].astype(np.float32),
            )

    verts, faces, edges, adj_norm = generate_icosphere_numpy(subdivisions=3, radius=0.65)
    if base_dir:
        save_p = os.path.join(base_dir, "assets", "icosphere_template.npz")
        os.makedirs(os.path.dirname(save_p), exist_ok=True)
        np.savez_compressed(save_p, vertices=verts, faces=faces, edges=edges, adj_norm=adj_norm)
    return verts, faces, edges, adj_norm
