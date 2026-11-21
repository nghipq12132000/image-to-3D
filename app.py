"""
FastAPI Backend Server for Image-to-3D Mesh Reconstruction Studio.
Zero-dependency on python-multipart (uses direct raw binary streaming & JSON endpoints).
Provides:
  - Model Import & Directory Scanner (.pt / .pth from Session 1 & Session 2)
  - Multi-Model 2D-to-3D Mesh Inference Engine
  - Multi-Format 3D Export (.OBJ, .GLB, .STL, .PLY, and Batch .ZIP)
  - Real-Time Server Hardware & System Specifications Telemetry
"""
import os
import io
import sys
import time
import uuid
import json
import base64
import shutil
import socket
import zipfile
import platform
from datetime import datetime
from typing import List, Optional

import numpy as np
import psutil
import torch
from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.icosphere_template import load_icosphere_template
from core.models import inspect_checkpoint_metadata, load_model_from_checkpoint
from core.preprocess import preprocess_image_bytes
from core.exporter import (
    compute_vertex_and_face_normals,
    apply_adaptive_edge_pruning,
    refine_and_fair_3d_mesh,
    export_to_obj_bytes,
    export_to_stl_bytes,
    export_to_ply_bytes,
    export_to_glb_bytes,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
SAMPLES_DIR = os.path.join(ASSETS_DIR, "sample_images")
IMPORTED_MODELS_DIR = os.path.join(BASE_DIR, "models", "imported")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")

for d in [STATIC_DIR, ASSETS_DIR, SAMPLES_DIR, IMPORTED_MODELS_DIR, EXPORTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load canonical Icosphere template once
TPL_VERTS_NP, TPL_FACES_NP, TPL_EDGES_NP, TPL_ADJ_NP = load_icosphere_template(BASE_DIR)
TPL_VERTS_T = torch.tensor(TPL_VERTS_NP, dtype=torch.float32, device=DEVICE)
TPL_ADJ_T = torch.tensor(TPL_ADJ_NP, dtype=torch.float32, device=DEVICE)

app = FastAPI(
    title="Image-to-3D Mesh Reconstruction Studio",
    description="Single-View 3D Mesh Reconstruction & Multi-Model Comparison Studio",
    version="2.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

# In-memory registries
MODEL_REGISTRY = {}       # model_id -> metadata dict
LOADED_MODELS_CACHE = {}  # model_id -> PyTorch nn.Module
JOBS_CACHE = {}           # job_id -> {image_name, models: {model_id: {verts, faces, normals, info}}}

DEFAULT_SESSION2_DIR = r"G:\My Drive\Colab Notebooks\ChuyenDe_TGMT\Bài tập nhóm 2\notebook\training\session2\saved_models"
DEFAULT_SESSION1_DIR = r"G:\My Drive\Colab Notebooks\ChuyenDe_TGMT\Bài tập nhóm 2\notebook\training\session1\saved_models"


def _get_cpu_brand_name():
    """Retrieves human-readable CPU brand name on Windows/Linux."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            cpu_name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            winreg.CloseKey(key)
            if cpu_name:
                return cpu_name.strip()
        except Exception:
            pass
    return platform.processor() or "Generic x86_64 Processor"


CPU_BRAND_NAME = _get_cpu_brand_name()


def register_checkpoint_file(pt_path: str, fast_scan: bool = True) -> Optional[dict]:
    """Inspects and registers a single .pt / .pth checkpoint file into MODEL_REGISTRY."""
    if not os.path.exists(pt_path) or not pt_path.lower().endswith((".pt", ".pth")):
        return None
    abs_path = os.path.abspath(pt_path)
    for existing_id, info in MODEL_REGISTRY.items():
        if os.path.abspath(info["path"]) == abs_path:
            return info

    try:
        meta = inspect_checkpoint_metadata(abs_path, fast_scan=fast_scan)
        fname = os.path.basename(abs_path)
        clean_name = os.path.splitext(fname)[0].replace("_best", "")
        session_tag = "S2Run" if meta["session_version"] == "session2" else "S1Run"
        model_id = f"{session_tag}_{clean_name}_{uuid.uuid4().hex[:6]}"

        entry = {
            "id": model_id,
            "name": clean_name,
            "filename": fname,
            "path": abs_path,
            "backbone": meta["backbone_name"],
            "decoder": meta["decoder_type"],
            "session_version": meta["session_version"],
            "session_label": meta["session_label"],
            "split_tag": meta["split_tag"],
            "params_m": meta["params_m"],
            "file_size_mb": meta["file_size_mb"],
            "feat_dim": meta["feat_dim"],
        }
        MODEL_REGISTRY[model_id] = entry
        return entry
    except Exception as e:
        print(f"[WARN] Could not register checkpoint {pt_path}: {e}")
        return None


def auto_discover_initial_models():
    """Fast-scans local models/imported and default Session 2 directory on startup."""
    for folder in [IMPORTED_MODELS_DIR, DEFAULT_SESSION2_DIR]:
        if os.path.exists(folder):
            for fname in sorted(os.listdir(folder)):
                if fname.lower().endswith((".pt", ".pth")):
                    register_checkpoint_file(os.path.join(folder, fname), fast_scan=True)


auto_discover_initial_models()


class ScanDirRequest(BaseModel):
    folder_path: str


class ConvertRequest(BaseModel):
    model_ids: List[str]
    mask_mode: str = "auto"
    sensitivity: float = 1.0
    refine_mode: str = "cad_smooth"
    use_pruning: bool = False
    sample_name: Optional[str] = None
    image_base64: Optional[str] = None
    image_filename: Optional[str] = None


class PreviewMaskRequest(BaseModel):
    mask_mode: str = "auto"
    sensitivity: float = 1.0
    sample_name: Optional[str] = None
    image_base64: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/system-specs")
async def get_system_specs():
    """Returns comprehensive real-time server hardware & environment configuration."""
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    cpu_pct = psutil.cpu_percent(interval=0.1)
    per_core = psutil.cpu_percent(interval=0.05, percpu=True)

    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "total_vram_mb": round(props.total_memory / (1024 ** 2), 1),
                "allocated_vram_mb": round(torch.cuda.memory_allocated(i) / (1024 ** 2), 1),
                "reserved_vram_mb": round(torch.cuda.memory_reserved(i) / (1024 ** 2), 1),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })

    disks = []
    seen_mounts = set()
    for part in psutil.disk_partitions(all=False):
        if part.mountpoint in seen_mounts:
            continue
        try:
            usage = psutil.disk_usage(part.mountpoint)
            seen_mounts.add(part.mountpoint)
            disks.append({
                "device": part.device,
                "mountpoint": part.mountpoint,
                "fstype": part.fstype,
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "used_gb": round(usage.used / (1024 ** 3), 2),
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "percent": usage.percent,
            })
        except Exception:
            pass

    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": socket.gethostname(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
        },
        "cpu": {
            "brand": CPU_BRAND_NAME,
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_threads": psutil.cpu_count(logical=True) or 1,
            "current_freq_mhz": round(cpu_freq.current, 1) if cpu_freq else 0,
            "max_freq_mhz": round(cpu_freq.max, 1) if cpu_freq and cpu_freq.max else 0,
            "usage_percent": cpu_pct,
            "per_core_percent": per_core,
        },
        "ram": {
            "total_gb": round(vm.total / (1024 ** 3), 2),
            "used_gb": round(vm.used / (1024 ** 3), 2),
            "available_gb": round(vm.available / (1024 ** 3), 2),
            "percent": vm.percent,
        },
        "gpu": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda or "N/A (CPU Build)",
            "active_device": str(DEVICE).upper(),
            "device_count": len(gpus),
            "devices": gpus,
        },
        "disks": disks,
        "studio": {
            "repo_path": BASE_DIR,
            "registered_models_count": len(MODEL_REGISTRY),
            "cached_models_in_ram": len(LOADED_MODELS_CACHE),
            "template_mesh": f"Icosphere ({len(TPL_VERTS_NP)} vertices, {len(TPL_FACES_NP)} faces)",
        }
    }


@app.get("/api/models")
async def list_models():
    """Returns all registered 3D reconstruction models and preset folder paths."""
    return {
        "models": list(MODEL_REGISTRY.values()),
        "presets": {
            "session2_dir": DEFAULT_SESSION2_DIR if os.path.exists(DEFAULT_SESSION2_DIR) else "",
            "session1_dir": DEFAULT_SESSION1_DIR if os.path.exists(DEFAULT_SESSION1_DIR) else "",
        }
    }


@app.post("/api/models/upload-raw")
async def upload_raw_model_file(request: Request, filename: str = Query(...)):
    """
    Streams a raw binary .pt / .pth file directly to models/imported without requiring python-multipart.
    """
    safe_fname = os.path.basename(filename)
    if not safe_fname.lower().endswith((".pt", ".pth")):
        raise HTTPException(status_code=400, detail=f"{safe_fname}: Chỉ hỗ trợ định dạng .pt hoặc .pth")

    body_bytes = await request.body()
    dest_path = os.path.join(IMPORTED_MODELS_DIR, safe_fname)
    with open(dest_path, "wb") as f:
        f.write(body_bytes)

    entry = register_checkpoint_file(dest_path, fast_scan=False)
    if not entry:
        raise HTTPException(status_code=400, detail=f"{safe_fname}: Cấu trúc state_dict không hợp lệ")

    return {
        "added": entry,
        "all_models": list(MODEL_REGISTRY.values()),
    }


@app.post("/api/models/scan-dir")
async def scan_directory_for_models(req: ScanDirRequest):
    """Scans a local folder path for trained .pt / .pth models and registers them."""
    folder = req.folder_path.strip().strip('"').strip("'")
    if not os.path.exists(folder) or not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Thư mục không tồn tại: {folder}")

    added = []
    for root_dir, _, files in os.walk(folder):
        for fname in sorted(files):
            if fname.lower().endswith((".pt", ".pth")):
                entry = register_checkpoint_file(os.path.join(root_dir, fname), fast_scan=True)
                if entry:
                    added.append(entry)

    return {
        "scanned_folder": folder,
        "found_count": len(added),
        "models": list(MODEL_REGISTRY.values()),
    }


@app.post("/api/models/clear")
async def clear_model_registry():
    """Clears the current model list and frees cached models from RAM/VRAM."""
    MODEL_REGISTRY.clear()
    LOADED_MODELS_CACHE.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "cleared", "models": []}


@app.get("/api/sample-images")
async def get_sample_images():
    """Returns built-in sample furniture images for 1-click testing."""
    samples = []
    if os.path.exists(SAMPLES_DIR):
        for fname in sorted(os.listdir(SAMPLES_DIR)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                cat = fname.replace("sample_", "").split(".")[0].capitalize()
                samples.append({
                    "name": fname,
                    "label": f"Mẫu Pix3D: {cat}",
                    "url": f"/samples/{fname}",
                })
    return {"samples": samples}


@app.post("/api/preview-mask")
async def preview_silhouette_mask(req: PreviewMaskRequest):
    """Generates an instant 2D ROI & Silhouette Mask preview when the user drops an image or adjusts sensitivity."""
    if req.image_base64:
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
    elif req.sample_name:
        sp = os.path.join(SAMPLES_DIR, os.path.basename(req.sample_name))
        if not os.path.exists(sp):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {req.sample_name}")
        with open(sp, "rb") as f:
            img_bytes = f.read()
    else:
        raise HTTPException(status_code=400, detail="Chưa có dữ liệu ảnh để tách nền.")

    prep = preprocess_image_bytes(img_bytes, DEVICE, mask_mode=req.mask_mode, sensitivity=req.sensitivity)
    return {
        "roi_rgb_b64": prep["roi_rgb_b64"],
        "mask_b64": prep["mask_b64"],
        "orig_size": prep["orig_size"],
        "foreground_coverage_pct": prep["foreground_coverage_pct"],
    }


@app.post("/api/convert")
async def convert_image_to_3d(req: ConvertRequest):
    """
    Runs 2D-to-3D Mesh Reconstruction across all user-selected models and returns
    the 3D mesh geometry for the right-side 3D Grid + downloadable file endpoints.
    """
    selected_ids = req.model_ids
    if not selected_ids:
        raise HTTPException(status_code=400, detail="Vui lòng tích chọn ít nhất 1 mô hình để chuyển đổi.")

    if req.image_base64:
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_str)
        img_name = os.path.splitext(req.image_filename or "uploaded_image")[0]
    elif req.sample_name:
        sp = os.path.join(SAMPLES_DIR, os.path.basename(req.sample_name))
        if not os.path.exists(sp):
            raise HTTPException(status_code=404, detail=f"Không tìm thấy ảnh mẫu: {req.sample_name}")
        with open(sp, "rb") as f:
            img_bytes = f.read()
        img_name = os.path.splitext(req.sample_name)[0]
    else:
        raise HTTPException(status_code=400, detail="Vui lòng kéo-thả hoặc chọn một ảnh 2D đầu vào.")

    prep = preprocess_image_bytes(img_bytes, DEVICE, mask_mode=req.mask_mode, sensitivity=req.sensitivity)
    img_t = prep["img_tensor"]
    msk_t = prep["mask_tensor"]

    job_id = uuid.uuid4().hex[:12]
    job_models_data = {}
    results_payload = []

    t_job_start = time.time()

    for mid in selected_ids:
        if mid not in MODEL_REGISTRY:
            continue
        m_info = MODEL_REGISTRY[mid]

        # Lazy load and cache model in memory
        if mid not in LOADED_MODELS_CACHE:
            model_obj, exact_meta = load_model_from_checkpoint(m_info["path"], DEVICE)
            LOADED_MODELS_CACHE[mid] = model_obj
            m_info["backbone"] = exact_meta["backbone_name"]
            m_info["decoder"] = exact_meta["decoder_type"]
            m_info["params_m"] = exact_meta["params_m"]
        else:
            model_obj = LOADED_MODELS_CACHE[mid]

        t0 = time.perf_counter()
        with torch.no_grad():
            _, v_fine_t, _ = model_obj(img_t, msk_t, TPL_VERTS_T, TPL_ADJ_T)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        v_raw_np = v_fine_t[0].detach().cpu().numpy().astype(np.float32)
        verts_np, faces_np = refine_and_fair_3d_mesh(
            v_raw_np,
            TPL_VERTS_NP,
            TPL_FACES_NP,
            TPL_ADJ_NP,
            refine_mode=req.refine_mode,
            use_pruning=req.use_pruning,
        )

        v_normals_np, _ = compute_vertex_and_face_normals(verts_np, faces_np)

        bbox_min = verts_np.min(axis=0).round(3).tolist()
        bbox_max = verts_np.max(axis=0).round(3).tolist()
        bbox_extents = (verts_np.max(axis=0) - verts_np.min(axis=0)).round(3).tolist()

        job_models_data[mid] = {
            "verts": verts_np,
            "faces": faces_np,
            "normals": v_normals_np,
            "info": m_info,
        }

        results_payload.append({
            "job_id": job_id,
            "model_id": mid,
            "model_name": m_info["name"],
            "filename": m_info["filename"],
            "session_label": m_info["session_label"],
            "session_version": m_info["session_version"],
            "backbone": m_info["backbone"],
            "decoder": m_info["decoder"],
            "split_tag": m_info["split_tag"],
            "params_m": m_info["params_m"],
            "latency_ms": latency_ms,
            "num_vertices": int(len(verts_np)),
            "num_faces": int(len(faces_np)),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "bbox_extents": bbox_extents,
            "vertices": np.round(verts_np, 5).tolist(),
            "faces": faces_np.tolist(),
            "normals": np.round(v_normals_np, 4).tolist(),
            "downloads": {
                "obj": f"/api/download/{job_id}/{mid}/obj",
                "glb": f"/api/download/{job_id}/{mid}/glb",
                "stl": f"/api/download/{job_id}/{mid}/stl",
                "ply": f"/api/download/{job_id}/{mid}/ply",
            }
        })

    JOBS_CACHE[job_id] = {
        "created_at": time.time(),
        "image_name": img_name,
        "models": job_models_data,
    }
    if len(JOBS_CACHE) > 25:
        oldest_key = sorted(JOBS_CACHE.keys(), key=lambda k: JOBS_CACHE[k]["created_at"])[0]
        JOBS_CACHE.pop(oldest_key, None)

    return {
        "job_id": job_id,
        "image_name": img_name,
        "total_time_ms": round((time.time() - t_job_start) * 1000.0, 1),
        "preprocessing": {
            "roi_rgb_b64": prep["roi_rgb_b64"],
            "mask_b64": prep["mask_b64"],
            "orig_size": prep["orig_size"],
            "foreground_coverage_pct": prep["foreground_coverage_pct"],
        },
        "results": results_payload,
        "download_all_zip": f"/api/download-all/{job_id}",
    }


@app.get("/api/download/{job_id}/{model_id}/{fmt}")
async def download_3d_file(job_id: str, model_id: str, fmt: str):
    """Exports and downloads a reconstructed 3D model in .OBJ, .GLB, .STL, or .PLY format."""
    if job_id not in JOBS_CACHE or model_id not in JOBS_CACHE[job_id]["models"]:
        raise HTTPException(status_code=404, detail="Phiên làm việc hoặc mô hình 3D không tồn tại.")

    job = JOBS_CACHE[job_id]
    m_data = job["models"][model_id]
    verts = m_data["verts"]
    faces = m_data["faces"]
    normals = m_data["normals"]
    m_name = m_data["info"]["name"]
    img_name = job["image_name"]
    safe_base = f"{img_name}__{m_name}"

    fmt = fmt.lower().strip()
    if fmt == "obj":
        data_bytes = export_to_obj_bytes(verts, faces, normals, model_name=safe_base)
        media_type = "text/plain; charset=utf-8"
        ext = "obj"
    elif fmt == "glb":
        data_bytes = export_to_glb_bytes(verts, faces, normals, model_name=safe_base)
        media_type = "model/gltf-binary"
        ext = "glb"
    elif fmt == "stl":
        data_bytes = export_to_stl_bytes(verts, faces, model_name=safe_base)
        media_type = "application/sla"
        ext = "stl"
    elif fmt == "ply":
        data_bytes = export_to_ply_bytes(verts, faces, normals, model_name=safe_base)
        media_type = "application/octet-stream"
        ext = "ply"
    else:
        raise HTTPException(status_code=400, detail=f"Định dạng 3D không hỗ trợ: {fmt}")

    filename = f"{safe_base}.{ext}"
    return Response(
        content=data_bytes,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/download-all/{job_id}")
async def download_all_models_zip(job_id: str):
    """Packages all reconstructed 3D models from a conversion job (.OBJ + .GLB + .STL + .PLY) into a single .ZIP."""
    if job_id not in JOBS_CACHE:
        raise HTTPException(status_code=404, detail="Phiên làm việc không tồn tại.")

    job = JOBS_CACHE[job_id]
    img_name = job["image_name"]
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for mid, m_data in job["models"].items():
            verts = m_data["verts"]
            faces = m_data["faces"]
            normals = m_data["normals"]
            m_name = m_data["info"]["name"]
            sess = m_data["info"]["session_label"].replace(" ", "")
            folder_prefix = f"{sess}_{m_name}"
            safe_base = f"{img_name}__{m_name}"

            zf.writestr(f"{folder_prefix}/{safe_base}.obj", export_to_obj_bytes(verts, faces, normals, safe_base))
            zf.writestr(f"{folder_prefix}/{safe_base}.glb", export_to_glb_bytes(verts, faces, normals, safe_base))
            zf.writestr(f"{folder_prefix}/{safe_base}.stl", export_to_stl_bytes(verts, faces, safe_base))
            zf.writestr(f"{folder_prefix}/{safe_base}.ply", export_to_ply_bytes(verts, faces, normals, safe_base))

    buf.seek(0)
    zip_filename = f"3D_Models_{img_name}_{job_id}.zip"
    return Response(
        content=buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


if __name__ == "__main__":
    import uvicorn
    print("=" * 72)
    print("🚀 IMAGE-TO-3D MESH RECONSTRUCTION STUDIO SERVER")
    print(f"  • URL giao diện Web : http://localhost:8000")
    print(f"  • Thiết bị suy luận : {DEVICE} ({CPU_BRAND_NAME})")
    print(f"  • Số model nạp sẵn  : {len(MODEL_REGISTRY)}")
    print("=" * 72)
    uvicorn.run(app, host="0.0.0.0", port=8000)
