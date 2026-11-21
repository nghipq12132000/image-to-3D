"""
Automated End-to-End Verification Test for Image-to-3D Mesh Reconstruction Studio.
Tests:
  1. Auto-discovery and checkpoint inspection of real trained .pt models (Session 1 & Session 2).
  2. Real-time Server Specifications API (/api/system-specs).
  3. Multi-model 2D-to-3D mesh conversion (/api/convert).
  4. Multi-format 3D file exports (.OBJ, .GLB, .STL, .PLY, and batch .ZIP).
"""
import os
import asyncio
from app import (
    ConvertRequest,
    SAMPLES_DIR,
    get_system_specs,
    list_models,
    convert_image_to_3d,
    download_3d_file,
    download_all_models_zip,
)


async def run_verification():
    print("=== 1. TESTING SERVER SPECS TELEMETRY ===")
    specs = await get_system_specs()
    print(f"  Host: {specs['hostname']} | OS: {specs['os']['system']} {specs['os']['release']}")
    print(f"  CPU : {specs['cpu']['brand']} ({specs['cpu']['physical_cores']}C/{specs['cpu']['logical_threads']}T)")
    print(f"  RAM : {specs['ram']['used_gb']} / {specs['ram']['total_gb']} GB ({specs['ram']['percent']}%)")
    print(f"  GPU : Active={specs['gpu']['active_device']} | CUDA={specs['gpu']['cuda_available']}")

    print("\n=== 2. TESTING MODEL REGISTRY & AUTO-INSPECTOR ===")
    m_res = await list_models()
    models = m_res["models"]
    print(f"  Total registered models: {len(models)}")
    assert len(models) > 0, "Expected at least 1 registered model from Session 2!"
    for m in models[:4]:
        print(f"    - {m['name']} | {m['session_label']} | {m['backbone']} | {m['decoder']} | {m['params_m']}M")

    print("\n=== 3. TESTING MULTI-MODEL 2D-TO-3D CONVERSION ===")
    test_ids = [m["id"] for m in models[:2]]
    sample_files = [f for f in os.listdir(SAMPLES_DIR) if f.endswith(".png")]
    assert len(sample_files) > 0, "Expected sample images in assets/sample_images!"

    req = ConvertRequest(
        model_ids=test_ids,
        mask_mode="auto",
        use_pruning=False,
        sample_name=sample_files[0],
    )
    conv_res = await convert_image_to_3d(req)
    job_id = conv_res["job_id"]
    print(f"  Job ID: {job_id} | Total Time: {conv_res['total_time_ms']} ms | Models Converted: {len(conv_res['results'])}")
    assert len(conv_res["results"]) == len(test_ids)

    first_res = conv_res["results"][0]
    mid = first_res["model_id"]
    print(f"  First 3D Mesh: {first_res['model_name']} -> {first_res['num_vertices']} vertices, {first_res['num_faces']} faces, {first_res['latency_ms']} ms")

    print("\n=== 4. TESTING 3D GRAPHICS FILE EXPORTS (.OBJ, .GLB, .STL, .PLY, .ZIP) ===")
    for fmt in ["obj", "glb", "stl", "ply"]:
        resp = await download_3d_file(job_id, mid, fmt)
        assert resp.status_code == 200
        print(f"    [OK] .{fmt.upper():3s} export: {len(resp.body):,} bytes")

    zip_resp = await download_all_models_zip(job_id)
    assert zip_resp.status_code == 200
    print(f"    [OK] .ZIP batch export: {len(zip_resp.body):,} bytes")
    print("\nALL END-TO-END STUDIO VERIFICATION TESTS PASSED 100%!")


if __name__ == "__main__":
    asyncio.run(run_verification())
