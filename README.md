# Image-to-3D Mesh Reconstruction Studio (`GeoAttMesh-Lite`)

Ứng dụng giao diện trực quan chuyển đổi **ảnh đơn 2D sang mô hình lưới tam giác 3D (Single-View 3D Mesh Reconstruction)** sử dụng kiến trúc **`GeoAttMesh-Lite`** (*Mask-Guided Multi-Scale Feature Gate + Anisotropic Ellipsoid Alignment + Residual 3D Edge-Aware GATv2*) cùng các mô hình đối sánh (`LatentMLP-Deform`, `Pixel2Mesh-Lite`, `Vox2Mesh-Hybrid`) trên 4 Backbone (`MobileNetV3-Large`, `EfficientNet-B0`, `ResNet-18`, `ConvNeXt-Tiny`).

---

## 1. Tính Năng Chính

* **Cột Trái (Control & Input Sidebar):**
  1. **Import Trained Models (`.pt` / `.pth`):** Hỗ trợ chọn/kéo-thả nhiều file `.pt` từ máy tính hoặc quét nguyên thư mục `saved_models` từ `Session 1` và `Session 2`. Hệ thống tự động đọc `state_dict` để nhận diện chính xác phiên bản kiến trúc, Backbone, Decoder và số lượng tham số.
  2. **Drag & Drop 2D Image:** Kéo-thả ảnh 2D đầu vào (`.png`, `.jpg`, `.webp`) hoặc chọn nhanh ảnh mẫu Pix3D có sẵn, tự động trích xuất mặt nạ vật thể (Silhouette Mask) bằng GrabCut/Otsu và chuẩn hóa vùng ROI $224 \times 224$.
  3. **Multi-Model Checklist:** Tích chọn đồng thời nhiều model để so sánh trực quan kết quả tái tạo 3D trên cùng một bức ảnh.
  4. **Nút `"Load 3D Model"`:** Chạy suy luận tạo lưới 3D song song cho tất cả các model đã chọn.
* **Cột Phải (Interactive 3D Grid View):**
  * Hiển thị dạng **Grid** — mỗi ô là một khung nhìn 3D WebGL (Three.js) biểu diễn mô hình 3D tương ứng với một model đã chọn, kèm thông số tốc độ suy luận (`ms`), số đỉnh (`642`), số mặt tam giác (`1.280`).
* **Dialog Xem Chi Tiết Mô Hình 3D & Xuất File Đồ Họa:**
  * Bấm vào bất kỳ ô nào trên Grid sẽ mở **Dialog 3D phóng lớn** hỗ trợ xoay $360^\circ$, thu phóng, đổi chế độ hiển thị (*Studio Clay*, *Pháp tuyến 3D*, *Metallic CAD*, *Wireframe*, *Đám mây đỉnh*).
  * Hỗ trợ tải xuống **4 định dạng 3D tiêu chuẩn** tương thích với mọi công cụ đồ họa phổ biến:
    * **`.OBJ` (Wavefront OBJ kèm pháp tuyến đỉnh):** *Blender, Autodesk Maya, 3ds Max, Cinema 4D, ZBrush*.
    * **`.GLB` (Binary glTF 2.0 chuẩn Khronos):** *Blender, Unity, Unreal Engine, Godot, Three.js, Windows 3D Viewer*.
    * **`.STL` (Binary STL):** *In 3D (Cura, PrusaSlicer, Bambu Studio), SolidWorks, AutoCAD, Fusion 360*.
    * **`.PLY` (Stanford PLY kèm màu đỉnh & pháp tuyến):** *MeshLab, CloudCompare, Open3D*.
* **Dialog Cấu Hình Máy Server (`🖥️ Cấu hình Server`):**
  * Hiển thị thông số phần cứng máy chủ theo thời gian thực: CPU (Tên chip, số nhân/luồng, xung nhịp, % tải), RAM (Tổng/Đang dùng/Khả dụng), GPU/CUDA (Tên GPU, VRAM, phiên bản PyTorch) và Dung lượng các ổ đĩa.

---

## 2. Cách Khởi Chạy

### Cách 1: Chạy nhanh bằng file `.bat` (Windows)
Nhấp đúp vào file **`run_studio.bat`** hoặc chạy lệnh:
```powershell
cd D:\project\AI-repo\image-to-3D
python app.py
```
Sau đó mở trình duyệt truy cập: **`http://localhost:8000`**

---

## 3. Cấu Trúc Thư Mục

```text
D:\project\AI-repo\image-to-3D\
├── app.py                      # FastAPI Backend Server & API Endpoints
├── run_studio.bat              # Script khởi chạy nhanh trên Windows
├── requirements.txt            # Danh sách thư viện Python
├── test_studio_pipeline.py     # Script kiểm thử tự động toàn bộ Pipeline & API
├── core/
│   ├── icosphere_template.py   # Bộ nạp & khởi tạo lưới mẫu Icosphere (642 đỉnh, 1280 mặt)
│   ├── models.py               # Kiến trúc PyTorch & Bộ tự động nhận diện Checkpoint (Session 1 & 2)
│   ├── preprocess.py           # Tiền xử lý ảnh 2D ROI 224x224 & Tách nền Silhouette tự động
│   └── exporter.py             # Bộ xuất file 3D đa định dạng (.OBJ, .GLB, .STL, .PLY)
├── static/
│   └── index.html              # Giao diện Web Studio 2 cột + Three.js 3D Grid & Modals
├── assets/
│   ├── icosphere_template.npz  # Lưới cầu mẫu chuẩn
│   └── sample_images/          # Ảnh mẫu nội thất Pix3D để kiểm thử 1-click
├── models/
│   └── imported/               # Thư mục lưu các file .pt người dùng upload trực tiếp
└── exports/                    # Thư mục tạm xuất file 3D
```
