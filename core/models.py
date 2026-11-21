"""
Unified PyTorch Model Architecture & Automatic Checkpoint Inspector for Image-to-3D Studio.
Seamlessly loads any trained .pt / .pth model from both Session 1 (Baseline) and Session 2 (Refined F1-Booster).
Includes Zero-Copy Fast Metadata Discovery for instant folder scanning over Google Drive / network mounts.
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def encode_3d_fourier_pos(verts):
    """9D Fourier Positional Encoding: [v, sin(pi*v), cos(pi*v)]."""
    return torch.cat([verts, torch.sin(math.pi * verts), torch.cos(math.pi * verts)], dim=-1)


def project_3d_to_2d_features(verts_3d, f_mid, f_top):
    """Samples multi-scale 2D features at projected 2D vertex coordinates via bilinear grid_sample."""
    grid_xy = torch.clamp(verts_3d[:, :, :2], -0.98, 0.98).unsqueeze(2)
    s_mid = F.grid_sample(f_mid, grid_xy, mode="bilinear", padding_mode="border", align_corners=True).squeeze(-1).permute(0, 2, 1)
    s_top = F.grid_sample(f_top, grid_xy, mode="bilinear", padding_mode="border", align_corners=True).squeeze(-1).permute(0, 2, 1)
    return torch.cat([s_mid, s_top], dim=-1)


class MultiScale2DEncoder(nn.Module):
    """
    Supports both Session 1 (global_fc input = c_top) and Session 2 (global_fc input = c_top + c_mid)
    across 4 Backbones: MobileNetV3-Large, EfficientNet-B0, ResNet-18, ConvNeXt-Tiny.
    """
    def __init__(self, backbone_name="MobileNetV3-Large", feat_dim=160, use_mask_gate=True, session_version="session2", pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.use_mask_gate = use_mask_gate
        self.session_version = session_version
        weights_arg = "DEFAULT" if pretrained else None

        if backbone_name == "MobileNetV3-Large":
            net = tv_models.mobilenet_v3_large(weights=weights_arg)
            self.stage_mid = net.features[:11]   # 14x14, 80ch
            self.stage_top = net.features[11:]   # 7x7, 960ch
            c_mid, c_top = 80, 960
        elif backbone_name == "EfficientNet-B0":
            net = tv_models.efficientnet_b0(weights=weights_arg)
            self.stage_mid = net.features[:6]    # 14x14, 112ch
            self.stage_top = net.features[6:]    # 7x7, 1280ch
            c_mid, c_top = 112, 1280
        elif backbone_name == "ResNet-18":
            net = tv_models.resnet18(weights=weights_arg)
            self.stage_mid = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2, net.layer3)
            self.stage_top = net.layer4          # 7x7, 512ch
            c_mid, c_top = 256, 512
        elif backbone_name == "ConvNeXt-Tiny":
            net = tv_models.convnext_tiny(weights=weights_arg)
            self.stage_mid = net.features[:6]    # 14x14, 384ch
            self.stage_top = net.features[6:]    # 7x7, 768ch
            c_mid, c_top = 384, 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.proj_mid = nn.Sequential(nn.Conv2d(c_mid, feat_dim // 2, 1, bias=False), nn.BatchNorm2d(feat_dim // 2), nn.SiLU())
        self.proj_top = nn.Sequential(nn.Conv2d(c_top, feat_dim // 2, 1, bias=False), nn.BatchNorm2d(feat_dim // 2), nn.SiLU())

        global_in_dim = (c_top + c_mid) if session_version == "session2" else c_top
        self.global_fc = nn.Sequential(
            nn.Linear(global_in_dim, feat_dim),
            nn.LayerNorm(feat_dim),
            nn.SiLU()
        )
        init_alpha = 1.10 if session_version == "session2" else 0.75
        self.mask_alpha = nn.Parameter(torch.tensor(init_alpha))

    def forward(self, img, mask=None):
        f_mid_raw = self.stage_mid(img)
        f_top_raw = self.stage_top(f_mid_raw)

        f_mid = self.proj_mid(f_mid_raw)
        f_top = self.proj_top(f_top_raw)

        if self.session_version == "session2":
            if self.use_mask_gate and mask is not None:
                alpha = torch.sigmoid(self.mask_alpha)
                m_mid = F.interpolate(mask, size=f_mid.shape[-2:], mode="bilinear", align_corners=False)
                m_top = F.interpolate(mask, size=f_top.shape[-2:], mode="bilinear", align_corners=False)
                f_mid = f_mid * (alpha * m_mid + (1.0 - alpha * 0.85))
                f_top = f_top * (alpha * m_top + (1.0 - alpha * 0.85))
                m_mid_raw = F.interpolate(mask, size=f_mid_raw.shape[-2:], mode="bilinear", align_corners=False)
                g_mid = (f_mid_raw * (0.7 * m_mid_raw + 0.3)).mean(dim=(-2, -1))
            else:
                g_mid = F.adaptive_avg_pool2d(f_mid_raw, 1).flatten(1)
            g_top = F.adaptive_avg_pool2d(f_top_raw, 1).flatten(1)
            g_vec = self.global_fc(torch.cat([g_top, g_mid], dim=-1))
        else:
            g_vec = self.global_fc(F.adaptive_avg_pool2d(f_top_raw, 1).flatten(1))
            if self.use_mask_gate and mask is not None:
                alpha = torch.sigmoid(self.mask_alpha)
                m_mid = F.interpolate(mask, size=f_mid.shape[-2:], mode="bilinear", align_corners=False)
                m_top = F.interpolate(mask, size=f_top.shape[-2:], mode="bilinear", align_corners=False)
                f_mid = f_mid * (alpha * m_mid + (1.0 - alpha))
                f_top = f_top * (alpha * m_top + (1.0 - alpha))

        return f_mid, f_top, g_vec


class IsotropicGCNBlock(nn.Module):
    """Standard Isotropic Graph Convolution Layer (Pixel2Mesh-Lite & Vox2Mesh-Hybrid)."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc_self  = nn.Linear(in_dim, hidden_dim)
        self.fc_neigh = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.SiLU()

    def forward(self, h, adj_norm):
        neigh_agg = torch.matmul(adj_norm.unsqueeze(0), h)
        out = self.fc_self(h) + self.fc_neigh(neigh_agg)
        return self.act(self.norm(out))


class AnisotropicGATv2BlockV1(nn.Module):
    """Session 1 Anisotropic GATv2 Block."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh = nn.Linear(in_dim, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(in_dim * 2 + 3, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.SiLU()

    def forward(self, h, verts, adj_norm):
        neigh_mean = torch.matmul(adj_norm.unsqueeze(0), h)
        v_neigh_mean = torch.matmul(adj_norm.unsqueeze(0), verts)
        delta_v = v_neigh_mean - verts
        gate_in = torch.cat([h, neigh_mean, delta_v], dim=-1)
        aniso_weight = self.edge_gate(gate_in)
        out = self.lin_self(h) + aniso_weight * self.lin_neigh(neigh_mean)
        return self.act(self.norm(out))


class AnisotropicGATv2BlockV2(nn.Module):
    """Session 2 Residual 3D Edge-Aware Anisotropic GATv2 Block."""
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.lin_self  = nn.Linear(in_dim, hidden_dim)
        self.lin_neigh = nn.Linear(in_dim, hidden_dim)
        self.res_proj  = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim, bias=False)
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.act  = nn.SiLU()

    def forward(self, h, verts, adj_norm):
        h_self = self.lin_self(h)
        h_neigh_raw = self.lin_neigh(h)
        neigh_mean = torch.matmul(adj_norm.unsqueeze(0), h_neigh_raw)

        v_neigh_mean = torch.matmul(adj_norm.unsqueeze(0), verts)
        delta_v = v_neigh_mean - verts
        dist_v  = torch.norm(delta_v, dim=-1, keepdim=True)

        gate_in = torch.cat([h_self, neigh_mean, delta_v, dist_v], dim=-1)
        aniso_weight = torch.sigmoid(self.edge_gate(gate_in) + 1.0)
        out = h_self + aniso_weight * neigh_mean + 0.35 * self.res_proj(h)
        return self.act(self.norm(out))


class SingleView3DMeshReconstructor(nn.Module):
    """
    Unified 3D Mesh Reconstructor supporting all 4 Decoders, 4 Backbones, and both Session 1 & Session 2 checkpoints.
    """
    def __init__(self, backbone_name="MobileNetV3-Large", decoder_type="GeoAttMesh-Lite",
                 feat_dim=160, use_mask_gate=True, session_version="session2", pretrained=False):
        super().__init__()
        self.backbone_name = backbone_name
        self.decoder_type = decoder_type
        self.session_version = session_version
        effective_mask_gate = use_mask_gate if decoder_type == "GeoAttMesh-Lite" else False

        self.encoder = MultiScale2DEncoder(
            backbone_name=backbone_name,
            feat_dim=feat_dim,
            use_mask_gate=effective_mask_gate,
            session_version=session_version,
            pretrained=pretrained
        )

        in_node_dim = 3 + feat_dim * 2

        if decoder_type == "LatentMLP-Deform":
            self.mlp_deform = nn.Sequential(
                nn.Linear(3 + feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.SiLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim),
                nn.SiLU(),
                nn.Linear(feat_dim, 3),
                nn.Tanh()
            )
        elif decoder_type == "Pixel2Mesh-Lite":
            self.gcn1 = IsotropicGCNBlock(in_node_dim, feat_dim)
            self.gcn2 = IsotropicGCNBlock(feat_dim, feat_dim)
            self.offset1 = nn.Linear(feat_dim, 3)
            self.gcn3 = IsotropicGCNBlock(in_node_dim, feat_dim)
            self.gcn4 = IsotropicGCNBlock(feat_dim, feat_dim)
            self.offset2 = nn.Linear(feat_dim, 3)
        elif decoder_type == "Vox2Mesh-Hybrid":
            self.voxel_head = nn.Sequential(
                nn.Linear(feat_dim, 512),
                nn.SiLU(),
                nn.Linear(512, 24 * 24 * 24)
            )
            self.gcn1 = IsotropicGCNBlock(in_node_dim + 1, feat_dim)
            self.gcn2 = IsotropicGCNBlock(feat_dim, feat_dim)
            self.offset1 = nn.Linear(feat_dim, 3)
        elif decoder_type == "GeoAttMesh-Lite":
            if session_version == "session2":
                self.ellipsoid_head = nn.Sequential(
                    nn.Linear(feat_dim, 64),
                    nn.SiLU(),
                    nn.Linear(64, 6)
                )
                gat_in_dim = 9 + feat_dim * 2
                self.gat1 = AnisotropicGATv2BlockV2(gat_in_dim, feat_dim)
                self.gat2 = AnisotropicGATv2BlockV2(feat_dim, feat_dim)
                self.offset1 = nn.Linear(feat_dim, 3)

                self.gat3 = AnisotropicGATv2BlockV2(gat_in_dim + feat_dim, feat_dim)
                self.gat4 = AnisotropicGATv2BlockV2(feat_dim, feat_dim)
                self.offset2 = nn.Linear(feat_dim, 3)
            else:
                self.gat1 = AnisotropicGATv2BlockV1(in_node_dim, feat_dim)
                self.gat2 = AnisotropicGATv2BlockV1(feat_dim, feat_dim)
                self.offset1 = nn.Linear(feat_dim, 3)

                self.gat3 = AnisotropicGATv2BlockV1(in_node_dim, feat_dim)
                self.gat4 = AnisotropicGATv2BlockV1(feat_dim, feat_dim)
                self.offset2 = nn.Linear(feat_dim, 3)
        else:
            raise ValueError(f"Unsupported decoder_type: {decoder_type}")

    def forward(self, img, mask, tpl_verts, adj_norm):
        B = img.shape[0]
        Nv = tpl_verts.shape[0]
        f_mid, f_top, g_vec = self.encoder(img, mask)
        v0 = tpl_verts.unsqueeze(0).expand(B, Nv, 3)
        g_exp = g_vec.unsqueeze(1).expand(B, Nv, -1)
        voxel_logits = None

        if self.decoder_type == "LatentMLP-Deform":
            v_coarse = v0 + 0.65 * self.mlp_deform(torch.cat([v0, g_exp], dim=-1))
            v_fine = v_coarse

        elif self.decoder_type == "Pixel2Mesh-Lite":
            p1 = project_3d_to_2d_features(v0, f_mid, f_top)
            h1 = self.gcn2(self.gcn1(torch.cat([v0, p1, g_exp], dim=-1), adj_norm), adj_norm)
            v_coarse = v0 + 0.45 * torch.tanh(self.offset1(h1))

            p2 = project_3d_to_2d_features(v_coarse, f_mid, f_top)
            h2 = self.gcn4(self.gcn3(torch.cat([v_coarse, p2, g_exp], dim=-1), adj_norm), adj_norm)
            v_fine = v_coarse + 0.25 * torch.tanh(self.offset2(h2))

        elif self.decoder_type == "Vox2Mesh-Hybrid":
            voxel_logits = self.voxel_head(g_vec).view(B, 1, 24, 24, 24)
            vox_prob = torch.sigmoid(voxel_logits)
            grid_3d = torch.clamp(v0, -0.98, 0.98).view(B, Nv, 1, 1, 3)
            occ_feat = F.grid_sample(vox_prob, grid_3d, mode="bilinear", padding_mode="border", align_corners=True).view(B, Nv, 1)
            p1 = project_3d_to_2d_features(v0, f_mid, f_top)
            h1 = self.gcn2(self.gcn1(torch.cat([v0, p1, g_exp, occ_feat], dim=-1), adj_norm), adj_norm)
            v_coarse = v0 + 0.55 * torch.tanh(self.offset1(h1))
            v_fine = v_coarse

        elif self.decoder_type == "GeoAttMesh-Lite":
            if self.session_version == "session2":
                elip_params = self.ellipsoid_head(g_vec).unsqueeze(1)
                scale_xyz = 1.0 + 0.45 * torch.tanh(elip_params[:, :, :3])
                trans_xyz = 0.15 * torch.tanh(elip_params[:, :, 3:])
                v_init = v0 * scale_xyz + trans_xyz

                p1 = project_3d_to_2d_features(v_init, f_mid, f_top)
                pos1 = encode_3d_fourier_pos(v_init)
                h1 = self.gat2(self.gat1(torch.cat([pos1, p1, g_exp], dim=-1), v_init, adj_norm), v_init, adj_norm)
                v_coarse = v_init + 0.48 * torch.tanh(self.offset1(h1))

                p2 = project_3d_to_2d_features(v_coarse, f_mid, f_top)
                pos2 = encode_3d_fourier_pos(v_coarse)
                h2 = self.gat4(self.gat3(torch.cat([pos2, p2, g_exp, h1], dim=-1), v_coarse, adj_norm), v_coarse, adj_norm)
                v_fine = v_coarse + 0.30 * torch.tanh(self.offset2(h2))
            else:
                p1 = project_3d_to_2d_features(v0, f_mid, f_top)
                h1 = self.gat2(self.gat1(torch.cat([v0, p1, g_exp], dim=-1), v0, adj_norm), v0, adj_norm)
                v_coarse = v0 + 0.45 * torch.tanh(self.offset1(h1))

                p2 = project_3d_to_2d_features(v_coarse, f_mid, f_top)
                h2 = self.gat4(self.gat3(torch.cat([v_coarse, p2, g_exp], dim=-1), v_coarse, adj_norm), v_coarse, adj_norm)
                v_fine = v_coarse + 0.25 * torch.tanh(self.offset2(h2))

        return v_coarse, v_fine, voxel_logits


def _parse_state_dict_signature(state_dict, pt_path):
    """Extracts exact architecture metadata from an in-memory state_dict."""
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    c_mid = int(state_dict["encoder.proj_mid.0.weight"].shape[1])
    c_top = int(state_dict["encoder.proj_top.0.weight"].shape[1])
    feat_dim = int(state_dict["encoder.global_fc.0.weight"].shape[0])
    global_in_dim = int(state_dict["encoder.global_fc.0.weight"].shape[1])

    if c_mid == 80 and c_top == 960:
        backbone_name = "MobileNetV3-Large"
    elif c_mid == 112 and c_top == 1280:
        backbone_name = "EfficientNet-B0"
    elif c_mid == 256 and c_top == 512:
        backbone_name = "ResNet-18"
    elif c_mid == 384 and c_top == 768:
        backbone_name = "ConvNeXt-Tiny"
    else:
        raise ValueError(f"Unrecognized encoder channel signature: c_mid={c_mid}, c_top={c_top}")

    session_version = "session2" if (global_in_dim == (c_top + c_mid) or "ellipsoid_head.0.weight" in state_dict) else "session1"

    if "mlp_deform.0.weight" in state_dict:
        decoder_type = "LatentMLP-Deform"
    elif "voxel_head.0.weight" in state_dict:
        decoder_type = "Vox2Mesh-Hybrid"
    elif "gcn1.fc_self.weight" in state_dict:
        decoder_type = "Pixel2Mesh-Lite"
    elif "gat1.lin_self.weight" in state_dict:
        decoder_type = "GeoAttMesh-Lite"
    else:
        raise ValueError("Unrecognized decoder signature in state_dict.")

    total_params = sum(int(v.numel()) for v in state_dict.values() if isinstance(v, torch.Tensor))
    params_m = round(total_params / 1e6, 2)

    fname = os.path.basename(pt_path)
    split_tag = "S2" if "_S2_" in fname or fname.endswith("_S2.pt") else ("S1" if "_S1_" in fname or fname.endswith("_S1.pt") else "Custom")

    norm_path = pt_path.replace("\\", "/").lower()
    if "/session1/" in norm_path:
        folder_session = "Session 1"
    elif "/session2/" in norm_path:
        folder_session = "Session 2"
    else:
        folder_session = "Session 2" if session_version == "session2" else "Session 1"

    return {
        "backbone_name": backbone_name,
        "decoder_type": decoder_type,
        "session_version": session_version,
        "session_label": folder_session,
        "feat_dim": feat_dim,
        "split_tag": split_tag,
        "params_m": params_m,
        "file_size_mb": round(os.path.getsize(pt_path) / (1024 * 1024), 2),
    }


# Fast lookup table for standard Pix3D experiment filenames (avoids reading 1.1GB over Google Drive on startup)
KNOWN_MODEL_PREFIXES = {
    "M1_LatentMLP_MBV3":      ("MobileNetV3-Large", "LatentMLP-Deform", 3.28),
    "M2_Pixel2Mesh_MBV3":     ("MobileNetV3-Large", "Pixel2Mesh-Lite",  3.53),
    "M3_Vox2Mesh_MBV3":       ("MobileNetV3-Large", "Vox2Mesh-Hybrid",  10.55),
    "M4_GeoAttMesh_MBV3":     ("MobileNetV3-Large", "GeoAttMesh-Lite",  4.04),
    "M5_GeoAttMesh_EffB0":    ("EfficientNet-B0",   "GeoAttMesh-Lite",  5.16),
    "M6_GeoAttMesh_Res18":    ("ResNet-18",         "GeoAttMesh-Lite",  12.18),
    "M7_GeoAttMesh_ConvNeXt": ("ConvNeXt-Tiny",     "GeoAttMesh-Lite",  28.92),
    "A0_Base_IsotropicGCN":   ("ResNet-18",         "Pixel2Mesh-Lite",  11.67),
    "A1_Plus_MaskGate_GATv2": ("ResNet-18",         "GeoAttMesh-Lite",  12.18),
    "A2_Plus_SymmetryRule":   ("ResNet-18",         "GeoAttMesh-Lite",  12.18),
    "A3_Plus_PlanarRule":     ("ResNet-18",         "GeoAttMesh-Lite",  12.18),
}


def inspect_checkpoint_metadata(pt_path, fast_scan=True):
    """
    Fast-path metadata discovery (<0.1ms per file for known naming patterns,
    with automatic fallback to full state_dict inspection for custom-named .pt files).
    """
    fname = os.path.basename(pt_path)
    norm_path = pt_path.replace("\\", "/").lower()

    if fast_scan:
        for prefix, (bb, dec, p_m) in KNOWN_MODEL_PREFIXES.items():
            if fname.startswith(prefix):
                is_s1_folder = "/session1/" in norm_path
                session_version = "session1" if is_s1_folder else "session2"
                folder_session = "Session 1" if is_s1_folder else "Session 2"
                split_tag = "S2" if "_S2" in fname else ("S1" if "_S1" in fname else "S1")
                fsize_mb = round(os.path.getsize(pt_path) / (1024 * 1024), 2) if os.path.exists(pt_path) else 0.0
                return {
                    "backbone_name": bb,
                    "decoder_type": dec,
                    "session_version": session_version,
                    "session_label": folder_session,
                    "feat_dim": 160,
                    "split_tag": split_tag,
                    "params_m": p_m,
                    "file_size_mb": fsize_mb,
                }

    state_dict = torch.load(pt_path, map_location="cpu")
    return _parse_state_dict_signature(state_dict, pt_path)


def load_model_from_checkpoint(pt_path, device):
    """
    Loads the actual state_dict once and verifies its exact tensor signature before instantiating the model.
    Guarantees 100% compatibility regardless of filename or session folder.
    """
    state_dict = torch.load(pt_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    meta = _parse_state_dict_signature(state_dict, pt_path)
    use_mask_gate = True if meta["decoder_type"] == "GeoAttMesh-Lite" else False

    model = SingleView3DMeshReconstructor(
        backbone_name=meta["backbone_name"],
        decoder_type=meta["decoder_type"],
        feat_dim=meta["feat_dim"],
        use_mask_gate=use_mask_gate,
        session_version=meta["session_version"],
        pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, meta
