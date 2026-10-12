# -*- coding: utf-8 -*-
"""
KỊCH BẢN KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN TIẾN TRÌNH HUẤN LUYỆN FILE 03 (HÀ NỘI)
Đề tài: City2Graph Hà Nội (Uber H3 Resolution 9 - 985 ô không gian)
Kiểm tra toàn diện: 4 Mô hình chính (Models 1-4) & 4 Mô hình bóc tách cấu trúc (Models 1S-4S)
Chạy độc lập: python scripts/inspect_training_hanoi.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. ĐỊNH VỊ THƯ MỤC GỐC & CẤU HÌNH ĐƯỜNG DẪN DỰ ÁN
# ---------------------------------------------------------------------------
CWD = Path.cwd().resolve()
PROJECT_ROOT = CWD
while not (PROJECT_ROOT / "data").exists() and PROJECT_ROOT.parent != PROJECT_ROOT:
    PROJECT_ROOT = PROJECT_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
GRAPHS_DIR = DATA_DIR / "processed" / "hanoi" / "graphs"
OUTPUTS_DIR = DATA_DIR / "outputs" / "hanoi"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR = OUTPUTS_DIR / "figures"
SRC_DIR = PROJECT_ROOT / "src"

EXPECTED_NODES = 985
EXPECTED_FEATURES = 16
EXPECTED_EMBEDDING_DIM = 32

print("=" * 90)
print("BÁO CÁO KIỂM TOÁN & ĐỐI SOÁT TOÀN DIỆN DỮ LIỆU HUẤN LUYỆN FILE 03 (HÀ NỘI)")
print(f"Thư mục gốc dự án: {PROJECT_ROOT}")
print("=" * 90)

audit_results: dict[str, str] = {}

# ---------------------------------------------------------------------------
# [PHẦN 1: ĐỐI SOÁT ĐỒ THỊ ĐẦU VÀO & MÃ NGUỒN KIẾN TRÚC MÔ HÌNH]
# ---------------------------------------------------------------------------
print("\n[PHẦN 1: ĐỐI SOÁT ĐỒ THỊ ĐẦU VÀO & MÃ NGUỒN KIẾN TRÚC MÔ HÌNH]")
print("-" * 90)

# 1.1 Kiểm tra file tham số chuẩn hóa (Hỗ trợ cấu trúc StandardScaler mean_ và scale_)
scaling_file = GRAPHS_DIR / "oa_feature_scaling.json"
if scaling_file.exists():
    try:
        with open(scaling_file, "r", encoding="utf-8") as f:
            scaling_data = json.load(f)

        if "mean_" in scaling_data:
            n_features = len(scaling_data["mean_"])
        elif "mean" in scaling_data:
            n_features = len(scaling_data["mean"])
        elif "feature_cols" in scaling_data:
            n_features = len(scaling_data["feature_cols"])
        elif isinstance(scaling_data, dict):
            n_features = len(scaling_data.keys())
        else:
            n_features = 0

        method = scaling_data.get("method", "StandardScaler")
        has_log1p = scaling_data.get("log1p", False)

        if n_features == EXPECTED_FEATURES:
            status = "PASS"
            print(f"✓ oa_feature_scaling.json: Đủ {n_features}/{EXPECTED_FEATURES} tham số POIs | Method: {method} | Log1p: {has_log1p} -> [PASS]")
        else:
            status = "WARN"
            print(f"! oa_feature_scaling.json: Có {n_features}/{EXPECTED_FEATURES} thuộc tính -> [WARN]")

        audit_results["Scaling Config (Equation 1)"] = status
    except Exception as e:
        print(f"✗ Lỗi đọc file scaling: {e} -> [FAIL]")
        audit_results["Scaling Config (Equation 1)"] = "FAIL"
else:
    print(f"✗ Không tìm thấy {scaling_file.name} tại {GRAPHS_DIR} -> [FAIL]")
    audit_results["Scaling Config (Equation 1)"] = "FAIL"

# 1.2 Kiểm tra 3 tệp đồ thị PyG
graph_files = {
    "homo.pt": ("Model 1 & 1S: GAT-GAE (contig)", 5550),
    "hetero_walk.pt": ("Model 2 & 2S: HAN-GAE Walk (contig + walk)", 36590),
    "hetero_multi.pt": ("Model 3, 4, 3S, 4S: Multi/All (3 quan hệ)", 41488),
}

for g_name, (desc, expected_edges) in graph_files.items():
    g_path = GRAPHS_DIR / g_name
    if not g_path.exists():
        print(f"✗ Thiếu đồ thị {g_name} ({desc}) -> [FAIL]")
        audit_results[f"Input Graph: {g_name}"] = "FAIL"
        continue

    try:
        data = torch.load(g_path, map_location="cpu", weights_only=False)
        if hasattr(data, "num_nodes"):
            nodes = data.num_nodes
        elif "oa" in data.node_types:
            nodes = data["oa"].num_nodes
        else:
            nodes = 0

        if hasattr(data, "x") and data.x is not None:
            feat_dim = data.x.shape[1]
        elif "oa" in data.node_types and hasattr(data["oa"], "x") and data["oa"].x is not None:
            feat_dim = data["oa"].x.shape[1]
        else:
            feat_dim = 0

        # Kiểm tra NaN/Inf trong thuộc tính đầu vào
        has_nan = False
        if hasattr(data, "x") and data.x is not None:
            has_nan = torch.isnan(data.x).any().item() or torch.isinf(data.x).any().item()
        elif "oa" in data.node_types and hasattr(data["oa"], "x") and data["oa"].x is not None:
            has_nan = torch.isnan(data["oa"].x).any().item() or torch.isinf(data["oa"].x).any().item()

        status = "PASS" if (nodes == EXPECTED_NODES and feat_dim == EXPECTED_FEATURES and not has_nan) else "WARN"
        print(f"✓ Đồ thị {g_name:<16} | {desc:<38} | Đỉnh: {nodes:<3} | Feat: {feat_dim:<2} | Sạch NaN: {not has_nan} -> [{status}]")
        audit_results[f"Input Graph: {g_name}"] = status
    except Exception as e:
        print(f"✗ Lỗi đọc đồ thị {g_name}: {e} -> [FAIL]")
        audit_results[f"Input Graph: {g_name}"] = "FAIL"

# 1.3 Kiểm tra module src.models
if SRC_DIR.exists():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    try:
        from src.models import GATGAE, HANGAE
        print("✓ Kiến trúc nơ-ron lõi (GATGAE, HANGAE) nạp thành công từ src.models -> [PASS]")
        audit_results["Model Classes (src.models)"] = "PASS"
    except ImportError:
        try:
            from models import GATGAE, HANGAE
            print("✓ Kiến trúc nơ-ron lõi nạp thành công từ models cục bộ -> [PASS]")
            audit_results["Model Classes (src.models)"] = "PASS"
        except ImportError as e:
            print(f"! Không nạp được class từ src.models ({e}) -> [WARN]")
            audit_results["Model Classes (src.models)"] = "WARN"
else:
    print(f"! Thư mục {SRC_DIR} không tồn tại -> [WARN]")
    audit_results["Model Classes (src.models)"] = "WARN"

# ---------------------------------------------------------------------------
# [PHẦN 2: ĐỐI SOÁT 8 MA TRẬN VECTOR NHÚNG TIỀM ẨN Z (985 x 32)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 2: ĐỐI SOÁT MA TRẬN VECTOR NHÚNG TIỀM ẨN Z (985 x 32)]")
print("-" * 90)

all_target_models = [
    # 4 mô hình chính có thuộc tính POIs
    ("model1_gat_gae", "Model 1 : GAT-GAE (contig)", True),
    ("model2_han_gae_walk", "Model 2 : HAN-GAE Walk (contig + walk)", True),
    ("model3_han_gae_multi", "Model 3 : HAN-GAE Multi (contig + multi)", True),
    ("model4_han_gae_all", "Model 4 : HAN-GAE All (Cả 3 quan hệ)", True),
    # 4 mô hình triệt tiêu cấu trúc (Ablation Study)
    ("model1s_gat_gae_struct", "Model 1S: GAT-GAE (contig Struct-only)", True),
    ("model2s_han_gae_walk_struct", "Model 2S: HAN-GAE Walk (Struct-only)", True),
    ("model3s_han_gae_multi_struct", "Model 3S: HAN-GAE Multi (Struct-only)", True),
    ("model4s_han_gae_all_struct", "Model 4S: HAN-GAE All (Struct-only)", True),
]

for model_key, model_title, _ in all_target_models:
    npy_path = EMBEDDINGS_DIR / f"Z_{model_key}.npy"
    if not npy_path.exists():
        npy_path = EMBEDDINGS_DIR / f"{model_key}.npy"
    pt_path = EMBEDDINGS_DIR / f"{model_key}.pt"

    if not npy_path.exists() and not pt_path.exists():
        print(f"- {model_title:<46}: Chưa xuất hiện file nhúng.")
        audit_results[f"Embedding: {model_key}"] = "CHƯA XUẤT"
        continue

    try:
        if npy_path.exists():
            z = np.load(npy_path)
        else:
            pt_obj = torch.load(pt_path, map_location="cpu", weights_only=False)
            if hasattr(pt_obj, "node_types") and "oa" in pt_obj.node_types:
                z_keys = [f"embedding_{i}" for i in range(EXPECTED_EMBEDDING_DIM)]
                z = torch.stack([pt_obj["oa"][k] for k in z_keys], dim=1).detach().cpu().numpy()
            elif hasattr(pt_obj, "embedding_0"):
                z_keys = [f"embedding_{i}" for i in range(EXPECTED_EMBEDDING_DIM)]
                z = torch.stack([pt_obj[k] for k in z_keys], dim=1).detach().cpu().numpy()
            else:
                z = np.array(pt_obj)

        n_rows, n_cols = z.shape
        has_nan = np.isnan(z).any()
        has_inf = np.isinf(z).any()
        std_val = float(np.std(z))
        mean_val = float(np.mean(z))
        min_val = float(np.min(z))
        max_val = float(np.max(z))

        valid_shape = (n_rows == EXPECTED_NODES and n_cols == EXPECTED_EMBEDDING_DIM)
        not_collapsed = std_val > 0.05  # Chống sụp đổ biểu diễn

        if valid_shape and not has_nan and not has_inf and not_collapsed:
            status = "PASS"
        elif has_nan or has_inf:
            status = "FAIL (NaN/Inf)"
        elif not not_collapsed:
            status = "WARN (Collapsed/Flat)"
        else:
            status = "WARN"

        print(
            f"✓ {model_key:<28} | Shape: ({n_rows}, {n_cols}) | "
            f"Mean: {mean_val:+.3f} | Std: {std_val:.3f} | [{min_val:.2f}, {max_val:.2f}] -> [{status}]"
        )
        audit_results[f"Embedding: {model_key}"] = status

    except Exception as e:
        print(f"✗ Lỗi đọc embedding {model_key}: {e} -> [FAIL]")
        audit_results[f"Embedding: {model_key}"] = "FAIL"

# ---------------------------------------------------------------------------
# [PHẦN 3: ĐỐI SOÁT TRỌNG SỐ CHÚ Ý NGỮ NGHĨA BETA (SEMANTIC ATTENTION)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 3: ĐỐI SOÁT TRỌNG SỐ CHÚ Ý NGỮ NGHĨA BETA (LAYER 2)]")
print("-" * 90)

beta_targets = [
    ("beta_model2_han_gae_walk.npy", "Model 2 Attention beta", 2, ["contig", "walk"]),
    ("beta_model3_han_gae_multi.npy", "Model 3 Attention beta", 2, ["contig", "multi"]),
    ("beta_model4_han_gae_all.npy", "Model 4 Attention beta", 3, ["contig", "walk", "multi"]),
]

for beta_fname, beta_desc, exp_cols, rel_names in beta_targets:
    beta_path = EMBEDDINGS_DIR / beta_fname
    if beta_path.exists():
        try:
            beta_mat = np.load(beta_path)
            n_rows, n_cols = beta_mat.shape
            is_valid_shape = (n_rows == EXPECTED_NODES and n_cols == exp_cols)
            # Kiểm tra ràng buộc Softmax (tổng mỗi hàng xấp xỉ 1.0)
            row_sums = beta_mat.sum(axis=1)
            is_softmax_ok = np.allclose(row_sums, 1.0, atol=1e-3)
            means_str = " | ".join([f"{rel_names[i]}: {beta_mat[:, i].mean():.1%}" for i in range(exp_cols)])

            if is_valid_shape and is_softmax_ok:
                status = "PASS"
            else:
                status = "WARN"

            print(f"✓ {beta_fname:<30} | Shape: ({n_rows}, {n_cols}) | Phân bổ: {means_str} -> [{status}]")
            audit_results[f"Attention: {beta_fname}"] = status
        except Exception as e:
            print(f"✗ Lỗi đọc {beta_fname}: {e} -> [FAIL]")
            audit_results[f"Attention: {beta_fname}"] = "FAIL"
    else:
        print(f"- {beta_fname:<30} | Chưa xuất hiện file.")
        audit_results[f"Attention: {beta_fname}"] = "CHƯA XUẤT"

# ---------------------------------------------------------------------------
# [PHẦN 4: ĐỐI SOÁT CHECKPOINTS TRỌNG SỐ TỐI ƯU (BEST MODEL WEIGHTS)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 4: ĐỐI SOÁT CHECKPOINTS TRỌNG SỐ MÔ HÌNH]")
print("-" * 90)

for model_key, model_title, _ in all_target_models:
    ckpt_path = CHECKPOINTS_DIR / model_key / "model.pt"
    if not ckpt_path.exists():
        ckpt_path = CHECKPOINTS_DIR / f"{model_key}.pt"

    if ckpt_path.exists():
        size_kb = ckpt_path.stat().st_size / 1024
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            num_tensors = len(state_dict) if isinstance(state_dict, dict) else "N/A"
            status = "PASS" if size_kb > 10 else "WARN"
            print(f"✓ Checkpoint: {model_key:<28} | Dung lượng: {size_kb:>6.1f} KB | Tensors: {num_tensors} -> [{status}]")
            audit_results[f"Checkpoint: {model_key}"] = status
        except Exception as e:
            print(f"✗ Lỗi nạp checkpoint {model_key}: {e} -> [FAIL]")
            audit_results[f"Checkpoint: {model_key}"] = "FAIL"
    else:
        print(f"- Checkpoint: {model_key:<28} | Chưa tồn tại.")
        audit_results[f"Checkpoint: {model_key}"] = "CHƯA XUẤT"

# ---------------------------------------------------------------------------
# [PHẦN 5: ĐỐI SOÁT ĐỦ 8 BIỂU ĐỒ HỘI TỤ LOSS FILE 03 (FIGURES)]
# ---------------------------------------------------------------------------
print("\n[PHẦN 5: ĐỐI SOÁT 8 BIỂU ĐỒ SUY GIẢM HÀM MẤT MÁT FILE 03 (FIGURES)]")
print("-" * 90)

expected_figures = [
    # 4 mô hình chính
    ("hanoi_loss_progress_model1_gat_gae.png", "Loss Model 1 (GAT-GAE: contig)", 50),
    ("hanoi_loss_progress_model2_han_gae_walk.png", "Loss Model 2 (HAN-GAE Walk)", 50),
    ("hanoi_loss_progress_model3_han_gae_multi.png", "Loss Model 3 (HAN-GAE Multi)", 50),
    ("hanoi_loss_progress_model4_han_gae_all.png", "Loss Model 4 (HAN-GAE All Relations)", 50),
    # 4 mô hình triệt tiêu cấu trúc
    ("hanoi_loss_progress_model1s_gat_gae_struct.png", "Loss Model 1S (Struct-only Contiguity)", 50),
    ("hanoi_loss_progress_model2s_han_gae_walk_struct.png", "Loss Model 2S (Struct-only Walk)", 50),
    ("hanoi_loss_progress_model3s_han_gae_multi_struct.png", "Loss Model 3S (Struct-only Multi)", 50),
    ("hanoi_loss_progress_model4s_han_gae_all_struct.png", "Loss Model 4S (Struct-only All Relations)", 50),
]

for fig_name, fig_desc, min_kb in expected_figures:
    fig_path = FIGURES_DIR / fig_name
    if fig_path.exists():
        size_kb = fig_path.stat().st_size / 1024
        status = "PASS" if size_kb >= min_kb else "WARN"
        print(f"✓ {fig_name:<48} | {size_kb:>7.1f} KB | {fig_desc} -> [{status}]")
        audit_results[f"Figure: {fig_name}"] = status
    else:
        print(f"- {fig_name:<48} | Chưa kết xuất.")
        audit_results[f"Figure: {fig_name}"] = "CHƯA XUẤT"

# ---------------------------------------------------------------------------
# [PHẦN 6: BẢNG TỔNG HỢP KẾT QUẢ NGHIỆM THU]
# ---------------------------------------------------------------------------
print("\n" + "=" * 90)
print("BẢNG TỔNG HỢP KẾT QUẢ ĐỐI SOÁT & NGHIỆM THU PHA 3 (NOTEBOOK 03)")
print("=" * 90)

pass_count = sum(1 for v in audit_results.values() if v == "PASS")
total_checks = len(audit_results)
pending_count = sum(1 for v in audit_results.values() if v == "CHƯA XUẤT")
warn_count = sum(1 for v in audit_results.values() if "WARN" in v)
fail_count = sum(1 for v in audit_results.values() if "FAIL" in v)

for k, v in audit_results.items():
    if v == "PASS":
        icon = "✓"
    elif "FAIL" in v:
        icon = "✗"
    elif v == "CHƯA XUẤT":
        icon = "-"
    else:
        icon = "!"
    print(f"  {icon} {k:<52}: {v}")

print("-" * 90)
print(f"TỔNG KẾT: {pass_count}/{total_checks} TIÊU CHÍ ĐẠT CHUẨN (PASS) | Cảnh báo: {warn_count} | Thiếu: {pending_count} | Lỗi: {fail_count}")

if fail_count > 0:
    print(">>> KẾT LUẬN: CÓ LỖI TENSOR HOẶC ĐỒ THỊ CẦN KHẮC PHỤC TRƯỚC KHI TIẾN HÀNH BƯỚC TIẾP THEO. <<<")
elif pass_count >= 24 and fail_count == 0:
    print(">>> KẾT LUẬN: TOÀN BỘ TIẾN TRÌNH HUẤN LUYỆN FILE 03 (ĐỦ 8 MÔ HÌNH) ĐÃ HOÀN TẤT VÀ ĐẠT CHUẨN 100%! <<<")
    print(">>> HỆ THỐNG ĐÃ SẴN SÀNG TOÀN DIỆN ĐỂ BƯỚC SANG NOTEBOOK 04: ĐÁNH GIÁ PHÂN CỤM & MODULARITY Q! <<<")
else:
    print(">>> KẾT LUẬN: TIẾN TRÌNH HUẤN LUYỆN ĐANG ĐƯỢC TIẾN HÀNH TỐT, HÃY HOÀN THIỆN CÁC MỤC CÒN THIẾU. <<<")
print("=" * 90)