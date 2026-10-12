# -*- coding: utf-8 -*-
"""
CHƯƠNG TRÌNH HUẤN LUYỆN MẠNG NƠ-RON ĐỒ THỊ TỰ GIÁM SÁT HÀ NỘI (ĐẦY ĐỦ 26 CELLS: PHẦN I, II & III)
Kế thừa & Chuyển đổi 1:1 từ: 03_model_training.py (Sato et al., CEUS 2026)
Đối tượng không gian: 985 ô lục giác Uber H3 Resolution 9 (16 đặc trưng POIs tầng trệt)
Kiến trúc mô hình: GATGAE (Đồng nhất) và HANGAE (Dị thể đa quan hệ)
Môi trường thực thi: VS Code Interactive Window (# %%) hoặc python dòng lệnh
"""

from __future__ import annotations

# %% [Cell 1]: Thiết lập môi trường thực thi và tự động định vị thư mục gốc
# --- Ghi chú giải thích từ Notebook gốc ---
# # 03. Model Implementation & Training (Hanoi Case Study)
#
# Giai đoạn huấn luyện mô hình học biểu diễn không gian tự giám sát (Section 4.4 & 4.5).
# Mục tiêu: Học các vector nhúng tiềm ẩn z in R^{985 x 32} bảo toàn thuộc tính 16 POIs
# và cấu trúc topo đa quan hệ (contig, 15_min_walk, 15_min_multi).
# -----------------------------------------

import os
import sys
import copy
import random
import warnings
from pathlib import Path

# Cấu hình OpenMP an toàn trên Windows chống xung đột runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
warnings.filterwarnings("ignore")

# Cơ chế Dynamic Root Discovery tự động định vị gốc repository City2Graph-Hanoi
CWD = Path.cwd().resolve()
REPO_ROOT = CWD
while not (REPO_ROOT / "data").exists() and REPO_ROOT.parent != REPO_ROOT:
    REPO_ROOT = REPO_ROOT.parent

# Đưa thư mục mã nguồn lõi src/ vào sys.path
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("=" * 85)
print(f"✓ [Cell 1] Đã định vị thư mục gốc dự án: {REPO_ROOT}")
print(f"✓ [Cell 1] Đã kết nối module mã nguồn lõi: {SRC_DIR}")
print("=" * 85)


# %% [Cell 2]: Khai báo thư viện, module học sâu và kiểm tra tính sẵn sàng của GPU/CPU
# --- Ghi chú giải thích từ Notebook gốc ---
# Import dependencies, PyTorch, PyG and Core Graph Neural Network Architectures
# -----------------------------------------

import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Nạp hai lớp kiến trúc mạng lõi từ src.models
try:
    from src.models import GATGAE, HANGAE
    print("✓ [Cell 2] Nạp thành công các lớp mô hình: GATGAE, HANGAE từ src.models.")
except ImportError as e:
    # Dự phòng nếu nạp trực tiếp khi đang đứng trong thư mục src
    try:
        from models import GATGAE, HANGAE
        print("✓ [Cell 2] Nạp thành công GATGAE, HANGAE từ models cục bộ.")
    except ImportError:
        raise ImportError(f"Không thể nạp GATGAE/HANGAE từ {SRC_DIR}. Chi tiết lỗi: {e}")

# Thiết lập đường dẫn dữ liệu chuẩn hóa của Hà Nội
DATA_DIR = REPO_ROOT / "data"
GRAPH_DIR = DATA_DIR / "processed" / "hanoi" / "graphs"
OUTPUT_DIR = DATA_DIR / "outputs" / "hanoi"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
EMBEDDING_DIR = OUTPUT_DIR / "embeddings"
FIGURE_DIR = OUTPUT_DIR / "figures"
CLUSTER_DIR = OUTPUT_DIR / "clusters"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ [Cell 2] Thư mục nạp đồ thị: {GRAPH_DIR}")
print(f"✓ [Cell 2] Thư mục lưu embeddings & checkpoints: {OUTPUT_DIR}")


# %% [Cell 3]: Tải tệp cấu hình, thiết lập tính xác định (Determinism) & Thiết bị
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 1. Configuration & Setup
# We load hyperparameters from experiment_config.yaml. This ensures reproducibility
# and easy tuning of parameters like learning rate, hidden dimensions, and structure loss weights.
# -----------------------------------------

def load_config(config_path: Path | str) -> dict:
    """Tải cấu hình siêu tham số và tự động bổ sung fallback nếu thiếu khóa."""
    path_obj = Path(config_path)
    if not path_obj.exists():
        fallback_cfg = REPO_ROOT / "configs" / "experiment_config.yaml"
        if fallback_cfg.exists():
            path_obj = fallback_cfg
        else:
            print(f"! Không tìm thấy {config_path}, áp dụng cấu hình mặc định chuẩn CEUS 2026 cho Hà Nội.")
            return {
                "seeds": {"global": 42, "model": 42},
                "experiment": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
                "model": {
                    "in_dim": 16,        # Chuẩn hóa 16 nhóm POIs tầng trệt của Hà Nội
                    "hidden_dim": 64,
                    "out_dim": 32,       # Chiều vector nhúng tiềm ẩn z
                    "heads": 4,
                    "dropout": 0.6,
                    "lambda_struct": 1.0
                },
                "training": {
                    "lr": 0.005,
                    "epochs": 200,
                    "early_stopping_patience": 100,
                    "neg_sampling_ratio": 1.0
                },
                "data": {
                    "root": str(GRAPH_DIR),
                    "homo": "homo.pt",
                    "hetero_walk": "hetero_walk.pt",
                    "hetero_multi": "hetero_multi.pt"
                }
            }

    with open(path_obj, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "data" not in cfg:
        cfg["data"] = {}
    cfg["data"]["root"] = str(GRAPH_DIR)
    cfg["data"]["homo"] = "homo.pt"
    cfg["data"]["hetero_walk"] = "hetero_walk.pt"
    cfg["data"]["hetero_multi"] = "hetero_multi.pt"
    cfg["model"]["in_dim"] = 16  # Chuẩn hóa 16 đặc trưng POIs Hà Nội
    return cfg

config_candidates = [
    REPO_ROOT / "configs" / "experiment_config.yaml",
    REPO_ROOT / "experiment_config.yaml",
    CWD / "experiment_config.yaml"
]
chosen_cfg_path = next((p for p in config_candidates if p.exists()), REPO_ROOT / "configs" / "experiment_config.yaml")
config = load_config(chosen_cfg_path)

def configure_determinism(seed: int) -> None:
    """Cố định toàn diện hạt giống ngẫu nhiên trên CPU và CUDA để tái lập kết quả 100%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    print(f"✓ Cố định hạt giống ngẫu nhiên (Random seed): {seed}")

global_seed = config["seeds"].get("global", 42)
configure_determinism(global_seed)

device_pref = config["experiment"].get("device", "cpu")
device = torch.device(device_pref if torch.cuda.is_available() and device_pref == "cuda" else "cpu")
print(f"✓ Thiết bị tính toán được chỉ định: {device}")


# %% [Cell 4]: Hàm huấn luyện tổng quát train_model (Joint Loss & Early Stopping)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 2. Training Utility
# This general training loop handles:
# 1. Optimization: Updating model weights using Adam.
# 2. Loss Calculation: Combining Feature Reconstruction Loss (Smooth L1) and Structure Loss (DistMult).
# 3. Early Stopping: Preventing overfitting by monitoring joint loss improvement.
# 4. Embedding Generation: Extracting the latent z vectors from the best model state.
# -----------------------------------------

def train_model(
    model: torch.nn.Module,
    data: any,
    config: dict,
    model_name: str,
    save_artifacts: bool = True,
    epochs_override: int | None = None
) -> tuple[any, list[float]]:
    """
    Vòng lặp huấn luyện tự giám sát thống nhất cho cả GATGAE và HANGAE.
    
    Hàm mất mát liên hợp (Joint Loss - Equation 14):
        L_total = L_feat + (1 / |M|) * sum_{m in M} L_struct^{(m)}
    """
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("Thuật toán xác định chưa được kích hoạt. Hãy gọi configure_determinism(seed) trước.")

    print(f"\n-> Bắt đầu huấn luyện mô hình: {model_name}...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["lr"],
        weight_decay=config["training"].get("weight_decay", 1e-4)
    )
    
    best_loss = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    patience = config["training"]["early_stopping_patience"]
    patience_counter = 0
    epochs = config["training"]["epochs"] if epochs_override is None else epochs_override

    checkpoint_path = None
    if save_artifacts:
        ckpt_dir = CHECKPOINT_DIR / model_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / "model.pt"

    losses: list[float] = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Tính hàm mất mát liên hợp (Joint Loss) từ kiến trúc src.models
        loss, l_feat, l_struct = model.compute_loss(
            data,
            lambda_struct=config["model"]["lambda_struct"],
            neg_sampling_ratio=config["training"]["neg_sampling_ratio"]
        )
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        losses.append(current_loss)

        # In nhật ký tiến trình mỗi 20 epochs
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"  • Epoch {epoch:3d}/{epochs:3d} | Total Loss: {current_loss:.4f} | Feat: {l_feat.item():.4f} | Struct: {l_struct.item():.4f}")

        # Cơ chế dừng sớm (Early Stopping) lưu trạng thái tối ưu
        if current_loss < best_loss:
            best_loss = current_loss
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            if save_artifacts and checkpoint_path is not None:
                torch.save(best_state_dict, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  ! Kích hoạt Early Stopping tại epoch {epoch} (Best Loss: {best_loss:.4f})")
            break

    # Phục hồi trọng số tối ưu nhất
    model.load_state_dict(best_state_dict)

    # Trích xuất vector nhúng z in R^{N x 32} và trọng số chú ý ngữ nghĩa beta
    if save_artifacts:
        model.eval()
        with torch.no_grad():
            if isinstance(model, GATGAE):
                # GATGAE: trích xuất z từ forward pass
                z, _ = model(data)
                for i in range(z.shape[1]):
                    data[f"embedding_{i}"] = z[:, i]
                # Lưu đồng thời dạng mảng numpy thuận tiện cho Notebook 04 phân cụm
                z_np = z.detach().cpu().numpy()
                np.save(EMBEDDING_DIR / f"Z_{model_name}.npy", z_np)

            else:  # HANGAE
                # HANGAE: trích xuất z cùng trọng số chú ý beta giữa các quan hệ
                res = model(data)
                z = res[0]
                for i in range(z.shape[1]):
                    data["oa"][f"embedding_{i}"] = z[:, i]
                
                # Lưu ma trận vector nhúng numpy 32 chiều
                z_np = z.detach().cpu().numpy()
                np.save(EMBEDDING_DIR / f"Z_{model_name}.npy", z_np)

                # Lưu các ma trận trọng số chú ý beta1, beta2
                if len(res) >= 4:
                    beta1, beta2 = res[2], res[3]
                    for i in range(beta1.shape[1]):
                        data["oa"][f"beta1_{i}"] = beta1[:, i]
                    for i in range(beta2.shape[1]):
                        data["oa"][f"beta2_{i}"] = beta2[:, i]
                    np.save(EMBEDDING_DIR / f"beta_{model_name}.npy", beta2.detach().cpu().numpy())

            # Lưu đối tượng PyG hoàn chỉnh chứa embedding
            out_pt_path = EMBEDDING_DIR / f"{model_name}.pt"
            torch.save(data, out_pt_path)
            print(f"  ✓ Đã xuất vector nhúng {z.shape} tại: {out_pt_path.resolve()}")

    return data, losses


# %% [Cell 5]: Kiểm thử nhanh tính xác định (Determinism Smoke Test trên đồ thị Hà Nội)
# --- Ghi chú giải thích từ Notebook gốc ---
# Smoke test: repeated short runs must be exactly identical on the same stack.
# -----------------------------------------

def run_determinism_smoke_test():
    """
    Thực hiện 2 lượt huấn luyện độc lập ngắn (5 epochs) trên đồ thị đồng nhất homo.pt.
    Xác nhận lịch sử loss và trọng số mô hình khớp nhau tuyệt đối (chống lỗi tính xác định).
    """
    print("\n-> [Cell 5] Đang thực thi Smoke Test kiểm tra tính xác định (Determinism Smoke Test)...")
    
    homo_file = GRAPH_DIR / "homo.pt"
    if not homo_file.exists():
        raise FileNotFoundError(f"Chưa tìm thấy tệp đồ thị {homo_file}. Hãy chạy Notebook 02 trước!")

    smoke_test_epochs = 5

    def train_once():
        model_seed = config["seeds"].get("model", 42)
        configure_determinism(model_seed)
        
        smoke_data = torch.load(homo_file, map_location=device, weights_only=False).to(device)
        in_dim_smoke = smoke_data.x.shape[1] if hasattr(smoke_data, "x") else 16

        smoke_model = GATGAE(
            in_dim=in_dim_smoke,
            hidden_dim=config["model"]["hidden_dim"],
            out_dim=config["model"]["out_dim"],
            heads=config["model"]["heads"],
            dropout=config["model"]["dropout"],
        ).to(device)

        _, smoke_losses = train_model(
            smoke_model,
            smoke_data,
            config,
            "determinism_smoke_test",
            save_artifacts=False,
            epochs_override=smoke_test_epochs,
        )
        
        smoke_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in smoke_model.state_dict().items()
        }
        return smoke_losses, smoke_state

    losses_a, state_a = train_once()
    losses_b, state_b = train_once()

    # Kiểm tra khớp 100% lịch sử Loss
    assert losses_a == losses_b, "Kiểm thử tính xác định thất bại: Lịch sử loss giữa 2 lượt chạy khác nhau!"
    
    # Kiểm tra khớp 100% trọng số tensor
    assert all(torch.equal(state_a[name], state_b[name]) for name in state_a), (
        "Kiểm thử tính xác định thất bại: Trọng số mô hình giữa 2 lượt chạy khác nhau!"
    )
    
    print("=" * 85)
    print(f"✓ [Cell 5] KIỂM THỬ TÍNH XÁC ĐỊNH THÀNH CÔNG (DETERMINISM SMOKE TEST PASSED)!")
    print(f"  • Số epochs kiểm thử: {smoke_test_epochs} epochs lặp lại 2 lần độc lập.")
    print(f"  • Lịch sử loss khớp tuyệt đối: {losses_a}")
    print(f"  • Thiết bị kiểm thử: {device} | Số chiều đầu vào: 16 POIs Hà Nội.")
    print("=" * 85)

# Kích hoạt Smoke Test
run_determinism_smoke_test()


# %% [Cell 6]: Nạp dữ liệu đồ thị đồng nhất homo.pt (Model 1)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 3. Workflows
# ### 3.1. Model 1: GAT-GAE (Homogeneous)
# This model learns embeddings by aggregating features from spatially contiguous neighbors.
# It assumes that functional regions are spatially smooth.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MODEL 1: GAT-GAE (ĐỒ THỊ TIẾP GIÁP ĐỒNG NHẤT)")
print("=" * 85)

# Đường dẫn tệp đồ thị đồng nhất homo.pt đã nghiệm thu từ File 02
homo_path = GRAPH_DIR / "homo.pt"
if not homo_path.exists():
    homo_path = Path(config["data"]["root"]) / config["data"]["homo"]

print(f"-> [Cell 6] Đang nạp tệp đồ thị đồng nhất từ: {homo_path.resolve()}")
homo_data = torch.load(homo_path, map_location=device, weights_only=False)

# Kiểm tra tính toàn vẹn của đồ thị Model 1 trước khi huấn luyện
n_nodes_homo = homo_data.num_nodes if hasattr(homo_data, "num_nodes") else homo_data.x.shape[0]
n_edges_homo = homo_data.num_edges if hasattr(homo_data, "num_edges") else homo_data.edge_index.shape[1]
in_feat_dim = homo_data.x.shape[1] if hasattr(homo_data, "x") else config["model"]["in_dim"]

print(f"✓ [Cell 6] Nạp thành công đồ thị đồng nhất homo.pt:")
print(f"  • Số lượng đỉnh (Nodes):       {n_nodes_homo:,} ô H3 (Kỳ vọng: 985)")
print(f"  • Số lượng cạnh tiếp giáp:    {n_edges_homo:,} cạnh có hướng (Kỳ vọng: 5,550)")
print(f"  • Số chiều đặc trưng đầu vào: {in_feat_dim} nhóm POIs tầng trệt")


# %% [Cell 7]: Khởi tạo kiến trúc và huấn luyện Model 1: GAT-GAE (200 Epochs)
# --- Ghi chú giải thích từ Notebook gốc ---
# Ensure deterministic model training initialization
# Initialize Model 1: GAT-GAE
# Train Model 1
# -----------------------------------------

configure_determinism(config["seeds"]["model"])

homo_data = homo_data.to(device)

gat_model = GATGAE(
    in_dim=in_feat_dim,
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
).to(device)

print(f"✓ [Cell 7] Đã khởi tạo kiến trúc GATGAE:")
print(f"  • Encoder: 2 tầng Edge-Aware GAT (Heads={config['model']['heads']}, Hidden={config['model']['hidden_dim']}, Out={config['model']['out_dim']})")
print(f"  • Decoder: Cấu trúc DistMult song tuyến + Thuộc tính MLP Smooth L1")
print(f"  • Siêu tham số: Epochs={config['training']['epochs']}, LR={config['training']['lr']}, Patience={config['training']['early_stopping_patience']}")

data_model1, losses_model1 = train_model(
    gat_model,
    homo_data,
    config,
    "model1_gat_gae",
    save_artifacts=True,
)

print(f"✓ [Cell 7] Hoàn thành huấn luyện Model 1 (Số epochs thực tế: {len(losses_model1)} epochs | Loss cuối: {losses_model1[-1]:.4f})")


# %% [Cell 8]: Kiểm tra đối tượng dữ liệu data_model1 sau khi nhúng vector tiềm ẩn z
# --- Ghi chú giải thích từ Notebook gốc ---
# data_model1
# -----------------------------------------

print("\n-> [Cell 8] Kiểm tra đối tượng đồ thị data_model1 chứa các trường Embedding:")
print(data_model1)

z_keys = [f"embedding_{i}" for i in range(config["model"]["out_dim"])]
z_mat = torch.stack([data_model1[k] for k in z_keys], dim=1).detach().cpu().numpy()

print(f"\n✓ [Cell 8] Ma trận vector nhúng Z (Model 1):")
print(f"  • Kích thước tensor:           {z_mat.shape} (985 ô H3 × 32 chiều tiềm ẩn)")
print(f"  • Giá trị trung bình (Mean):   {z_mat.mean():.4f}")
print(f"  • Độ lệch chuẩn (Std):         {z_mat.std():.4f}")
print(f"  • Khoảng giá trị [Min, Max]:   [{z_mat.min():.4f}, {z_mat.max():.4f}]")
print(f"  • Tệp numpy nhúng lưu tại:     {(EMBEDDING_DIR / 'Z_model1_gat_gae.npy').resolve()}")
print(f"  • Tệp checkpoint mô hình tại:  {(CHECKPOINT_DIR / 'model1_gat_gae' / 'model.pt').resolve()}")


# %% [Cell 9]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 1 (Training Loss Progress)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model1)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Progress')
# plt.show()
# -----------------------------------------

print("\n-> [Cell 9] Đang kết xuất biểu đồ đường cong suy giảm hàm mất mát Model 1...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model1,
    color="#0284c7",
    linewidth=2.0,
    label="Model 1 (GAT-GAE: contig)",
)

best_epoch = int(np.argmin(losses_model1))
best_loss_val = float(np.min(losses_model1))
ax.scatter(
    [best_epoch],
    [best_loss_val],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_val:.4f} (Epoch {best_epoch})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Joint Loss (Feat + Struct)", fontsize=11, fontweight="bold")
ax.set_title("Model 1 (GAT-GAE: Contiguity) - Training Loss Progress (Hanoi Case Study)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m1_path = FIGURE_DIR / "hanoi_loss_progress_model1_gat_gae.png"
fig.savefig(fig_loss_m1_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 9] Đã lưu biểu đồ suy giảm Loss Model 1 tại: {fig_loss_m1_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 1 (CELL 6 - CELL 9) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 10]: Khởi tạo kiến trúc và huấn luyện Model 2: HAN-GAE Walk (200 Epochs)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 3.2. Model 2: HAN-GAE (Street)
# This model introduces heterogeneity. It learns to aggregate information not just from contiguous OAs,
# but also from OAs connected via the street network (15-min walk).
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MODEL 2: HAN-GAE (ĐỒ THỊ DỊ THỂ ĐI BỘ 15 PHÚT)")
print("=" * 85)

configure_determinism(config["seeds"]["model"])

walk_path = GRAPH_DIR / "hetero_walk.pt"
if not walk_path.exists():
    walk_path = Path(config["data"]["root"]) / config["data"]["hetero_walk"]

print(f"-> [Cell 10] Đang nạp đồ thị dị thể đi bộ từ: {walk_path.resolve()}")
walk_data = torch.load(walk_path, map_location=device, weights_only=False).to(device)

n_nodes_walk = walk_data["oa"].num_nodes if hasattr(walk_data["oa"], "num_nodes") else walk_data["oa"].x.shape[0]
n_edges_contig = walk_data[("oa", "contig", "oa")].edge_index.shape[1]
n_edges_walk = walk_data[("oa", "15_min_walk", "oa")].edge_index.shape[1]
in_feat_dim_walk = walk_data["oa"].x.shape[1] if hasattr(walk_data["oa"], "x") else config["model"]["in_dim"]

print(f"✓ [Cell 10] Nạp thành công đồ thị dị thể hetero_walk.pt:")
print(f"  • Số lượng đỉnh 'oa':             {n_nodes_walk:,} ô H3 (Kỳ vọng: 985)")
print(f"  • Cạnh tiếp giáp 'contig':        {n_edges_contig:,} cạnh")
print(f"  • Cạnh siêu đường đi '15_min_walk': {n_edges_walk:,} cạnh")
print(f"  • Số chiều đặc trưng đầu vào:     {in_feat_dim_walk} nhóm POIs tầng trệt")

metapaths_walk = ["contig", "15_min_walk"]

han_walk = HANGAE(
    in_dim=in_feat_dim_walk,
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_walk,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_walk": 1},
).to(device)

print(f"✓ [Cell 10] Đã khởi tạo kiến trúc HANGAE Model 2:")
print(f"  • Metapaths: {metapaths_walk} | Edge Dim: 1 (travel_time_sec)")
print(f"  • Heads={config['model']['heads']}, Hidden={config['model']['hidden_dim']}, Out={config['model']['out_dim']}, Dropout={config['model']['dropout']}")

data_model2, losses_model2 = train_model(
    han_walk,
    walk_data,
    config,
    "model2_han_gae_walk",
    save_artifacts=True,
)

print(f"✓ [Cell 10] Hoàn thành huấn luyện Model 2 (Số epochs thực tế: {len(losses_model2)} | Loss cuối: {losses_model2[-1]:.4f})")


# %% [Cell 11]: Kiểm tra đối tượng dữ liệu data_model2 & Phân tích trọng số Attention beta
# --- Ghi chú giải thích từ Notebook gốc ---
# data_model2
# -----------------------------------------

print("\n-> [Cell 11] Kiểm tra đối tượng đồ thị dị thể data_model2 sau huấn luyện:")
print(data_model2)

z_keys_m2 = [f"embedding_{i}" for i in range(config["model"]["out_dim"])]
z_mat_m2 = torch.stack([data_model2["oa"][k] for k in z_keys_m2], dim=1).detach().cpu().numpy()

print(f"\n✓ [Cell 11] Ma trận vector nhúng Z (Model 2):")
print(f"  • Kích thước tensor Z:         {z_mat_m2.shape} (985 ô H3 × 32 chiều)")
print(f"  • Giá trị trung bình (Mean):   {z_mat_m2.mean():.4f} | Độ lệch chuẩn (Std): {z_mat_m2.std():.4f}")
print(f"  • Tệp numpy nhúng lưu tại:     {(EMBEDDING_DIR / 'Z_model2_han_gae_walk.npy').resolve()}")

beta_keys = [k for k in data_model2["oa"].keys() if k.startswith("beta2_")]
if beta_keys:
    beta_mat_m2 = torch.stack([data_model2["oa"][k] for k in beta_keys], dim=1).detach().cpu().numpy()
    print(f"\n✓ [Cell 11] Phân tích trọng số Semantic Attention beta (Layer 2):")
    print(f"  • Kích thước ma trận beta:     {beta_mat_m2.shape} (985 ô × 2 quan hệ)")
    print(f"  • Tỷ trọng trung bình 'contig':       {beta_mat_m2[:, 0].mean():.4f} ± {beta_mat_m2[:, 0].std():.4f}")
    print(f"  • Tỷ trọng trung bình '15_min_walk':  {beta_mat_m2[:, 1].mean():.4f} ± {beta_mat_m2[:, 1].std():.4f}")
    print(f"  • Tệp numpy beta lưu tại:      {(EMBEDDING_DIR / 'beta_model2_han_gae_walk.npy').resolve()}")


# %% [Cell 12]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 2 (Training Loss Progress)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model2)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Progress')
# plt.show()
# -----------------------------------------

print("\n-> [Cell 12] Đang kết xuất biểu đồ đường cong suy giảm hàm mất mát Model 2...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model2,
    color="#059669",
    linewidth=2.0,
    label="Model 2 (HAN-GAE: contig + 15_min_walk)",
)

best_epoch_m2 = int(np.argmin(losses_model2))
best_loss_m2 = float(np.min(losses_model2))
ax.scatter(
    [best_epoch_m2],
    [best_loss_m2],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_m2:.4f} (Epoch {best_epoch_m2})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Joint Loss (Feat + Struct)", fontsize=11, fontweight="bold")
ax.set_title("Model 2 (HAN-GAE: Street Walk) - Training Loss Progress (Hanoi Case Study)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m2_path = FIGURE_DIR / "hanoi_loss_progress_model2_han_gae_walk.png"
fig.savefig(fig_loss_m2_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 12] Đã lưu biểu đồ suy giảm Loss Model 2 tại: {fig_loss_m2_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 2 (CELL 10 - CELL 12) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 13]: Khởi tạo kiến trúc và huấn luyện Model 3: HAN-GAE Multi-modal (200 Epochs)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 3.4. Model 3: HAN-GAE (Multi-modal)
# This is the advanced model. It incorporates public transit (Bus) edges into the accessibility layer.
# OAs are connected if reachable within 15 minutes by walking AND bus.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MODEL 3: HAN-GAE (ĐỒ THỊ DỊ THỂ ĐA PHƯƠNG THỨC XE BUÝT GTFS)")
print("=" * 85)

# Cố định hạt giống ngẫu nhiên phục vụ khởi tạo trọng số mô hình xác định 100%
configure_determinism(config["seeds"]["model"])

# Nạp tệp đồ thị dị thể đa phương thức hetero_multi.pt đã nghiệm thu từ File 02
multi_path = GRAPH_DIR / "hetero_multi.pt"
if not multi_path.exists():
    multi_path = Path(config["data"]["root"]) / config["data"]["hetero_multi"]

print(f"-> [Cell 13] Đang nạp đồ thị dị thể đa phương thức từ: {multi_path.resolve()}")
multi_data = torch.load(multi_path, map_location=device, weights_only=False).to(device)

# Kiểm tra tính toàn vẹn của đồ thị Model 3
n_nodes_multi = multi_data["oa"].num_nodes if hasattr(multi_data["oa"], "num_nodes") else multi_data["oa"].x.shape[0]
n_edges_contig_m3 = multi_data[("oa", "contig", "oa")].edge_index.shape[1]
n_edges_multi_m3 = multi_data[("oa", "15_min_multi", "oa")].edge_index.shape[1]
in_feat_dim_multi = multi_data["oa"].x.shape[1] if hasattr(multi_data["oa"], "x") else config["model"]["in_dim"]

print(f"✓ [Cell 13] Nạp thành công đồ thị dị thể hetero_multi.pt:")
print(f"  • Số lượng đỉnh 'oa':              {n_nodes_multi:,} ô H3 (Kỳ vọng: 985)")
print(f"  • Cạnh tiếp giáp 'contig':         {n_edges_contig_m3:,} cạnh")
print(f"  • Cạnh xe buýt mở rộng '15_min_multi': {n_edges_multi_m3:,} cạnh")
print(f"  • Số chiều đặc trưng đầu vào:      {in_feat_dim_multi} nhóm POIs tầng trệt")

# Khai báo metapaths cho Model 3 (2 quan hệ đối chuẩn: kề sát địa lý + buýt công cộng độc lập)
# Sato et al. (CEUS 2026, Section 4.5.1): Model 3 bóc tách tác động của mạng lưới GTFS độc lập
metapaths_multi = ["contig", "15_min_multi"]

# Khởi tạo mô hình HAN-GAE Model 3
han_multi = HANGAE(
    in_dim=in_feat_dim_multi,
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_multi,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_multi": 1},
).to(device)

print(f"✓ [Cell 13] Đã khởi tạo kiến trúc HANGAE Model 3:")
print(f"  • Metapaths: {metapaths_multi} | Edge Dim: 1 (travel_time_sec)")
print(f"  • Heads={config['model']['heads']}, Hidden={config['model']['hidden_dim']}, Out={config['model']['out_dim']}, Dropout={config['model']['dropout']}")

# Thực thi vòng lặp huấn luyện tự giám sát thống nhất cho Model 3
data_model3, losses_model3 = train_model(
    han_multi,
    multi_data,
    config,
    "model3_han_gae_multi",
    save_artifacts=True,
)

print(f"✓ [Cell 13] Hoàn thành huấn luyện Model 3 (Số epochs thực tế: {len(losses_model3)} | Loss cuối: {losses_model3[-1]:.4f})")


# %% [Cell 14]: Kiểm tra đối tượng dữ liệu data_model3 & Phân tích trọng số Semantic Attention beta
# --- Ghi chú giải thích từ Notebook gốc ---
# data_model3
# -----------------------------------------

print("\n-> [Cell 14] Kiểm tra đối tượng đồ thị dị thể data_model3 sau huấn luyện:")
print(data_model3)

# Trích xuất và kiểm tra ma trận vector nhúng Z in R^{985 x 32}
z_keys_m3 = [f"embedding_{i}" for i in range(config["model"]["out_dim"])]
z_mat_m3 = torch.stack([data_model3["oa"][k] for k in z_keys_m3], dim=1).detach().cpu().numpy()

print(f"\n✓ [Cell 14] Ma trận vector nhúng Z (Model 3):")
print(f"  • Kích thước tensor Z:         {z_mat_m3.shape} (985 ô H3 × 32 chiều)")
print(f"  • Giá trị trung bình (Mean):   {z_mat_m3.mean():.4f} | Độ lệch chuẩn (Std): {z_mat_m3.std():.4f}")
print(f"  • Tệp numpy nhúng lưu tại:     {(EMBEDDING_DIR / 'Z_model3_han_gae_multi.npy').resolve()}")

# Trích xuất và phân tích ma trận trọng số chú ý ngữ nghĩa beta (Layer 2)
beta_keys_m3 = [k for k in data_model3["oa"].keys() if k.startswith("beta2_")]
if beta_keys_m3:
    beta_mat_m3 = torch.stack([data_model3["oa"][k] for k in beta_keys_m3], dim=1).detach().cpu().numpy()
    print(f"\n✓ [Cell 14] Phân tích trọng số Semantic Attention beta (Layer 2):")
    print(f"  • Kích thước ma trận beta:     {beta_mat_m3.shape} (985 ô × 2 quan hệ)")
    print(f"  • Tỷ trọng trung bình 'contig':        {beta_mat_m3[:, 0].mean():.4f} ± {beta_mat_m3[:, 0].std():.4f}")
    print(f"  • Tỷ trọng trung bình '15_min_multi':   {beta_mat_m3[:, 1].mean():.4f} ± {beta_mat_m3[:, 1].std():.4f}")
    print(f"  • Tệp numpy beta lưu tại:       {(EMBEDDING_DIR / 'beta_model3_han_gae_multi.npy').resolve()}")


# %% [Cell 15]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 3 (Training Loss Progress)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model3)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Progress')
# plt.show()
# -----------------------------------------

print("\n-> [Cell 15] Đang kết xuất biểu đồ đường cong suy giảm hàm mất mát Model 3...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

# Vẽ đường cong loss của Model 3 (màu cam hổ phách theo bảng màu đặc trưng giao thông công cộng)
ax.plot(
    losses_model3,
    color="#d97706",
    linewidth=2.0,
    label="Model 3 (HAN-GAE: contig + 15_min_multi)",
)

# Đánh dấu epoch tối ưu nhất
best_epoch_m3 = int(np.argmin(losses_model3))
best_loss_m3 = float(np.min(losses_model3))
ax.scatter(
    [best_epoch_m3],
    [best_loss_m3],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_m3:.4f} (Epoch {best_epoch_m3})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Joint Loss (Feat + Struct)", fontsize=11, fontweight="bold")
ax.set_title("Model 3 (HAN-GAE: Transit Multi-modal) - Training Loss Progress (Hanoi Case Study)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

# Lưu biểu đồ vào thư mục figures
fig_loss_m3_path = FIGURE_DIR / "hanoi_loss_progress_model3_han_gae_multi.png"
fig.savefig(fig_loss_m3_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 15] Đã lưu biểu đồ suy giảm Loss Model 3 tại: {fig_loss_m3_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 3 (CELL 13 - CELL 15) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 16]: Khởi tạo kiến trúc và huấn luyện Model 4: HAN-GAE All (200 Epochs)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 3.5. Model 4: HAN-GAE (All Relations)
# This model combines all available signals: Contiguity, 15-min Street accessibility,
# and 15-min Multi-modal (Walking OR Bus) accessibility.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MODEL 4: HAN-GAE (ĐỒ THỊ DỊ THỂ TOÀN DIỆN CẢ 3 QUAN HỆ)")
print("=" * 85)

# Cố định hạt giống ngẫu nhiên phục vụ khởi tạo trọng số mô hình xác định 100%
configure_determinism(config["seeds"]["model"])

# Nạp tệp đồ thị dị thể đa phương thức hetero_multi.pt cho Model 4
multi_path4 = GRAPH_DIR / "hetero_multi.pt"
if not multi_path4.exists():
    multi_path4 = Path(config["data"]["root"]) / config["data"]["hetero_multi"]

print(f"-> [Cell 16] Đang nạp đồ thị dị thể toàn diện từ: {multi_path4.resolve()}")
multi_data4 = torch.load(multi_path4, map_location=device, weights_only=False).to(device)

# Kiểm tra tính toàn vẹn của đồ thị Model 4
n_nodes_m4 = multi_data4["oa"].num_nodes if hasattr(multi_data4["oa"], "num_nodes") else multi_data4["oa"].x.shape[0]
n_edges_c_m4 = multi_data4[("oa", "contig", "oa")].edge_index.shape[1]
n_edges_w_m4 = multi_data4[("oa", "15_min_walk", "oa")].edge_index.shape[1]
n_edges_m_m4 = multi_data4[("oa", "15_min_multi", "oa")].edge_index.shape[1]
in_feat_dim_m4 = multi_data4["oa"].x.shape[1] if hasattr(multi_data4["oa"], "x") else config["model"]["in_dim"]

print(f"✓ [Cell 16] Nạp thành công đồ thị dị thể hetero_multi.pt cho Model 4:")
print(f"  • Số lượng đỉnh 'oa':              {n_nodes_m4:,} ô H3 (Kỳ vọng: 985)")
print(f"  • Cạnh tiếp giáp 'contig':         {n_edges_c_m4:,} cạnh")
print(f"  • Cạnh đi bộ 15p '15_min_walk':    {n_edges_w_m4:,} cạnh")
print(f"  • Cạnh xe buýt mở rộng '15_min_multi': {n_edges_m_m4:,} cạnh")
print(f"  • Tổng liên kết đa tầng:           {n_edges_c_m4 + n_edges_w_m4 + n_edges_m_m4:,} cạnh")
print(f"  • Số chiều đặc trưng đầu vào:      {in_feat_dim_m4} nhóm POIs tầng trệt")

# Khai báo metapaths đầy đủ cả 3 nhánh quan hệ theo Section 4.5.1
# Model 4: M = {contig, 15_min_walk, 15_min_multi}
metapaths_all = ["contig", "15_min_walk", "15_min_multi"]

# Khởi tạo mô hình HAN-GAE Model 4
han_all = HANGAE(
    in_dim=in_feat_dim_m4,
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_all,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_walk": 1, "15_min_multi": 1},
).to(device)

print(f"✓ [Cell 16] Đã khởi tạo kiến trúc HANGAE Model 4:")
print(f"  • Metapaths: {metapaths_all} | Edge Dim: 1 cho mỗi quan hệ (travel_time_sec)")
print(f"  • Heads={config['model']['heads']}, Hidden={config['model']['hidden_dim']}, Out={config['model']['out_dim']}, Dropout={config['model']['dropout']}")

# Thực thi vòng lặp huấn luyện tự giám sát thống nhất cho Model 4
data_model4, losses_model4 = train_model(
    han_all,
    multi_data4,
    config,
    "model4_han_gae_all",
    save_artifacts=True,
)

print(f"✓ [Cell 16] Hoàn thành huấn luyện Model 4 (Số epochs thực tế: {len(losses_model4)} | Loss cuối: {losses_model4[-1]:.4f})")


# %% [Cell 17]: Kiểm tra đối tượng dữ liệu data_model4 & Phân tích trọng số Semantic Attention beta
# --- Ghi chú giải thích từ Notebook gốc ---
# data_model4
# -----------------------------------------

print("\n-> [Cell 17] Kiểm tra đối tượng đồ thị dị thể data_model4 sau huấn luyện:")
print(data_model4)

# Trích xuất và kiểm tra ma trận vector nhúng Z in R^{985 x 32}
z_keys_m4 = [f"embedding_{i}" for i in range(config["model"]["out_dim"])]
z_mat_m4 = torch.stack([data_model4["oa"][k] for k in z_keys_m4], dim=1).detach().cpu().numpy()

print(f"\n✓ [Cell 17] Ma trận vector nhúng Z (Model 4):")
print(f"  • Kích thước tensor Z:         {z_mat_m4.shape} (985 ô H3 × 32 chiều)")
print(f"  • Giá trị trung bình (Mean):   {z_mat_m4.mean():.4f} | Độ lệch chuẩn (Std): {z_mat_m4.std():.4f}")
print(f"  • Tệp numpy nhúng lưu tại:     {(EMBEDDING_DIR / 'Z_model4_han_gae_all.npy').resolve()}")

# Trích xuất và phân tích ma trận trọng số chú ý ngữ nghĩa beta (Layer 2) cả 3 nhánh
beta_keys_m4 = [k for k in data_model4["oa"].keys() if k.startswith("beta2_")]
if beta_keys_m4:
    beta_mat_m4 = torch.stack([data_model4["oa"][k] for k in beta_keys_m4], dim=1).detach().cpu().numpy()
    print(f"\n✓ [Cell 17] Phân tích trọng số Semantic Attention beta (Layer 2 - Đầy đủ 3 quan hệ):")
    print(f"  • Kích thước ma trận beta:     {beta_mat_m4.shape} (985 ô × 3 quan hệ)")
    print(f"  • Tỷ trọng trung bình 'contig':        {beta_mat_m4[:, 0].mean():.4f} ± {beta_mat_m4[:, 0].std():.4f}")
    print(f"  • Tỷ trọng trung bình '15_min_walk':   {beta_mat_m4[:, 1].mean():.4f} ± {beta_mat_m4[:, 1].std():.4f}")
    print(f"  • Tỷ trọng trung bình '15_min_multi':  {beta_mat_m4[:, 2].mean():.4f} ± {beta_mat_m4[:, 2].std():.4f}")
    np.save(EMBEDDING_DIR / f"beta_model4_han_gae_all.npy", beta_mat_m4)
    print(f"  • Tệp numpy beta lưu tại:       {(EMBEDDING_DIR / 'beta_model4_han_gae_all.npy').resolve()}")


# %% [Cell 18]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 4 (Training Loss Progress)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model4)
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training Loss Progress')
# plt.show()
# -----------------------------------------

print("\n-> [Cell 18] Đang kết xuất biểu đồ đường cong suy giảm hàm mất mát Model 4...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

# Vẽ đường cong loss của Model 4 (màu xanh dương đậm theo bảng màu mô hình toàn diện nhất)
ax.plot(
    losses_model4,
    color="#1d4ed8",
    linewidth=2.0,
    label="Model 4 (HAN-GAE: contig + 15_min_walk + 15_min_multi)",
)

# Đánh dấu epoch tối ưu nhất
best_epoch_m4 = int(np.argmin(losses_model4))
best_loss_m4 = float(np.min(losses_model4))
ax.scatter(
    [best_epoch_m4],
    [best_loss_m4],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_m4:.4f} (Epoch {best_epoch_m4})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Joint Loss (Feat + Struct)", fontsize=11, fontweight="bold")
ax.set_title("Model 4 (HAN-GAE: All Relations) - Training Loss Progress (Hanoi Case Study)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

# Lưu biểu đồ vào thư mục figures
fig_loss_m4_path = FIGURE_DIR / "hanoi_loss_progress_model4_han_gae_all.png"
fig.savefig(fig_loss_m4_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 18] Đã lưu biểu đồ suy giảm Loss Model 4 tại: {fig_loss_m4_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 4 (CELL 16 - CELL 18) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 19]: Khởi tạo kiến trúc và huấn luyện Model 1S: GAT-GAE Struct-Only (Contiguity)
# --- Ghi chú giải thích từ Notebook gốc ---
# ## 4. Structure-Only Baselines (Models 1S–4S)
# To isolate the effect of graph connectivity, we train the same encoders without node attributes,
# using only learnable node identity embeddings and the structure reconstruction loss.
# - Model 1S (GAT-GAE): contiguity edges only
# - Model 2S (HAN-GAE): contiguity + 15_min_walk
# - Model 3S (HAN-GAE): contiguity + 15_min_multi
# - Model 4S (HAN-GAE): contiguity + 15_min_walk + 15_min_multi
#
# Training changes:
# - Replace node features with learnable embeddings E in R^{N x d_0} (lookup by node id)
# - Remove attribute reconstruction (feature loss = 0)
# - Keep everything else identical: same relations, same negative sampling, same hidden/embedding sizes, same early stopping
# ### 4.1. Model 1S: GAT-GAE (Structure-Only, Contiguity)
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU HUẤN LUYỆN MODEL 1S: GAT-GAE (CẤU TRÚC THUẦN TÚY - TIẾP GIÁP ĐỊA LÝ)")
print("=" * 85)

# Cố định hạt giống ngẫu nhiên phục vụ tính xác định 100%
configure_determinism(config["seeds"]["model"])

# Nạp lại dữ liệu đồ thị đồng nhất gốc
homo_data_1s = torch.load(homo_path, map_location=device, weights_only=False).to(device)
n_nodes_1s = homo_data_1s.num_nodes if hasattr(homo_data_1s, "num_nodes") else homo_data_1s.x.shape[0]

# Khởi tạo Model 1S: use_features=False, kích hoạt bảng tra cứu embedding nút E in R^{N x in_dim}
gat_struct = GATGAE(
    in_dim=config["model"]["in_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    use_features=False,
    num_nodes=n_nodes_1s,
).to(device)

print(f"✓ [Cell 19] Đã khởi tạo kiến trúc GATGAE Model 1S (Structure-Only):")
print(f"  • use_features=False | num_nodes={n_nodes_1s} (Bảng tra cứu Node Identity Embedding)")
print(f"  • Cấu hình: Heads={config['model']['heads']}, Hidden={config['model']['hidden_dim']}, Out={config['model']['out_dim']}")

# Huấn luyện Model 1S
data_model1s, losses_model1s = train_model(
    gat_struct,
    homo_data_1s,
    config,
    "model1s_gat_gae_struct",
    save_artifacts=True,
)

print(f"✓ [Cell 19] Hoàn thành huấn luyện Model 1S (Số epochs: {len(losses_model1s)} | Loss cuối: {losses_model1s[-1]:.4f})")


# %% [Cell 20]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 1S (Structure-Only)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model1s)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model 1S: Training Loss Progress (Structure-Only)")
# plt.show()
# -----------------------------------------

print("\n-> [Cell 20] Đang kết xuất biểu đồ suy giảm hàm mất mát Model 1S (Structure-Only)...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model1s,
    color="#0284c7",
    linestyle="--",
    linewidth=2.0,
    label="Model 1S (GAT-GAE: contig Struct-only)",
)

best_epoch_1s = int(np.argmin(losses_model1s))
best_loss_1s = float(np.min(losses_model1s))
ax.scatter(
    [best_epoch_1s],
    [best_loss_1s],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_1s:.4f} (Epoch {best_epoch_1s})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Structure Loss (DistMult BCE)", fontsize=11, fontweight="bold")
ax.set_title("Model 1S (GAT-GAE: Contiguity) - Structure-Only Loss Progress (Hanoi)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m1s_path = FIGURE_DIR / "hanoi_loss_progress_model1s_gat_gae_struct.png"
fig.savefig(fig_loss_m1s_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 20] Đã lưu biểu đồ suy giảm Loss Model 1S tại: {fig_loss_m1s_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 1S (CELL 19 - CELL 20) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 21]: Khởi tạo kiến trúc và huấn luyện Model 2S: HAN-GAE Struct-Only (Walk)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 4.2. Model 2S: HAN-GAE (Structure-Only, Walk)
# Structure-only counterpart of Model 2. Uses contiguity + 15-min walk edges.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU HUẤN LUYỆN MODEL 2S: HAN-GAE (CẤU TRÚC THUẦN TÚY - ĐI BỘ 15 PHÚT)")
print("=" * 85)

configure_determinism(config["seeds"]["model"])

# Nạp lại dữ liệu đồ thị dị thể đi bộ
walk_data_2s = torch.load(walk_path, map_location=device, weights_only=False).to(device)
n_nodes_walk_2s = walk_data_2s["oa"].num_nodes if hasattr(walk_data_2s["oa"], "num_nodes") else walk_data_2s["oa"].x.shape[0]

metapaths_walk_2s = ["contig", "15_min_walk"]

han_walk_struct = HANGAE(
    in_dim=config["model"]["in_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_walk_2s,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_walk": 1},
    use_features=False,
    num_nodes=n_nodes_walk_2s,
).to(device)

print(f"✓ [Cell 21] Đã khởi tạo kiến trúc HANGAE Model 2S (Structure-Only):")
print(f"  • Metapaths: {metapaths_walk_2s} | use_features=False | num_nodes={n_nodes_walk_2s}")

data_model2s, losses_model2s = train_model(
    han_walk_struct,
    walk_data_2s,
    config,
    "model2s_han_gae_walk_struct",
    save_artifacts=True,
)

print(f"✓ [Cell 21] Hoàn thành huấn luyện Model 2S (Số epochs: {len(losses_model2s)} | Loss cuối: {losses_model2s[-1]:.4f})")


# %% [Cell 22]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 2S (Structure-Only)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model2s)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model 2S: Training Loss Progress (Structure-Only)")
# plt.show()
# -----------------------------------------

print("\n-> [Cell 22] Đang kết xuất biểu đồ suy giảm hàm mất mát Model 2S (Structure-Only)...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model2s,
    color="#059669",
    linestyle="--",
    linewidth=2.0,
    label="Model 2S (HAN-GAE: contig + walk Struct-only)",
)

best_epoch_2s = int(np.argmin(losses_model2s))
best_loss_2s = float(np.min(losses_model2s))
ax.scatter(
    [best_epoch_2s],
    [best_loss_2s],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_2s:.4f} (Epoch {best_epoch_2s})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Structure Loss (DistMult BCE)", fontsize=11, fontweight="bold")
ax.set_title("Model 2S (HAN-GAE: Street Walk) - Structure-Only Loss Progress (Hanoi)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m2s_path = FIGURE_DIR / "hanoi_loss_progress_model2s_han_gae_walk_struct.png"
fig.savefig(fig_loss_m2s_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 22] Đã lưu biểu đồ suy giảm Loss Model 2S tại: {fig_loss_m2s_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 2S (CELL 21 - CELL 22) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 23]: Khởi tạo kiến trúc và huấn luyện Model 3S: HAN-GAE Struct-Only (Multi-modal)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 4.3. Model 3S: HAN-GAE (Structure-Only, Multi-modal)
# Structure-only counterpart of Model 3. Uses contiguity + 15-min multi-modal edges.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU HUẤN LUYỆN MODEL 3S: HAN-GAE (CẤU TRÚC THUẦN TÚY - ĐA PHƯƠNG THỨC XE BUÝT)")
print("=" * 85)

configure_determinism(config["seeds"]["model"])

# Nạp lại dữ liệu đồ thị dị thể đa phương thức
multi_data_3s = torch.load(multi_path, map_location=device, weights_only=False).to(device)
n_nodes_multi_3s = multi_data_3s["oa"].num_nodes if hasattr(multi_data_3s["oa"], "num_nodes") else multi_data_3s["oa"].x.shape[0]

metapaths_multi_3s = ["contig", "15_min_multi"]

han_multi_struct = HANGAE(
    in_dim=config["model"]["in_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_multi_3s,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_multi": 1},
    use_features=False,
    num_nodes=n_nodes_multi_3s,
).to(device)

print(f"✓ [Cell 23] Đã khởi tạo kiến trúc HANGAE Model 3S (Structure-Only):")
print(f"  • Metapaths: {metapaths_multi_3s} | use_features=False | num_nodes={n_nodes_multi_3s}")

data_model3s, losses_model3s = train_model(
    han_multi_struct,
    multi_data_3s,
    config,
    "model3s_han_gae_multi_struct",
    save_artifacts=True,
)

print(f"✓ [Cell 23] Hoàn thành huấn luyện Model 3S (Số epochs: {len(losses_model3s)} | Loss cuối: {losses_model3s[-1]:.4f})")


# %% [Cell 24]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 3S (Structure-Only)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model3s)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model 3S: Training Loss Progress (Structure-Only)")
# plt.show()
# -----------------------------------------

print("\n-> [Cell 24] Đang kết xuất biểu đồ suy giảm hàm mất mát Model 3S (Structure-Only)...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model3s,
    color="#d97706",
    linestyle="--",
    linewidth=2.0,
    label="Model 3S (HAN-GAE: contig + multi Struct-only)",
)

best_epoch_3s = int(np.argmin(losses_model3s))
best_loss_3s = float(np.min(losses_model3s))
ax.scatter(
    [best_epoch_3s],
    [best_loss_3s],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_3s:.4f} (Epoch {best_epoch_3s})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Structure Loss (DistMult BCE)", fontsize=11, fontweight="bold")
ax.set_title("Model 3S (HAN-GAE: Transit Multi-modal) - Structure-Only Loss Progress (Hanoi)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m3s_path = FIGURE_DIR / "hanoi_loss_progress_model3s_han_gae_multi_struct.png"
fig.savefig(fig_loss_m3s_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 24] Đã lưu biểu đồ suy giảm Loss Model 3S tại: {fig_loss_m3s_path.resolve()}")
plt.show()
plt.close(fig)

print("=" * 85)
print("✓ HOÀN THÀNH HUẤN LUYỆN MODEL 3S (CELL 23 - CELL 24) CHO HÀ NỘI!")
print("=" * 85)


# %% [Cell 25]: Khởi tạo kiến trúc và huấn luyện Model 4S: HAN-GAE Struct-Only (All Relations)
# --- Ghi chú giải thích từ Notebook gốc ---
# ### 4.4. Model 4S: HAN-GAE (Structure-Only, All Relations)
# Structure-only counterpart of Model 4. Uses all three relation types:
# contiguity + 15-min walk + 15-min multi-modal.
# -----------------------------------------

print("\n" + "=" * 85)
print("BẮT ĐẦU HUẤN LUYỆN MODEL 4S: HAN-GAE (CẤU TRÚC THUẦN TÚY - TOÀN DIỆN CẢ 3 QUAN HỆ)")
print("=" * 85)

configure_determinism(config["seeds"]["model"])

# Nạp lại dữ liệu đồ thị dị thể toàn diện
multi_data_4s = torch.load(multi_path4, map_location=device, weights_only=False).to(device)
n_nodes_multi_4s = multi_data_4s["oa"].num_nodes if hasattr(multi_data_4s["oa"], "num_nodes") else multi_data_4s["oa"].x.shape[0]

metapaths_all_4s = ["contig", "15_min_walk", "15_min_multi"]

han_all_struct = HANGAE(
    in_dim=config["model"]["in_dim"],
    hidden_dim=config["model"]["hidden_dim"],
    out_dim=config["model"]["out_dim"],
    metapaths=metapaths_all_4s,
    heads=config["model"]["heads"],
    dropout=config["model"]["dropout"],
    edge_dim_dict={"contig": 1, "15_min_walk": 1, "15_min_multi": 1},
    use_features=False,
    num_nodes=n_nodes_multi_4s,
).to(device)

print(f"✓ [Cell 25] Đã khởi tạo kiến trúc HANGAE Model 4S (Structure-Only):")
print(f"  • Metapaths: {metapaths_all_4s} | use_features=False | num_nodes={n_nodes_multi_4s}")

data_model4s, losses_model4s = train_model(
    han_all_struct,
    multi_data_4s,
    config,
    "model4s_han_gae_all_struct",
    save_artifacts=True,
)

print(f"✓ [Cell 25] Hoàn thành huấn luyện Model 4S (Số epochs: {len(losses_model4s)} | Loss cuối: {losses_model4s[-1]:.4f})")


# %% [Cell 26]: Trực quan hóa tiến trình suy giảm hàm mất mát Model 4S (Structure-Only)
# --- Ghi chú giải thích từ Notebook gốc ---
# plt.plot(losses_model4s)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Model 4S: Training Loss Progress (Structure-Only)")
# plt.show()
# -----------------------------------------

print("\n-> [Cell 26] Đang kết xuất biểu đồ suy giảm hàm mất mát Model 4S (Structure-Only)...")

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)

ax.plot(
    losses_model4s,
    color="#1d4ed8",
    linestyle="--",
    linewidth=2.0,
    label="Model 4S (HAN-GAE: All Relations Struct-only)",
)

best_epoch_4s = int(np.argmin(losses_model4s))
best_loss_4s = float(np.min(losses_model4s))
ax.scatter(
    [best_epoch_4s],
    [best_loss_4s],
    color="#ef4444",
    s=65,
    zorder=5,
    label=f"Best Loss: {best_loss_4s:.4f} (Epoch {best_epoch_4s})",
)

ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
ax.set_ylabel("Structure Loss (DistMult BCE)", fontsize=11, fontweight="bold")
ax.set_title("Model 4S (HAN-GAE: All Relations) - Structure-Only Loss Progress (Hanoi)", fontsize=12, fontweight="bold", pad=12)
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right", frameon=True, fontsize=10)

plt.tight_layout()

fig_loss_m4s_path = FIGURE_DIR / "hanoi_loss_progress_model4s_han_gae_all_struct.png"
fig.savefig(fig_loss_m4s_path, dpi=200, bbox_inches="tight")
print(f"✓ [Cell 26] Đã lưu biểu đồ suy giảm Loss Model 4S tại: {fig_loss_m4s_path.resolve()}")
plt.show()
plt.close(fig)

print("\n" + "=" * 85)
print("✓ HOÀN THÀNH 100% TIẾN TRÌNH HUẤN LUYỆN TOÀN BỘ 26 CELLS FILE 03 CHO HÀ NỘI!")
print("=" * 85)
print("• Đã xuất đầy đủ 8 vector nhúng Z (Z_model1..4 và Z_model1s..4s) tại data/outputs/hanoi/embeddings/")
print("• Đã xuất đầy đủ 8 biểu đồ động học hàm mất mát tại data/outputs/hanoi/figures/")
print("• Toàn bộ hệ thống sẵn sàng 100% để bước sang Notebook 04: Đánh giá phân cụm & kiểm định Modularity Q!")
print("=" * 85)