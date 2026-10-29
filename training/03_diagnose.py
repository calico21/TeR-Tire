#!/usr/bin/env python3
"""Diagnóstico: suelo de ruido del sensor y desglose del error por condición (conjunto VAL)."""

import sys, math
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from data.loaders import TTCLoader
from models.grey_box_tire import GreyBoxTireModel

BLOCK, VAL_FRAC = 256, 0.15          # idéntico a 02_train_pinn.py
WEIGHTS = "export/weights/pinn_residual_best.pth"


def rms(e):
    return torch.sqrt(torch.mean(e ** 2)).item()


def noise_sigma(y):
    """Ruido blanco estimado con la segunda diferencia (var = 6 sigma^2); robusto con MAD."""
    d2 = y[:-2] - 2 * y[1:-1] + y[2:]
    mad = torch.median(torch.abs(d2 - torch.median(d2)))
    return (1.4826 * mad / math.sqrt(6.0)).item()


def table(title, v, edges, e_b, e_h):
    print(f"\n{title}")
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        m = (v >= lo) & (v < hi + (1e-6 if i == len(edges) - 2 else 0.0))
        if m.sum() < 50:
            continue
        print(f"  [{lo:9.2f},{hi:9.2f}) n={m.sum().item():6d} | Pacejka {rms(e_b[m]):6.1f} | "
              f"Híbrido {rms(e_h[m]):6.1f} | sesgo {e_h[m].mean().item():+6.1f}")


def quant_edges(v, k=6):
    return torch.quantile(v, torch.linspace(0, 1, k, device=v.device)).tolist()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw = TTCLoader("config/ttc_dataset.yaml").load_specific_runs(
    "data/raw_ttc/RunData_Cornering_ASCII_SI_Round9", run_numbers=[8, 9])
x, y = raw.x.to(device), raw.y.to(device)

model = GreyBoxTireModel("config/pacejka_mf62_params.yaml").to(device).eval()
model.residual.load_state_dict(torch.load(WEIGHTS, map_location=device))

with torch.no_grad():
    fy_b, mz_b = model.baseline(x[:, 0], x[:, 2], x[:, 3], x[:, 4])
    d_fy, d_mz = model.residual(*x[:, :7].unbind(-1))
e_b_fy, e_h_fy = fy_b - y[:, 1], fy_b + d_fy - y[:, 1]
e_b_mz, e_h_mz = mz_b - y[:, 2], mz_b + d_mz - y[:, 2]

# Mismo split que en el entrenamiento
n = len(x)
n_blocks = math.ceil(n / BLOCK)
perm_b = torch.randperm(n_blocks, generator=torch.Generator().manual_seed(0))
val_blocks = perm_b[:max(1, int(VAL_FRAC * n_blocks))]
val = torch.isin(torch.arange(n) // BLOCK, val_blocks).to(device)

print("=== Suelo de ruido (sobre todo el dataset) ===")
print(f"sigma ruido Fy ~ {noise_sigma(y[:, 1]):.1f} N | RMSE híbrido Fy (todo) {rms(e_h_fy):.1f} N")
print(f"sigma ruido Mz ~ {noise_sigma(y[:, 2]):.2f} N*m | RMSE híbrido Mz (todo) {rms(e_h_mz):.2f} N*m")
ac = torch.corrcoef(torch.stack([e_h_fy[:-1], e_h_fy[1:]]))[0, 1].item()
print(f"Autocorrelación lag-1 del error Fy: {ac:.3f}  [~1 = error sistemático/lento, ~0 = ruido blanco]")

print("\n=== Error en VAL por condición (RMSE en N; sesgo = media del error del híbrido) ===")
v = lambda t: t[val]
table("|alpha| [deg]", v(torch.rad2deg(x[:, 0].abs())), [0, 1, 3, 6, 10, 40], v(e_b_fy), v(e_h_fy))
table("Fz [N]", v(x[:, 3]), quant_edges(v(x[:, 3])), v(e_b_fy), v(e_h_fy))
table("gamma [deg]", v(torch.rad2deg(x[:, 2])), quant_edges(v(torch.rad2deg(x[:, 2]))), v(e_b_fy), v(e_h_fy))
table("Presión", v(x[:, 4]), quant_edges(v(x[:, 4]), 4), v(e_b_fy), v(e_h_fy))
table("T_avg", v(x[:, 5]), quant_edges(v(x[:, 5])), v(e_b_fy), v(e_h_fy))
table("alpha_dot [rad/s]", v(x[:, 1].abs()), quant_edges(v(x[:, 1].abs()), 4), v(e_b_fy), v(e_h_fy))