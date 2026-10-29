#!/usr/bin/env python3
"""
Entrenamiento del corrector residual PINN (7 entradas) sobre la base congelada de Pacejka MF 6.2.
- Validación por bloques contiguos (sin fuga entre muestras vecinas).
- EMA de pesos, OneCycle, clipping, selección del mejor checkpoint por RMSE Fy de validación.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import copy
import math
import torch
import torch.optim as optim
from data.loaders import TTCLoader
from models.grey_box_tire import GreyBoxTireModel
from models.physics_loss import PINNTireLoss

EPOCHS = 400
BATCH = 1024
LR = 3e-3
WEIGHT_DECAY = 1e-4
BLOCK = 256          # muestras contiguas por bloque de validación
VAL_FRAC = 0.15
EMA_DECAY = 0.998
SAVE_PATH = "export/weights/pinn_residual_best.pth"


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def train_pinn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Entrenamiento PINN Residual (7 entradas) — dispositivo: {device} ===")

    loader = TTCLoader("config/ttc_dataset.yaml")
    data_dir = "data/raw_ttc/RunData_Cornering_ASCII_SI_Round9"
    raw_dataset = loader.load_specific_runs(data_dir, run_numbers=[8, 9])

    # Columnas: 0=alpha, 1=alpha_dot, 2=gamma, 3=fz, 4=pressure, 5=temp_avg, 6=delta_temp
    x_all = raw_dataset.x.to(device)
    y_all = raw_dataset.y.to(device)  # [fx, fy, mz]

    model = GreyBoxTireModel("config/pacejka_mf62_params.yaml").to(device)
    model.eval()

    with torch.no_grad():
        fy_base, mz_base = model.baseline(x_all[:, 0], x_all[:, 2], x_all[:, 3], x_all[:, 4])
        res_fy = y_all[:, 1] - fy_base
        res_mz = y_all[:, 2] - mz_base

    inputs = x_all[:, :7].contiguous()
    targets = torch.stack([res_fy, res_mz, fy_base, mz_base], dim=-1)
    n = len(inputs)

    # ---------------- Diagnóstico de causa raíz ----------------
    alpha_d = x_all[:, 0]
    print("\n--- Diagnóstico ---")
    print(f"Pacejka global      -> RMSE Fy {rmse(res_fy, 0*res_fy):.2f} N | RMSE Mz {rmse(res_mz, 0*res_mz):.2f} N*m")
    print(f"|res_Fy| > {model.residual.max_delta_fy:.0f} N (guarda): {(res_fy.abs() > model.residual.max_delta_fy).float().mean().item()*100:.2f} %")
    print(f"|res_Fy| > 250 N (límite anterior): {(res_fy.abs() > 250).float().mean().item()*100:.2f} %")
    small = alpha_d.abs() < 0.01
    if small.any():
        print(f"|alpha|<0.01 rad     -> media res_Fy {res_fy[small].mean().item():+.2f} N | RMSE {rmse(res_fy[small], 0*res_fy[small]):.2f} N "
              f"({small.float().mean().item()*100:.1f} % de datos)  [si es alto, subir gate_floor]")
    mask = (x_all[:, 2].abs() < 0.015) & (alpha_d.abs() > 0.02)
    if mask.any():
        viol = (y_all[mask, 1] * torch.sign(alpha_d[mask]) < 0).float().mean().item() * 100
        print(f"Datos MEDIDOS que violan Fy*sign(alpha)>0: {viol:.2f} %  [debe ser ~0; si no, revisar convención de signos SAE/ISO]")
    step_a = (alpha_d[1:] - alpha_d[:-1]).abs().median().item()
    print(f"Mediana |d alpha| entre muestras consecutivas: {step_a:.2e} rad  [si es grande, los datos no están en orden temporal y los bloques no evitan la fuga]")
    print("---------------------\n")

    # ---------------- Split por bloques contiguos ----------------
    n_blocks = math.ceil(n / BLOCK)
    g = torch.Generator().manual_seed(0)
    perm_b = torch.randperm(n_blocks, generator=g)
    val_blocks = perm_b[:max(1, int(VAL_FRAC * n_blocks))]
    block_id = torch.arange(n) // BLOCK
    val_mask = torch.isin(block_id, val_blocks).to(device)

    x_tr, t_tr = inputs[~val_mask], targets[~val_mask]
    x_va, t_va = inputs[val_mask], targets[val_mask]
    base_val_fy = rmse(t_va[:, 0], torch.zeros_like(t_va[:, 0]))
    base_val_mz = rmse(t_va[:, 1], torch.zeros_like(t_va[:, 1]))
    print(f"Train: {len(x_tr)} | Val: {len(x_va)}")
    print(f"Pacejka en VAL -> RMSE Fy: {base_val_fy:.2f} N | RMSE Mz: {base_val_mz:.2f} N*m")

    criterion = PINNTireLoss(lambda_data=1.0, lambda_passivity=0.5, lambda_reg=1e-4)
    optimizer = optim.AdamW(model.residual.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(x_tr) / BATCH)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos", final_div_factor=100.0
    )

    ema = copy.deepcopy(model.residual).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    def ema_update(step):
        decay = min(EMA_DECAY, (1.0 + step) / (10.0 + step))
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), model.residual.parameters()):
                pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)

    @torch.no_grad()
    def evaluate(net, x, t):
        d_fy, d_mz = net(*x.unbind(-1))
        return rmse(d_fy, t[:, 0]), rmse(d_mz, t[:, 1])

    os.makedirs("export/weights", exist_ok=True)
    best_val = base_val_fy
    best_state = None
    best_epoch = 0
    step = 0

    for epoch in range(1, EPOCHS + 1):
        model.residual.train()
        perm = torch.randperm(len(x_tr), device=device)
        loss_accum = 0.0

        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            b_in, b_tgt = x_tr[idx], t_tr[idx]

            optimizer.zero_grad(set_to_none=True)
            d_fy, d_mz = model.residual(*b_in.unbind(-1))
            fy_total = b_tgt[:, 2] + d_fy
            loss, _, _ = criterion(d_fy, d_mz, b_tgt[:, 0], b_tgt[:, 1], fy_total, b_in[:, 0], b_in[:, 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            ema_update(step)
            loss_accum += loss.item() * len(idx)

        v_fy, v_mz = evaluate(ema, x_va, t_va)
        if v_fy < best_val:
            best_val, best_epoch = v_fy, epoch
            best_state = copy.deepcopy(ema.state_dict())
            torch.save(best_state, SAVE_PATH)

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            t_fy, _ = evaluate(ema, x_tr, t_tr)
            print(f"Epoch {epoch:03d}/{EPOCHS} | Loss {loss_accum/len(x_tr):.3e} | "
                  f"Train Fy {t_fy:.2f} N | Val Fy {v_fy:.2f} N | Val Mz {v_mz:.2f} N*m | LR {optimizer.param_groups[0]['lr']:.2e}")

    print("\nEntrenamiento finalizado.")
    print(f"Pacejka (VAL)      : Fy {base_val_fy:.2f} N")
    if best_state is None:
        print("La red no mejoró a Pacejka en validación; no se guardaron pesos.")
        return
    ema.load_state_dict(best_state)
    v_fy, v_mz = evaluate(ema, x_va, t_va)
    print(f"PINN híbrido (VAL) : Fy {v_fy:.2f} N (−{(base_val_fy - v_fy)/base_val_fy*100:.1f} %) | "
          f"Mz {v_mz:.2f} N*m (−{(base_val_mz - v_mz)/base_val_mz*100:.1f} %)  [mejor época: {best_epoch}]")
    print(f"Pesos guardados en: {SAVE_PATH}")


if __name__ == "__main__":
    train_pinn()