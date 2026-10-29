#!/usr/bin/env python3
"""
Ajuste desacoplado de la Magic Formula 6.2 sobre los 107.112 puntos de TTC Calspan.
Optimización de mínimos cuadrados no lineales en PyTorch mediante Adam.
"""

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto siempre esté en sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import torch
import torch.optim as optim
from data.loaders import TTCLoader
from physics.pacejka_baseline import PacejkaMF62

def fit():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Iniciando ajuste MF 6.2 en dispositivo: {device}")

    # 1. Cargar el dataset consolidado
    loader = TTCLoader("config/ttc_dataset.yaml")
    data_dir = "data/raw_ttc/RunData_Cornering_ASCII_SI_Round9"
    dataset = loader.load_specific_runs(data_dir, run_numbers=[8, 9])

    x = dataset.x.to(device)  # [alpha, kappa, gamma, fz, pressure, temp]
    y = dataset.y.to(device)  # [fx, fy, mz]

    alpha = x[:, 0]
    gamma = x[:, 2]
    fz = x[:, 3]
    pressure = x[:, 4]

    fy_target = y[:, 1]
    mz_target = y[:, 2]

    model = PacejkaMF62().to(device)

    # ----------------------------------------------------
    # FASE 1: Ajuste de Parámetros de Fuerza Lateral (Fy0)
    # ----------------------------------------------------
    print("\n--- FASE 1: Optimización de Parámetros Fy0 ---")
    fy_params = [p for name, p in model.named_parameters() if name.startswith("p_")]
    optimizer_fy = optim.Adam(fy_params, lr=0.02)

    for epoch in range(1500):
        optimizer_fy.zero_grad()
        fy_pred, _ = model(alpha, gamma, fz, pressure)
        loss_fy = torch.mean((fy_pred - fy_target) ** 2)
        loss_fy.backward()
        optimizer_fy.step()

        if (epoch + 1) % 300 == 0:
            rmse_fy = torch.sqrt(loss_fy).item()
            print(f"Epoch {epoch+1:04d} | RMSE Fy: {rmse_fy:.2f} N")

    # ----------------------------------------------------
    # FASE 2: Ajuste de Momento Autoalineante (Mz0)
    # ----------------------------------------------------
    print("\n--- FASE 2: Optimización de Parámetros Mz0 (Congelando Fy) ---")
    for p in fy_params:
        p.requires_grad = False

    mz_params = [p for name, p in model.named_parameters() if name.startswith("q_")]
    optimizer_mz = optim.Adam(mz_params, lr=0.01)

    for epoch in range(1500):
        optimizer_mz.zero_grad()
        _, mz_pred = model(alpha, gamma, fz, pressure)
        loss_mz = torch.mean((mz_pred - mz_target) ** 2)
        loss_mz.backward()
        optimizer_mz.step()

        if (epoch + 1) % 300 == 0:
            rmse_mz = torch.sqrt(loss_mz).item()
            print(f"Epoch {epoch+1:04d} | RMSE Mz: {rmse_mz:.2f} N*m")

    # Guardar parámetros ajustados
    os.makedirs("config", exist_ok=True)
    out_yaml = "config/pacejka_mf62_params.yaml"
    model.export_params_yaml(out_yaml)
    print(f"\nScaffolding base MF 6.2 completado. Parámetros exportados a: {out_yaml}")

if __name__ == "__main__":
    fit()