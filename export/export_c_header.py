#!/usr/bin/env python3
"""
Exportador de pesos PINN (7 -> 64 -> 64 -> 2) y coeficientes Pacejka MF 6.2
a estructuras estáticas en C++ (header-only) para Mirena / GDExtension.
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import yaml
import torch

def export_to_header(
    weights_path="export/weights/pinn_residual_best.pth",
    params_path="config/pacejka_mf62_params.yaml",
    output_path="godot_integration/cpp_extension/pinn_tire_weights.h"
):
    print(f"Cargando pesos desde: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")

    print(f"Cargando parámetros Pacejka desde: {params_path}")
    with open(params_path, "r") as f:
        pacejka_params = yaml.safe_load(f)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extraer pesos de la red residual (MLP 7 -> 64 -> 64 -> 2)
    w0 = state_dict["net.0.weight"].numpy()  # [64, 7]
    b0 = state_dict["net.0.bias"].numpy()    # [64]
    w2 = state_dict["net.2.weight"].numpy()  # [64, 64]
    b2 = state_dict["net.2.bias"].numpy()    # [64]
    w4 = state_dict["net.4.weight"].numpy()  # [2, 64]
    b4 = state_dict["net.4.bias"].numpy()    # [2]

    x_mean = state_dict["x_mean"].numpy()    # [7]
    x_std = state_dict["x_std"].numpy()      # [7]

    with open(output_path, "w") as f:
        f.write("// ============================================================================\n")
        f.write("// ARCHIVO AUTOGENERADO POR export_c_header.py - NO EDITAR MANUALMENTE\n")
        f.write("// Modelo Hibrido Grey-Box PINN: Pacejka MF 6.2 + Residual MLP (Hoosier R20 8in)\n")
        f.write("// ============================================================================\n\n")
        f.write("#pragma once\n\n")
        f.write("namespace Mirena::Physics::Tire {\n\n")

        # Constantes de normalización
        f.write("// --- Factores de Normalización de Entrada (7 dimensiones) ---\n")
        f.write(f"inline constexpr float X_MEAN[7] = {{{', '.join(f'{v:.8e}f' for v in x_mean)}}};\n")
        f.write(f"inline constexpr float X_STD[7]  = {{{', '.join(f'{v:.8e}f' for v in x_std)}}};\n\n")

        # Escalas de salida
        f.write("// --- Cotas Residuales y Escalas de Salida ---\n")
        f.write("inline constexpr float OUT_SCALE_FY = 250.0f;\n")
        f.write("inline constexpr float OUT_SCALE_MZ = 10.0f;\n")
        f.write("inline constexpr float MAX_DELTA_FY = 600.0f;\n")
        f.write("inline constexpr float MAX_DELTA_MZ = 30.0f;\n\n")

        # Capa 0: Linear(7, 64)
        f.write("// --- Capa 0: Pesos [64 x 7] y Sesgos [64] ---\n")
        f.write("alignas(32) inline constexpr float W0[64][7] = {\n")
        for row in w0:
            f.write(f"    {{{', '.join(f'{v:.8e}f' for v in row)}}},\n")
        f.write("};\n\n")
        f.write(f"alignas(32) inline constexpr float B0[64] = {{{', '.join(f'{v:.8e}f' for v in b0)}}};\n\n")

        # Capa 2: Linear(64, 64)
        f.write("// --- Capa 2: Pesos [64 x 64] y Sesgos [64] ---\n")
        f.write("alignas(32) inline constexpr float W2[64][64] = {\n")
        for row in w2:
            f.write(f"    {{{', '.join(f'{v:.8e}f' for v in row)}}},\n")
        f.write("};\n\n")
        f.write(f"alignas(32) inline constexpr float B2[64] = {{{', '.join(f'{v:.8e}f' for v in b2)}}};\n\n")

        # Capa 4: Linear(64, 2)
        f.write("// --- Capa 4: Pesos [2 x 64] y Sesgos [2] ---\n")
        f.write("alignas(32) inline constexpr float W4[2][64] = {\n")
        for row in w4:
            f.write(f"    {{{', '.join(f'{v:.8e}f' for v in row)}}},\n")
        f.write("};\n\n")
        f.write(f"alignas(32) inline constexpr float B4[2] = {{{', '.join(f'{v:.8e}f' for v in b4)}}};\n\n")

        # Coeficientes Pacejka MF 6.2
        f.write("// --- Coeficientes Pacejka MF 6.2 Baseline ---\n")
        f.write("struct PacejkaMF62Coeffs {\n")
        f.write("    float fz0 = 700.0f;\n")
        f.write("    float p0  = 82.7f;\n")
        f.write("    float r0  = 0.2032f;\n")
        for k, v in pacejka_params.items():
            f.write(f"    float {k} = {v:.8e}f;\n")
        f.write("};\n\n")
        f.write("inline constexpr PacejkaMF62Coeffs MF62_PARAMS{};\n\n")

        f.write("} // namespace Mirena::Physics::Tire\n")

    print(f"Header C++ exportado correctamente en: {output_path}")

if __name__ == "__main__":
    export_to_header()