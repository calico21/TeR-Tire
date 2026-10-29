#!/usr/bin/env python3
"""
Módulo de física analítica: Pacejka Magic Formula 6.2 en PyTorch.
Totalmente diferenciable por batches para entrenamiento híbrido PINN y ajuste no lineal.
"""

import torch
import torch.nn as nn
import yaml

class PacejkaMF62(nn.Module):
    def __init__(self, fz0: float = 700.0, p0: float = 82.7, r0: float = 0.2032):
        super().__init__()
        self.register_buffer("fz0", torch.tensor(fz0, dtype=torch.float32))
        self.register_buffer("p0", torch.tensor(p0, dtype=torch.float32))
        self.register_buffer("r0", torch.tensor(r0, dtype=torch.float32))
        self.eps = 1e-6

        # --- Parámetros de Fuerza Lateral (Fy0) ---
        self.p_Cy1 = nn.Parameter(torch.tensor(1.45, dtype=torch.float32))
        self.p_Dy1 = nn.Parameter(torch.tensor(2.60, dtype=torch.float32))
        self.p_Dy2 = nn.Parameter(torch.tensor(-0.40, dtype=torch.float32))
        self.p_Dy3 = nn.Parameter(torch.tensor(5.00, dtype=torch.float32))
        self.p_Ey1 = nn.Parameter(torch.tensor(-0.50, dtype=torch.float32))
        self.p_Ey2 = nn.Parameter(torch.tensor(-0.30, dtype=torch.float32))
        self.p_Ey3 = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.p_Ey4 = nn.Parameter(torch.tensor(2.00, dtype=torch.float32))
        self.p_Ky1 = nn.Parameter(torch.tensor(35.0, dtype=torch.float32))
        self.p_Ky2 = nn.Parameter(torch.tensor(1.80, dtype=torch.float32))
        self.p_Hy1 = nn.Parameter(torch.tensor(0.002, dtype=torch.float32))
        self.p_Hy2 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.p_Hy3 = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.p_Vy1 = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.p_Vy2 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.p_Vy3 = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.p_Vy4 = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self.p_py1 = nn.Parameter(torch.tensor(-0.20, dtype=torch.float32))
        self.p_py2 = nn.Parameter(torch.tensor(-0.10, dtype=torch.float32))

        # --- Parámetros de Momento Autoalineante (Mz0) ---
        self.q_Bz1 = nn.Parameter(torch.tensor(8.00, dtype=torch.float32))
        self.q_Bz2 = nn.Parameter(torch.tensor(-2.00, dtype=torch.float32))
        self.q_Bz3 = nn.Parameter(torch.tensor(1.00, dtype=torch.float32))
        self.q_Bz5 = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))
        self.q_Cz1 = nn.Parameter(torch.tensor(1.20, dtype=torch.float32))
        self.q_Dz1 = nn.Parameter(torch.tensor(0.12, dtype=torch.float32))
        self.q_Dz2 = nn.Parameter(torch.tensor(-0.03, dtype=torch.float32))
        self.q_Dz3 = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.q_Dz6 = nn.Parameter(torch.tensor(0.005, dtype=torch.float32))
        self.q_Dz7 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.q_Dz8 = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.q_Dz9 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.q_Ez1 = nn.Parameter(torch.tensor(-1.00, dtype=torch.float32))
        self.q_Ez2 = nn.Parameter(torch.tensor(-0.50, dtype=torch.float32))
        self.q_Ez4 = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))
        self.q_Ez5 = nn.Parameter(torch.tensor(1.00, dtype=torch.float32))
        self.q_Hz1 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.q_Hz2 = nn.Parameter(torch.tensor(0.001, dtype=torch.float32))
        self.q_Hz3 = nn.Parameter(torch.tensor(0.02, dtype=torch.float32))
        self.q_Hz4 = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.q_pz1 = nn.Parameter(torch.tensor(-0.15, dtype=torch.float32))

    def forward(self, alpha: torch.Tensor, gamma: torch.Tensor, fz: torch.Tensor, pressure: torch.Tensor):
        """
        Inputs en tensores PyTorch:
          alpha: Ángulo de deriva [rad]
          gamma: Ángulo de caída [rad]
          fz: Carga normal [N] (ISO: positivo hacia arriba)
          pressure: Presión de inflado [kPa]
        Retorna:
          fy0: Fuerza lateral [N]
          mz0: Momento autoalineante [N*m]
        """
        dfz = (fz - self.fz0) / (self.fz0 + self.eps)
        dpi = (pressure - self.p0) / (self.p0 + self.eps)

        # 1. Fuerza Lateral Pura (Fy0)
        mu_y = (self.p_Dy1 + self.p_Dy2 * dfz) * (1.0 + self.p_py1 * dpi) * (1.0 - self.p_Dy3 * gamma**2)
        Dy = mu_y * fz
        Cy = self.p_Cy1

        # Rigidez de deriva inicial Kya0
        Kya0 = self.p_Ky1 * self.fz0 * torch.sin(2.0 * torch.atan(fz / (self.p_Ky2 * self.fz0 + self.eps))) * (1.0 + self.p_py2 * dpi)
        By = Kya0 / (Cy * Dy + self.eps)

        S_Hy = self.p_Hy1 + self.p_Hy2 * dfz + self.p_Hy3 * gamma
        S_Vy = fz * ((self.p_Vy1 + self.p_Vy2 * dfz) + (self.p_Vy3 + self.p_Vy4 * dfz) * gamma)
        alpha_y = alpha + S_Hy

        Ey = torch.tanh(self.p_Ey1 + self.p_Ey2 * dfz) * (1.0 - (self.p_Ey3 + self.p_Ey4 * gamma) * torch.sign(alpha_y))

        # Ecuación de Pacejka para Fy0
        B_ay = By * alpha_y
        atan_B_ay = torch.atan(B_ay)
        inside_sin = Cy * torch.atan(B_ay - Ey * (B_ay - atan_B_ay))
        Fy0 = Dy * torch.sin(inside_sin) + S_Vy

        # 2. Momento Autoalineante Puro (Mz0)
        Fy0_prime = Fy0 - S_Vy  # Fuerza lateral sin desfase vertical

        alpha_t = alpha + self.q_Hz1 + self.q_Hz2 * dfz + (self.q_Hz3 + self.q_Hz4 * dfz) * gamma
        Dt = fz * (self.r0 / self.fz0) * (self.q_Dz1 + self.q_Dz2 * dfz) * (1.0 + self.q_pz1 * dpi) * (1.0 - self.q_Dz3 * torch.abs(gamma))
        Ct = self.q_Cz1

        Bt = (self.q_Bz1 + self.q_Bz2 * dfz + self.q_Bz3 * dfz**2) * (1.0 + self.q_Bz5 * torch.abs(gamma)) * (self.r0 / (self.fz0 * Dt + self.eps))
        Et = torch.tanh(self.q_Ez1 + self.q_Ez2 * dfz) * (1.0 + (self.q_Ez4 + self.q_Ez5 * gamma) * torch.sign(alpha_t))

        B_at = Bt * alpha_t
        t = Dt * torch.cos(Ct * torch.atan(B_at - Et * (B_at - torch.atan(B_at)))) * torch.cos(alpha)

        # Momento residual Mzr
        alpha_r = alpha + S_Hy + (S_Vy / (Kya0 + self.eps))
        Dr = fz * self.r0 * ((self.q_Dz6 + self.q_Dz7 * dfz) + (self.q_Dz8 + self.q_Dz9 * dfz) * gamma)
        Br = self.q_Bz9 * (self.r0 / self.fz0) if hasattr(self, 'q_Bz9') else (self.r0 / self.fz0)
        Cr = 1.0
        Mzr = Dr * torch.cos(Cr * torch.atan(Br * alpha_r))

        Mz0 = -t * Fy0_prime + Mzr
        return Fy0, Mz0

    def export_params_yaml(self, file_path: str):
        params_dict = {name: float(param.detach().cpu().item()) for name, param in self.named_parameters()}
        with open(file_path, "w") as f:
            yaml.dump(params_dict, f, default_flow_style=False)