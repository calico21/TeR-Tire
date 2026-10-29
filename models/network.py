#!/usr/bin/env python3
"""
Red Neuronal Residual de 7 dimensiones:
Entradas (7): [alpha, alpha_dot, gamma, Fz, P, T_avg, Delta_T]
Salidas  (2): [Delta_Fy, Delta_Mz]
"""

import torch
import torch.nn as nn

class ResidualTireMLP(nn.Module):
    def __init__(self, in_features: int = 7, hidden_dim: int = 64, gate_floor: float = 0.0):
        super().__init__()
        # Normalización: [alpha, alpha_dot, gamma, Fz, P, T_avg, Delta_T]
        self.register_buffer("x_mean", torch.tensor([0.0, 0.0, 0.0, 800.0, 83.0, 62.0, 1.5], dtype=torch.float32))
        self.register_buffer("x_std", torch.tensor([0.15, 0.25, 0.06, 450.0, 20.0, 14.0, 8.0], dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )
        # Arranca exactamente en la línea base de Pacejka (delta = 0)
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

        # Salida: max * tanh(raw * out_scale / max). Cerca de 0 es lineal con pendiente out_scale;
        # max_delta actúa solo como guarda de extrapolación, no como límite de ajuste.
        self.out_scale_fy = 250.0  # N por unidad de salida
        self.out_scale_mz = 10.0   # N*m por unidad de salida
        self.max_delta_fy = 600.0  # N
        self.max_delta_mz = 30.0   # N*m
        self.gate_floor = gate_floor

    def forward(self, alpha: torch.Tensor, alpha_dot: torch.Tensor, gamma: torch.Tensor, 
                fz: torch.Tensor, pressure: torch.Tensor, temp_avg: torch.Tensor, delta_temp: torch.Tensor):
        inputs = torch.stack([alpha, alpha_dot, gamma, fz, pressure, temp_avg, delta_temp], dim=-1)
        x_norm = (inputs - self.x_mean) / (self.x_std + 1e-6)

        raw_out = self.net(x_norm)

        raw_delta_fy = self.max_delta_fy * torch.tanh(raw_out[..., 0] * self.out_scale_fy / self.max_delta_fy)
        raw_delta_mz = self.max_delta_mz * torch.tanh(raw_out[..., 1] * self.out_scale_mz / self.max_delta_mz)

        gate = torch.tanh(15.0 * torch.abs(alpha) + 5.0 * torch.abs(gamma) + 2.0 * torch.abs(alpha_dot))
        gate = self.gate_floor + (1.0 - self.gate_floor) * gate

        return raw_delta_fy * gate, raw_delta_mz * gate