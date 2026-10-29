#!/usr/bin/env python3
"""
Pérdida informada por la física (PINN) adimensionalizada para el residuo del neumático.
"""

import torch
import torch.nn as nn

class PINNTireLoss(nn.Module):
    def __init__(self, lambda_data: float = 1.0, lambda_passivity: float = 0.5, lambda_reg: float = 1e-4):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_passivity = lambda_passivity
        self.lambda_reg = lambda_reg
        self.huber = nn.HuberLoss(delta=5.0)  # casi MSE, alineado con la métrica RMSE

        # Escalas de normalización de fuerzas para mantener la pérdida en O(1)
        self.fy_scale = 2500.0  # N
        self.mz_scale = 40.0    # N*m
        # Escalas del término de datos = RMSE de Pacejka, para que Fy y Mz pesen igual
        self.fy_data_scale = 70.0  # N
        self.mz_data_scale = 8.3   # N*m

    def forward(self, delta_fy: torch.Tensor, delta_mz: torch.Tensor,
                res_fy_target: torch.Tensor, res_mz_target: torch.Tensor,
                fy_total: torch.Tensor, alpha: torch.Tensor, gamma: torch.Tensor):
        
        # 1. Pérdida sobre los residuos empíricos (adimensionalizada)
        l_res_fy = self.huber(delta_fy / self.fy_data_scale, res_fy_target / self.fy_data_scale)
        l_res_mz = self.huber(delta_mz / self.mz_data_scale, res_mz_target / self.mz_data_scale)
        loss_data = l_res_fy + l_res_mz

        # 2. Pasividad energética neta (solo fuera de la zona de transición de camber)
        # Cuando gamma ~ 0 y |alpha| > 1 deg, Fy y alpha deben tener el mismo signo en ISO
        zero_camber_mask = (torch.abs(gamma) < 0.015) & (torch.abs(alpha) > 0.02)
        if torch.any(zero_camber_mask):
            fy_sub = fy_total[zero_camber_mask] / self.fy_scale
            alpha_sub = alpha[zero_camber_mask]
            # Si fy * alpha < 0, la potencia disipada es negativa (generación espuria)
            energy_violation = torch.relu(-(fy_sub * torch.sign(alpha_sub)))
            loss_passivity = torch.mean(energy_violation ** 2)
        else:
            loss_passivity = torch.tensor(0.0, device=delta_fy.device)

        # 3. Regularización L2 del residuo (fuerza a que Delta -> 0 en zonas ambiguas)
        loss_reg = torch.mean((delta_fy / self.fy_scale) ** 2) + torch.mean((delta_mz / self.mz_scale) ** 2)

        total_loss = (self.lambda_data * loss_data +
                      self.lambda_passivity * loss_passivity +
                      self.lambda_reg * loss_reg)

        return total_loss, loss_data, loss_passivity