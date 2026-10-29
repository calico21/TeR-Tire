#!/usr/bin/env python3
"""
Modelo Grey-Box Híbrido: Pacejka MF 6.2 + Red Residual de 7 variables (Térmica + Dinámica).
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn
import yaml
from physics.pacejka_baseline import PacejkaMF62
from models.network import ResidualTireMLP

class GreyBoxTireModel(nn.Module):
    def __init__(self, mf62_params_yaml: str = "config/pacejka_mf62_params.yaml"):
        super().__init__()
        self.baseline = PacejkaMF62()
        self._load_baseline_params(mf62_params_yaml)

        for param in self.baseline.parameters():
            param.requires_grad = False

        self.residual = ResidualTireMLP(in_features=7)

    def _load_baseline_params(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            params = yaml.safe_load(f)
        for name, value in params.items():
            if hasattr(self.baseline, name):
                getattr(self.baseline, name).data.fill_(value)

    def forward(self, alpha: torch.Tensor, alpha_dot: torch.Tensor, gamma: torch.Tensor, 
                fz: torch.Tensor, pressure: torch.Tensor, temp_avg: torch.Tensor, delta_temp: torch.Tensor):
        fy_base, mz_base = self.baseline(alpha, gamma, fz, pressure)
        delta_fy, delta_mz = self.residual(alpha, alpha_dot, gamma, fz, pressure, temp_avg, delta_temp)

        fy_total = fy_base + delta_fy
        mz_total = mz_base + delta_mz

        return fy_total, mz_total, fy_base, mz_base

    def export_residual_weights(self, file_path: str):
        torch.save(self.residual.state_dict(), file_path)