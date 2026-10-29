#!/usr/bin/env python3
"""
Parser y PyTorch Dataset para datos Calspan TTC Round 9 (.dat ASCII SI).
Incluye gradiente térmico lateral (TSTI, TSTC, TSTO) y tasa de deriva temporal (alpha_dot).
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import yaml

class CalspanTireDataset(Dataset):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        """
        x_data: [alpha, alpha_dot, gamma, fz, pressure, temp_avg, delta_temp, kappa]
        y_data: [fx, fy, mz]
        """
        self.x = torch.tensor(x_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TTCLoader:
    def __init__(self, config_path: str = "config/ttc_dataset.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

    def parse_dat_file(self, file_path: str, target_tire: str = "16.0X7.5-10", target_rim_width: float = 8.0) -> pd.DataFrame:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline()
            lines = [first_line] + f.readlines()

        if target_tire and target_tire not in first_line:
            return pd.DataFrame()
        if target_rim_width is not None:
            rim_str = f"Rim_Width={target_rim_width:.1f}"
            rim_alt = f"Rim_Width={int(target_rim_width)}"
            if (rim_str not in first_line) and (rim_alt not in first_line):
                return pd.DataFrame()

        header_idx = -1
        for i, line in enumerate(lines):
            if "ET" in line and ("SA" in line or "FY" in line):
                header_idx = i
                break

        if header_idx == -1:
            return pd.DataFrame()

        df = pd.read_csv(
            file_path,
            skiprows=[j for j in range(header_idx)] + [header_idx + 1],
            sep="\t",
            engine="python",
            on_bad_lines="skip"
        )

        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(how="all")
        return df

    def clean_and_convert(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        c = self.cfg["channels"]
        conv = self.cfg["conversions"]["sae_to_iso"]
        filt = self.cfg["filters"]

        required = [c["slip_angle"], c["fz"], c["fy"], c["time"]]
        for req in required:
            if req not in df.columns:
                return pd.DataFrame()

        df = df.dropna(subset=required).copy()

        if c["velocity"] in df.columns:
            v_val = df[c["velocity"]].abs()
            df = df[v_val >= filt["min_velocity_kmh"]]

        # Convenio SAE -> ISO
        raw_fz = df[c["fz"]].to_numpy()
        fz_iso = -raw_fz if conv["invert_fz"] else raw_fz

        raw_fy = df[c["fy"]].to_numpy()
        fy_iso = -raw_fy if conv["invert_fy"] else raw_fy

        raw_fx = df[c["fx"]].to_numpy() if c["fx"] in df.columns else np.zeros_like(fy_iso)
        fx_iso = raw_fx

        raw_mz = df[c["mz"]].to_numpy() if c["mz"] in df.columns else np.zeros_like(fy_iso)
        mz_iso = -raw_mz if conv["invert_mz"] else raw_mz

        sa_deg = df[c["slip_angle"]].to_numpy()
        alpha_rad = np.deg2rad(sa_deg) if conv["slip_angle_to_rad"] else sa_deg

        ia_deg = df[c["camber"]].to_numpy() if c["camber"] in df.columns else np.zeros_like(sa_deg)
        gamma_rad = np.deg2rad(ia_deg) if conv["camber_to_rad"] else ia_deg

        # Tasa de variación del slip angle (alpha_dot) en rad/s
        time_s = df[c["time"]].to_numpy()
        dt = np.gradient(time_s)
        dt[dt <= 0.0] = 0.01  # Prevenir divisiones por cero en discontinuidades
        alpha_dot = np.gradient(alpha_rad) / dt
        alpha_dot = np.clip(alpha_dot, -2.0, 2.0)

        if "SL" in df.columns and not df["SL"].isna().all():
            kappa = df["SL"].to_numpy()
        elif "SR" in df.columns and not df["SR"].isna().all():
            kappa = df["SR"].to_numpy()
        else:
            kappa = np.zeros_like(sa_deg)

        if np.nanmax(np.abs(kappa)) > 1.5:
            kappa = kappa / 100.0

        p_kpa = df[c["pressure"]].to_numpy() if c["pressure"] in df.columns else np.full_like(sa_deg, 82.7)

        # Gradiente térmico lateral
        t_in = df["TSTI"].to_numpy() if "TSTI" in df.columns else np.full_like(sa_deg, 60.0)
        t_cen = df["TSTC"].to_numpy() if "TSTC" in df.columns else np.full_like(sa_deg, 60.0)
        t_out = df["TSTO"].to_numpy() if "TSTO" in df.columns else np.full_like(sa_deg, 60.0)

        temp_avg = (t_in + t_cen + t_out) / 3.0
        delta_temp = t_in - t_out

        cleaned = pd.DataFrame({
            "alpha": alpha_rad,
            "alpha_dot": alpha_dot,
            "gamma": gamma_rad,
            "fz": fz_iso,
            "pressure": p_kpa,
            "temp_avg": temp_avg,
            "delta_temp": delta_temp,
            "kappa": kappa,
            "fx": fx_iso,
            "fy": fy_iso,
            "mz": mz_iso
        })

        cleaned = cleaned[
            (cleaned["fz"] >= filt["fz_min_threshold_n"]) &
            (cleaned["fz"] <= filt["fz_max_threshold_n"]) &
            (np.abs(np.rad2deg(cleaned["alpha"])) <= filt["slip_angle_max_deg"])
        ]

        return cleaned

    def load_specific_runs(self, dir_path: str, run_numbers: list = [8, 9]) -> CalspanTireDataset:
        frames = []
        for run_num in run_numbers:
            pattern = os.path.join(dir_path, f"*run{run_num}.dat")
            matches = glob.glob(pattern)
            if not matches:
                pattern = os.path.join(dir_path, f"*run{run_num:02d}.dat")
                matches = glob.glob(pattern)

            if not matches:
                continue

            file_path = matches[0]
            raw_df = self.parse_dat_file(file_path, target_tire="16.0X7.5-10", target_rim_width=8.0)
            clean_df = self.clean_and_convert(raw_df)
            if not clean_df.empty:
                frames.append(clean_df)

        if not frames:
            raise RuntimeError(f"No se pudieron cargar datos válidos de los runs {run_numbers} en {dir_path}")

        full_df = pd.concat(frames, ignore_index=True)

        x_cols = ["alpha", "alpha_dot", "gamma", "fz", "pressure", "temp_avg", "delta_temp", "kappa"]
        y_cols = ["fx", "fy", "mz"]

        return CalspanTireDataset(full_df[x_cols].to_numpy(), full_df[y_cols].to_numpy())