#!/usr/bin/env python3
"""
Visualizador PINN Grey-Box (7 -> 64 -> 64 -> 2) — entrenamiento en vivo.

Qué se ve:
  - Red: una columna vertical por capa. Un "probe" (α barriendo ±12° en senoidal) atraviesa la red
    y es el que enciende las neuronas -> la red responde a una entrada concreta, no a una media de batch.
    El brillo de cada sinapsis (escala de grises) = |w_ij| · activación_i (flujo de señal real).
    Borde de neurona: cian = activación positiva, magenta = negativa. Destello blanco = cambio brusco.
  - Curvas Fy(α) y Mz(α): TTC vs Pacejka base vs PINN, actualizadas en vivo, con área de residuo
    sombreada, marcador del probe y estela fantasma de épocas anteriores.
  - Métricas: RMSE val de Fy y Mz normalizados a Pacejka (100 %), loss (log) y norma de gradiente.

Supuestos (documentados):
  - model.residual.net == [Linear, SiLU, Linear, SiLU, Linear] (mismos índices de hook que el original).
  - Columnas de entrada: [α, α̇, γ, Fz, P, Tavg, ΔT]; y[:,1] = Fy, y[:,2] = Mz.
  - Barrido nominal: α̇=0, γ=0, Fz=700 N, P=82.7, Tavg=60, ΔT=0 (idéntico al original).
  - La entrada de la red se colorea por z-score (media/σ del set de train) sólo para display.
  - PROPAGATION_LAG desfasa visualmente capas sucesivas (efecto de pulso); es cosmético. 0 = desactivado.
  - Entrenamiento idéntico a 02_train_pinn.py (split por bloques, AdamW, OneCycleLR, clip 1.0).

Uso:
  python visualize_pinn_training.py [--epochs 180] [--fps 24] [--probe-hz 0.35] [--record out.mp4]
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from torch.utils.data import DataLoader, TensorDataset

from data.loaders import TTCLoader
from models.grey_box_tire import GreyBoxTireModel
from models.physics_loss import PINNTireLoss

BG, PANEL, GRID = "#06070a", "#0b0d14", "#4a5568"
CYAN, MINT, RED, AMBER, MUTED = "#66fcf1", "#00ffcc", "#ff0055", "#ffb86c", "#7f8c8d"

N_SWEEP = 160
SWEEP_MAX_DEG = 12.0
PROPAGATION_LAG = 2  # frames de desfase entre capas consecutivas (cosmético)
NOMINAL = dict(fz=700.0, p=82.7, tavg=60.0)


class NeuralVisualizer:
    KEEP_FRAC = (0.30, 0.18, 0.60)  # fracción de sinapsis estructuralmente visibles por capa

    def __init__(self, layer_sizes=(7, 64, 64, 2), flash_decay=0.58, flash_gain=2.4,
                 lag_frames=PROPAGATION_LAG):
        self.sizes = tuple(layer_sizes)
        self.n = len(self.sizes)
        self.x_coords = np.linspace(0.0, 4.05, self.n)
        self.base_s = [130 if s <= 7 else 22 for s in self.sizes]
        self.flash_decay, self.flash_gain, self.lag = flash_decay, flash_gain, lag_frames

        self.pos = []
        for x, s in zip(self.x_coords, self.sizes):
            y_max = 1.05 if s <= 7 else 1.42
            self.pos.append(np.column_stack([np.full(s, x), np.linspace(-y_max, y_max, s)]))

        # Orden src-major: idx = i*n_dst + j, coherente con w.T.ravel() (shape in×out)
        self.segments = []
        for src, dst in zip(self.pos[:-1], self.pos[1:]):
            seg = np.empty((len(src), len(dst), 2, 2))
            seg[:, :, 0] = src[:, None, :]
            seg[:, :, 1] = dst[None, :, :]
            self.segments.append(seg.reshape(-1, 2, 2))

        self.level = [np.zeros(s) for s in self.sizes]
        self.prev = [np.zeros(s) for s in self.sizes]
        self.flash = [np.zeros(s) for s in self.sizes]
        self.scale = [1e-6] * self.n          # techo EMA por capa: evita parpadeo por renormalización
        self.flow_scale = [1e-9] * (self.n - 1)
        self.history = deque(maxlen=self.lag * (self.n - 1) + 1)

        self.cmap = LinearSegmentedColormap.from_list(
            "neuron_flash", ["#0c101c", "#0044cc", "#00ffcc", "#ffffff"], N=256)
        self._pos_rgba = np.array(to_rgba(CYAN))
        self._neg_rgba = np.array(to_rgba(RED))

    def setup_axis(self, ax):
        ax.set_title("Topología Neuronal (probe α → pulsos de activación)", fontsize=11,
                     color=MINT, pad=12, fontweight="bold")
        ax.set_facecolor(BG)
        ax.set_xlim(-0.85, 4.85)
        ax.set_ylim(-1.65, 1.75)
        ax.axis("off")

        self.syn = []
        for segs in self.segments:
            lc = LineCollection(segs, cmap=plt.cm.gray, zorder=1)
            lc.set_clim(0.0, 1.0)
            lc.set_array(np.full(len(segs), 0.03))
            lc.set_linewidth(np.zeros(len(segs)))
            ax.add_collection(lc)
            self.syn.append(lc)

        self.glow, self.core = [], []
        for l, xy in enumerate(self.pos):
            z = np.zeros(len(xy))
            s = self.base_s[l]
            self.glow.append(ax.scatter(xy[:, 0], xy[:, 1], s=s * 2.8, c=z, cmap=self.cmap,
                                        vmin=0, vmax=1, alpha=0.35, zorder=2))
            self.core.append(ax.scatter(xy[:, 0], xy[:, 1], s=s, c=z, cmap=self.cmap, vmin=0, vmax=1,
                                        edgecolors="#222f44", linewidth=0.9, zorder=3))

        labels_in = [r"$\alpha$", r"$\dot{\alpha}$", r"$\gamma$", r"$F_z$", r"$P$",
                     r"$T_{\mathrm{avg}}$", r"$\Delta T$"]
        labels_out = [r"$\Delta F_y$", r"$\Delta M_z$"]
        for (x, y), t in zip(self.pos[0], labels_in):
            ax.text(x - 0.12, y, t, color=CYAN, fontsize=9.5, ha="right", va="center", fontweight="bold")
        for (x, y), t in zip(self.pos[-1], labels_out):
            ax.text(x + 0.14, y, t, color=RED, fontsize=10, ha="left", va="center", fontweight="bold")
        for x, name in zip(self.x_coords, ["Entrada (7)", "Oculta 1 (64)", "Oculta 2 (64)", "Salida (2)"]):
            ax.text(x, -1.55, name, color=MUTED, fontsize=8.5, ha="center")
        self.probe_label = ax.text(-0.8, 1.62, "", color="white", fontsize=9.5, family="monospace", va="center")

    def update(self, weights, acts):
        """weights: [W1(64×7), W2(64×64), W3(2×64)]; acts: activaciones con signo por capa (probe)."""
        self.history.append([np.asarray(a, dtype=np.float32) for a in acts])
        h = len(self.history)

        for l in range(self.n):
            raw = self.history[max(0, h - 1 - l * self.lag)][l]
            mag = np.abs(raw)
            self.scale[l] = max(self.scale[l] * 0.985, float(mag.max()), 1e-6)
            norm = (mag / self.scale[l]) ** 0.7
            delta = np.abs(norm - self.prev[l])
            self.prev[l] = norm
            self.flash[l] = self.flash[l] * self.flash_decay + self.flash_gain * delta
            lvl = np.clip(0.75 * norm + self.flash[l], 0.0, 1.0)
            self.level[l] = lvl

            sizes = self.base_s[l] * (0.8 + 1.4 * lvl)
            edge = np.where((raw >= 0)[:, None], self._pos_rgba, self._neg_rgba)
            self.core[l].set_array(lvl)
            self.core[l].set_sizes(sizes)
            self.core[l].set_edgecolors(edge)
            self.glow[l].set_array(lvl)
            self.glow[l].set_sizes(sizes * 2.8)

        for l, w in enumerate(weights):
            wabs = np.abs(w.T)                                    # (in, out)
            flow = wabs * self.level[l][:, None]                  # flujo de señal por sinapsis
            self.flow_scale[l] = max(self.flow_scale[l] * 0.985, float(flow.max()), 1e-9)
            fn = np.clip(flow / self.flow_scale[l], 0.0, 1.0)
            mask = wabs >= np.quantile(wabs, 1.0 - self.KEEP_FRAC[l])
            gray = np.where(mask, 0.10 + 0.90 * fn ** 0.75, 0.035)
            widths = np.where(mask, 0.15 + 1.5 * fn, 0.0)
            self.syn[l].set_array(gray.ravel())
            self.syn[l].set_linewidth(widths.ravel())


class _WindowClosed(Exception):
    pass


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=10.5, color=MINT, pad=8, fontweight="bold")
    ax.set_facecolor(PANEL)
    ax.grid(True, linestyle="--", alpha=0.2, color=GRID)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9.5, color="white")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9.5, color="white")


def _pad_lim(*arrays, frac=0.08):
    v = np.concatenate([np.ravel(a) for a in arrays])
    lo, hi = float(v.min()), float(v.max())
    pad = (hi - lo) * frac + 1e-9
    return lo - pad, hi + pad


def run_visual_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Visualizador PINN Grey-Box — {device} ===")

    loader = TTCLoader("config/ttc_dataset.yaml")
    raw = loader.load_specific_runs("data/raw_ttc/RunData_Cornering_ASCII_SI_Round9", run_numbers=[8, 9])
    x_all, y_all = raw.x.to(device), raw.y.to(device)

    model = GreyBoxTireModel("config/pacejka_mf62_params.yaml").to(device)
    model.eval()

    with torch.no_grad():
        fy_b, mz_b = model.baseline(x_all[:, 0], x_all[:, 2], x_all[:, 3], x_all[:, 4])
        res_fy, res_mz = y_all[:, 1] - fy_b, y_all[:, 2] - mz_b

    inputs = x_all[:, :7].contiguous()
    targets = torch.stack([res_fy, res_mz, fy_b, mz_b], dim=-1)
    n = len(inputs)

    # Validación por bloques contiguos (idéntica a 02_train_pinn.py)
    BLOCK, VAL_FRAC = 256, 0.15
    n_blocks = math.ceil(n / BLOCK)
    g = torch.Generator().manual_seed(0)
    val_blocks = torch.randperm(n_blocks, generator=g)[:max(1, int(VAL_FRAC * n_blocks))]
    val_mask = torch.isin(torch.arange(n) // BLOCK, val_blocks).to(device)

    x_tr, t_tr = inputs[~val_mask], targets[~val_mask]
    x_va, t_va = inputs[val_mask], targets[val_mask]
    base_val_fy = torch.sqrt(torch.mean(t_va[:, 0] ** 2)).item()
    base_val_mz = torch.sqrt(torch.mean(t_va[:, 1] ** 2)).item()
    feat_mu, feat_sd = x_tr.mean(0), x_tr.std(0).clamp_min(1e-6)

    train_loader = DataLoader(TensorDataset(x_tr, t_tr), batch_size=1024, shuffle=True)

    # Hooks activos sólo durante el forward del probe (sin contaminar con train/val)
    cap_on = {"v": False}
    captured: dict[str, np.ndarray] = {}

    def make_hook(name):
        def hook(_m, _i, out):
            if cap_on["v"]:
                captured[name] = out[-1].detach().float().cpu().numpy()  # última fila = probe
        return hook

    net = model.residual.net
    net[1].register_forward_hook(make_hook("l1"))
    net[3].register_forward_hook(make_hook("l2"))
    net[4].register_forward_hook(make_hook("out"))
    lin = [net[0], net[2], net[4]]

    # Barrido nominal + 1 fila extra para el probe (un único forward sirve curva y activaciones)
    alpha_deg = np.linspace(-SWEEP_MAX_DEG, SWEEP_MAX_DEG, N_SWEEP)
    sweep_in = torch.zeros(N_SWEEP + 1, 7, device=device)
    sweep_in[:N_SWEEP, 0] = torch.tensor(np.deg2rad(alpha_deg), dtype=torch.float32, device=device)
    sweep_in[:, 3], sweep_in[:, 4], sweep_in[:, 5] = NOMINAL["fz"], NOMINAL["p"], NOMINAL["tavg"]

    with torch.no_grad():
        fy_base_t, mz_base_t = model.baseline(sweep_in[:N_SWEEP, 0], sweep_in[:N_SWEEP, 2],
                                              sweep_in[:N_SWEEP, 3], sweep_in[:N_SWEEP, 4])
    fy_base, mz_base = fy_base_t.cpu().numpy(), mz_base_t.cpu().numpy()

    @torch.no_grad()
    def probe_forward(alpha_rad):
        sweep_in[-1, 0] = alpha_rad
        was_training = model.residual.training
        model.residual.eval()
        cap_on["v"] = True
        d_fy, d_mz = model.residual(*sweep_in.unbind(-1))
        cap_on["v"] = False
        if was_training:
            model.residual.train()
        z = ((sweep_in[-1] - feat_mu) / feat_sd).cpu().numpy()
        return d_fy.cpu().numpy(), d_mz.cpu().numpy(), z

    # Datos TTC de la rebanada nominal
    xs = raw.x.numpy()
    sl = (np.abs(xs[:, 2]) < 0.015) & (xs[:, 3] >= 550.0) & (xs[:, 3] <= 850.0)
    a_ttc = np.rad2deg(xs[sl, 0])
    fy_ttc, mz_ttc = raw.y[sl, 1].numpy(), raw.y[sl, 2].numpy()

    # ------------------------------------------------------------------ lienzo
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 9), facecolor=BG)
    gs = gridspec.GridSpec(3, 3, figure=fig, width_ratios=[1.25, 1.2, 0.95], height_ratios=[1, 1, 1],
                           left=0.015, right=0.985, top=0.94, bottom=0.07, wspace=0.17, hspace=0.50)
    ax_net = fig.add_subplot(gs[:, 0])
    ax_fy = fig.add_subplot(gs[0:2, 1])
    ax_mz = fig.add_subplot(gs[2, 1], sharex=ax_fy)
    ax_met = fig.add_subplot(gs[0, 2])
    ax_loss = fig.add_subplot(gs[1, 2])
    ax_hud = fig.add_subplot(gs[2, 2])

    vis = NeuralVisualizer((7, 64, 64, 2))
    vis.setup_axis(ax_net)

    _style(ax_fy, r"Curva Lateral $F_y$ vs $\alpha$  ($F_z=700$ N, $\gamma=0^\circ$)", None, r"$F_y$ [N]")
    ax_fy.tick_params(labelbottom=False)
    ax_fy.scatter(a_ttc, fy_ttc, color="#45a29e", alpha=0.25, s=7, label="Calspan TTC")
    ax_fy.plot(alpha_deg, fy_base, color=CYAN, ls="--", lw=1.5, label="Pacejka MF 6.2")
    (ln_fy,) = ax_fy.plot(alpha_deg, fy_base, color=RED, lw=2.5, label="PINN híbrido", zorder=5)
    ax_fy.set_xlim(-SWEEP_MAX_DEG, SWEEP_MAX_DEG)
    ax_fy.set_ylim(*_pad_lim(fy_ttc, fy_base, frac=0.12))
    vl_fy = ax_fy.axvline(0.0, color="white", lw=0.8, alpha=0.35, zorder=4)
    (dot_fy,) = ax_fy.plot([0], [0], "o", ms=9, mfc="white", mec=RED, mew=2, zorder=7)
    ax_fy.legend(loc="upper left", framealpha=0.3, fontsize=8.5)

    _style(ax_mz, r"Momento Autoalineante $M_z$ vs $\alpha$", r"Slip Angle $\alpha$ [deg]", r"$M_z$ [N·m]")
    ax_mz.scatter(a_ttc, mz_ttc, color="#45a29e", alpha=0.25, s=7)
    ax_mz.plot(alpha_deg, mz_base, color=CYAN, ls="--", lw=1.4)
    (ln_mz,) = ax_mz.plot(alpha_deg, mz_base, color=RED, lw=2.2, zorder=5)
    ax_mz.set_ylim(*_pad_lim(mz_ttc, mz_base, frac=0.12))
    ax_mz.axvline(0.0, color="white", lw=0.8, alpha=0.2)
    (dot_mz,) = ax_mz.plot([0], [0], "o", ms=8, mfc="white", mec=RED, mew=2, zorder=7)

    EPOCHS = args.epochs
    _style(ax_met, "Validación vs Pacejka (100 % = base)", None, "RMSE / base [%]")
    ax_met.axhline(100.0, color=CYAN, ls="--", lw=1.2, label=f"Pacejka Fy {base_val_fy:.1f} N | Mz {base_val_mz:.2f} N·m")
    (ln_vfy,) = ax_met.plot([], [], color=RED, lw=2.0, label="Fy val")
    (ln_vmz,) = ax_met.plot([], [], color=AMBER, lw=2.0, label="Mz val")
    ax_met.set_xlim(1, EPOCHS)
    ax_met.legend(loc="upper right", framealpha=0.3, fontsize=7.5)

    _style(ax_loss, "Loss (log) y ‖∇‖ pre-clip", "Época")
    ax_loss.set_yscale("log")
    (ln_loss,) = ax_loss.plot([], [], color=MINT, lw=2.0, label="loss")
    ax_loss.set_xlim(1, EPOCHS)
    ax_gn = ax_loss.twinx()
    ax_gn.set_yscale("log")
    (ln_gn,) = ax_gn.plot([], [], color=AMBER, lw=1.4, alpha=0.85, label="‖∇‖")
    ax_gn.tick_params(colors=AMBER, labelsize=8)
    ax_loss.legend(handles=[ln_loss, ln_gn], loc="upper right", framealpha=0.3, fontsize=7.5)

    ax_hud.set_facecolor(PANEL)
    ax_hud.axis("off")
    hud_text = ax_hud.text(0.02, 0.5, "", color="white", fontsize=9.2, family="monospace", va="center")

    # ------------------------------------------------------------------ entrenamiento
    criterion = PINNTireLoss(lambda_data=1.0, lambda_passivity=0.5, lambda_reg=1e-4)
    optimizer = optim.AdamW(model.residual.parameters(), lr=3e-3, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(x_tr) / 1024)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos", final_div_factor=100.0)

    hist = dict(ep=[], vfy=[], vmz=[], loss=[], gn=[])
    ghosts: deque = deque(maxlen=14)
    fills: list = []
    st = dict(epoch=0, loss=float("nan"), val_fy=base_val_fy, val_mz=base_val_mz,
              best_ep=0, best_val=base_val_fy, gn=0.0, dfy_max=0.0, sps=0.0, eta=0.0)

    plt.ion()
    plt.show(block=False)

    stack = contextlib.ExitStack()
    writer = None
    if args.record:
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=args.fps, bitrate=8000)
        stack.enter_context(writer.saving(fig, args.record, dpi=110))

    t0 = time.perf_counter()
    interval = 1.0 / args.fps
    last_draw = 0.0
    frame = 0
    n_steps_total = 0
    last_gn = torch.zeros((), device=device)

    def draw(final_epoch_state=False):
        nonlocal last_draw, frame
        if not plt.fignum_exists(fig.number):
            raise _WindowClosed
        t = (frame / args.fps) if writer else (time.perf_counter() - t0)
        a_deg = SWEEP_MAX_DEG * math.sin(2.0 * math.pi * args.probe_hz * t)
        d_fy, d_mz, z = probe_forward(math.radians(a_deg))

        fy_pinn, mz_pinn = fy_base + d_fy[:N_SWEEP], mz_base + d_mz[:N_SWEEP]
        fy_p = float(np.interp(a_deg, alpha_deg, fy_base) + d_fy[N_SWEEP])
        mz_p = float(np.interp(a_deg, alpha_deg, mz_base) + d_mz[N_SWEEP])

        ln_fy.set_ydata(fy_pinn)
        ln_mz.set_ydata(mz_pinn)
        for c in fills:
            c.remove()
        fills[:] = [
            ax_fy.fill_between(alpha_deg, fy_base, fy_pinn, color=RED, alpha=0.15, lw=0, zorder=3),
            ax_mz.fill_between(alpha_deg, mz_base, mz_pinn, color=RED, alpha=0.15, lw=0, zorder=3),
        ]
        vl_fy.set_xdata([a_deg, a_deg])
        dot_fy.set_data([a_deg], [fy_p])
        dot_mz.set_data([a_deg], [mz_p])

        acts = [z, captured["l1"], captured["l2"], captured["out"]]
        weights = [m.weight.detach().cpu().numpy() for m in lin]
        vis.update(weights, acts)
        vis.probe_label.set_text(f"probe  α={a_deg:+6.1f}°   ΔFy={d_fy[N_SWEEP]:+7.1f} N   ΔMz={d_mz[N_SWEEP]:+6.2f} N·m")

        st["gn"] = float(last_gn.item())
        st["dfy_max"] = float(np.max(np.abs(d_fy[:N_SWEEP])))
        lr_cur = optimizer.param_groups[0]["lr"]
        mej_fy = (base_val_fy - st["val_fy"]) / base_val_fy * 100.0
        mej_mz = (base_val_mz - st["val_mz"]) / base_val_mz * 100.0
        eta = st["eta"]
        hud_text.set_text(
            "╔═══════════ TELEMETRÍA PINN HÍBRIDA ═══════════╗\n"
            f"  Época         : {st['epoch']:03d} / {EPOCHS:03d}   ETA {int(eta // 60):02d}:{int(eta % 60):02d}\n"
            f"  Loss          : {st['loss']:.5f}\n"
            f"  LR            : {lr_cur:.2e}   ‖∇‖ {st['gn']:.3f}\n"
            f"  Throughput    : {st['sps']:.0f} steps/s\n"
            "  ─────────────────────────────────────────────\n"
            f"  Pacejka Fy    : {base_val_fy:.2f} N\n"
            f"  PINN Fy RMSE  : {st['val_fy']:.2f} N   ({mej_fy:+.1f}%)\n"
            f"  PINN Mz RMSE  : {st['val_mz']:.2f} N·m ({mej_mz:+.1f}%)\n"
            f"  Mejor época   : {st['best_ep']:03d}  ({st['best_val']:.2f} N)\n"
            f"  |ΔFy| máx     : {st['dfy_max']:.1f} N\n"
            "╚═══════════════════════════════════════════════╝")

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        if writer is not None:
            writer.grab_frame()
        frame += 1
        last_draw = time.perf_counter()

    t_train0 = time.perf_counter()
    try:
        for epoch in range(1, EPOCHS + 1):
            model.residual.train()
            loss_acc = torch.zeros((), device=device)
            gn_acc = torch.zeros((), device=device)
            n_steps = 0

            for b_in, b_tgt in train_loader:
                optimizer.zero_grad(set_to_none=True)
                d_fy_b, d_mz_b = model.residual(*b_in.unbind(-1))
                fy_tot = b_tgt[:, 2] + d_fy_b
                loss, _, _ = criterion(d_fy_b, d_mz_b, b_tgt[:, 0], b_tgt[:, 1], fy_tot, b_in[:, 0], b_in[:, 2])
                loss.backward()
                last_gn = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # Acumulación en device: sin sync por batch
                loss_acc += loss.detach() * len(b_in)
                gn_acc += last_gn.detach()
                n_steps += 1
                n_steps_total += 1

                if time.perf_counter() - last_draw >= interval:
                    st["sps"] = n_steps_total / (time.perf_counter() - t_train0)
                    draw()

            st["loss"] = (loss_acc / len(x_tr)).item()
            gn_mean = (gn_acc / max(n_steps, 1)).item()

            model.residual.eval()
            with torch.no_grad():
                v_dfy, v_dmz = model.residual(*x_va.unbind(-1))
                val_fy = torch.sqrt(torch.mean((t_va[:, 0] - v_dfy) ** 2)).item()
                val_mz = torch.sqrt(torch.mean((t_va[:, 1] - v_dmz) ** 2)).item()

            if val_fy < st["best_val"]:
                st["best_val"], st["best_ep"] = val_fy, epoch
                os.makedirs("export/weights", exist_ok=True)
                model.export_residual_weights("export/weights/pinn_residual_best.pth")

            st.update(epoch=epoch, val_fy=val_fy, val_mz=val_mz)
            st["eta"] = (time.perf_counter() - t_train0) / epoch * (EPOCHS - epoch)

            hist["ep"].append(epoch)
            hist["vfy"].append(val_fy / base_val_fy * 100.0)
            hist["vmz"].append(val_mz / base_val_mz * 100.0)
            hist["loss"].append(st["loss"])
            hist["gn"].append(gn_mean)

            ln_vfy.set_data(hist["ep"], hist["vfy"])
            ln_vmz.set_data(hist["ep"], hist["vmz"])
            allv = hist["vfy"] + hist["vmz"]
            ax_met.set_ylim(max(0.0, min(allv) - 8.0), max(105.0, max(allv) + 5.0))
            ln_loss.set_data(hist["ep"], hist["loss"])
            ln_gn.set_data(hist["ep"], hist["gn"])
            for ax in (ax_loss, ax_gn):
                ax.relim()
                ax.autoscale_view(scalex=False)

            # Estela fantasma de la curva PINN cada 6 épocas (se desvanece con la antigüedad)
            if epoch % 6 == 1:
                with torch.no_grad():
                    d_sw, _ = model.residual(*sweep_in[:N_SWEEP].unbind(-1))
                (gl,) = ax_fy.plot(alpha_deg, fy_base + d_sw.cpu().numpy(), color=RED, lw=1.0, zorder=2)
                ghosts.append(gl)
                if len(ghosts) == ghosts.maxlen:
                    ghosts[0].remove()
                    ghosts.popleft()
                for i, gl_ in enumerate(ghosts):
                    gl_.set_alpha(0.06 + 0.30 * (i + 1) / len(ghosts))

            draw(final_epoch_state=True)

        print("\nEntrenamiento visual finalizado.")
        print(f"Mejor checkpoint (Época {st['best_ep']}): Fy = {st['best_val']:.2f} N | último Mz = {st['val_mz']:.2f} N·m")
        plt.ioff()
        plt.show()
    except _WindowClosed:
        print("Ventana cerrada: entrenamiento interrumpido.")
    finally:
        stack.close()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=180)
    p.add_argument("--fps", type=float, default=24.0, help="Refresco visual máximo (wall-clock)")
    p.add_argument("--probe-hz", type=float, default=0.35, help="Frecuencia del barrido senoidal de α del probe")
    p.add_argument("--record", type=str, default=None, help="Ruta .mp4 (requiere ffmpeg)")
    return p.parse_args()


if __name__ == "__main__":
    run_visual_training(parse_args())