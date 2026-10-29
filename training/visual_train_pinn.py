#!/usr/bin/env python3
"""
Visualizador Estético Dinámico para PINN Residual (7 -> 64 -> 64 -> 2).
- Cada capa en una única columna recta vertical alineada.
- Sinapsis exclusivamente en escala de grises (fondo tenue/plateado).
- Neuronas con parpadeo reactivo (destello cian/blanco al activarse o cambiar).
"""

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data.loaders import TTCLoader
from models.grey_box_tire import GreyBoxTireModel
from models.physics_loss import PINNTireLoss


class NeuralVisualizerSingleColumn:
    def __init__(self, layer_sizes=[7, 64, 64, 2]):
        self.layer_sizes = layer_sizes
        # Columnas fijas alineadas horizontalmente
        self.x_coords = [0.0, 1.35, 2.70, 4.05]
        self.node_positions = []
        
        # Generar una única columna recta vertical por capa
        for l_idx, (x, size) in enumerate(zip(self.x_coords, layer_sizes)):
            y_max = 1.05 if size <= 7 else 1.42
            y_vals = np.linspace(-y_max, y_max, size)
            self.node_positions.append([(x, yi) for yi in y_vals])

        # Precomputar conexiones sinápticas
        self.segments_per_layer = []
        for l in range(len(layer_sizes) - 1):
            src_nodes = self.node_positions[l]
            dst_nodes = self.node_positions[l + 1]
            segs = []
            for src in src_nodes:
                for dst in dst_nodes:
                    segs.append([src, dst])
            self.segments_per_layer.append(np.array(segs))

        # Memoria de activación previa y energía de destello (fósforo reactivo)
        self.prev_acts = [np.zeros(size) for size in layer_sizes]
        self.flash_energy = [np.zeros(size) for size in layer_sizes]

        # Mapa de color exclusivo para el destello neuronal:
        # Reposo (azul oscuro/apagado) -> Activación (azul eléctrico) -> Fuego (cian) -> Flash (blanco puro)
        self.neuron_cmap = LinearSegmentedColormap.from_list(
            "neuron_flash", ["#0c101c", "#0044cc", "#00ffcc", "#ffffff"], N=256
        )

    def setup_network_axis(self, ax):
        ax.set_title("Topología Neuronal (Pulsos de Activación)", fontsize=11, color="#00ffcc", pad=12, fontweight="bold")
        ax.set_facecolor("#06070a")
        ax.set_xlim(-0.85, 4.85)
        ax.set_ylim(-1.65, 1.65)
        ax.axis("off")

        # Sinapsis: estrictamente en escala de grises
        self.synapse_collections = []
        for segs in self.segments_per_layer:
            lc = LineCollection(segs, cmap=plt.cm.gray, zorder=1)
            lc.set_clim(0.0, 1.0)
            ax.add_collection(lc)
            self.synapse_collections.append(lc)

        # Capa de resplandor difuso (glow) y núcleo de la neurona
        self.glow_scatters = []
        self.core_scatters = []
        for l_idx, nodes in enumerate(self.node_positions):
            xy = np.array(nodes)
            base_s = 130 if self.layer_sizes[l_idx] <= 7 else 22
            
            # Halo difuso
            glow = ax.scatter(xy[:, 0], xy[:, 1], s=base_s * 2.8, c=np.zeros(len(xy)),
                              cmap=self.neuron_cmap, vmin=0.0, vmax=1.0, alpha=0.35, zorder=2)
            # Núcleo definido
            core = ax.scatter(xy[:, 0], xy[:, 1], s=base_s, c=np.zeros(len(xy)),
                              cmap=self.neuron_cmap, vmin=0.0, vmax=1.0,
                              edgecolors="#222f44", linewidth=0.6, zorder=3)
            self.glow_scatters.append(glow)
            self.core_scatters.append(core)

        # Etiquetas de canales
        labels_in = [r"$\alpha$", r"$\dot{\alpha}$", r"$\gamma$", r"$F_z$", r"$P$", r"$T_{\mathrm{avg}}$", r"$\Delta T$"]
        labels_out = [r"$\Delta F_y$", r"$\Delta M_z$"]
        
        for idx, (x, y) in enumerate(self.node_positions[0]):
            ax.text(x - 0.12, y, labels_in[idx], color="#66fcf1", fontsize=9.5, ha="right", va="center", fontweight="bold")
        for idx, (x, y) in enumerate(self.node_positions[-1]):
            ax.text(x + 0.14, y, labels_out[idx], color="#ff0055", fontsize=10, ha="left", va="center", fontweight="bold")

        layer_names = ["Entrada (7)", "Oculta 1 (64)", "Oculta 2 (64)", "Salida (2)"]
        for x, name in zip(self.x_coords, layer_names):
            ax.text(x, -1.55, name, color="#7f8c8d", fontsize=8.5, ha="center")

    def update_network(self, weights_list, current_acts):
        # 1. Sinapsis: escala de grises con filtrado de pesos débiles para evitar maraña
        for l_idx, w in enumerate(weights_list):
            w_flat = np.abs(w.T.flatten())
            max_w = np.max(w_flat) + 1e-6
            
            # Normalizar magnitud entre 0.0 y 1.0 (escala de grises)
            w_norm = np.clip(w_flat / max_w, 0.0, 1.0)
            
            # Poda visual: solo resaltar el 20% de conexiones más fuertes
            threshold = np.percentile(w_flat, 80.0)
            active = w_flat >= threshold
            
            # Brillo grisáceo proporcional a la importancia de la sinapsis
            gray_val = np.where(active, 0.2 + 0.7 * (w_norm ** 1.5), 0.04)
            widths = np.where(active, 0.25 + 1.1 * (w_norm ** 2.0), 0.0)

            self.synapse_collections[l_idx].set_array(gray_val)
            self.synapse_collections[l_idx].set_linewidth(widths)

        # 2. Neuronas: parpadeo reactivo al cambio de activación instantáneo
        for l_idx, acts in enumerate(current_acts):
            act_abs = np.abs(acts)
            act_norm = act_abs / (np.max(act_abs) + 1e-6)

            # Tasa de cambio respecto al paso anterior: mayor cambio = destello blanco
            delta = np.abs(act_norm - self.prev_acts[l_idx])
            self.prev_acts[l_idx] = act_norm.copy()

            # La energía sube con la actividad y el cambio, decayendo suavemente
            self.flash_energy[l_idx] = self.flash_energy[l_idx] * 0.65 + (act_norm * 0.4 + delta * 1.8)
            intensity = np.clip(self.flash_energy[l_idx], 0.0, 1.0)

            base_s = 130 if self.layer_sizes[l_idx] <= 7 else 22
            dynamic_s = base_s * (0.8 + 1.4 * intensity)

            self.core_scatters[l_idx].set_array(intensity)
            self.core_scatters[l_idx].set_sizes(dynamic_s)
            self.glow_scatters[l_idx].set_array(intensity)
            self.glow_scatters[l_idx].set_sizes(dynamic_s * 2.8)


def run_visual_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Visualizador PINN Grey-Box (Líneas Grises + Destello Neuronal) — {device} ===")

    loader = TTCLoader("config/ttc_dataset.yaml")
    data_dir = "data/raw_ttc/RunData_Cornering_ASCII_SI_Round9"
    raw_dataset = loader.load_specific_runs(data_dir, run_numbers=[8, 9])

    x_all = raw_dataset.x.to(device)
    y_all = raw_dataset.y.to(device)

    model = GreyBoxTireModel("config/pacejka_mf62_params.yaml").to(device)
    model.eval()

    # Precalcular residuos sobre Pacejka MF 6.2 base
    with torch.no_grad():
        fy_base, mz_base = model.baseline(x_all[:, 0], x_all[:, 2], x_all[:, 3], x_all[:, 4])
        res_fy = y_all[:, 1] - fy_base
        res_mz = y_all[:, 2] - mz_base

    inputs = x_all[:, :7].contiguous()
    targets = torch.stack([res_fy, res_mz, fy_base, mz_base], dim=-1)
    n = len(inputs)

    # Validación por bloques contiguos (idéntica a 02_train_pinn.py)
    BLOCK = 256
    VAL_FRAC = 0.15
    n_blocks = math.ceil(n / BLOCK)
    g = torch.Generator().manual_seed(0)
    perm_b = torch.randperm(n_blocks, generator=g)
    val_blocks = perm_b[:max(1, int(VAL_FRAC * n_blocks))]
    block_id = torch.arange(n) // BLOCK
    val_mask = torch.isin(block_id, val_blocks).to(device)

    x_tr, t_tr = inputs[~val_mask], targets[~val_mask]
    x_va, t_va = inputs[val_mask], targets[val_mask]

    base_val_fy = torch.sqrt(torch.mean(t_va[:, 0] ** 2)).item()
    base_val_mz = torch.sqrt(torch.mean(t_va[:, 1] ** 2)).item()

    dataset_tr = TensorDataset(x_tr, t_tr)
    train_loader = DataLoader(dataset_tr, batch_size=1024, shuffle=True)

    # Hooks en las salidas activadas (post-SiLU) para capturar el disparo biológico de cada neurona
    batch_acts = {}
    def hook_fn(name):
        def hook(mod, inp, out):
            # Media de activación a lo largo del mini-batch actual
            batch_acts[name] = out.detach().abs().mean(dim=0).cpu().numpy()
        return hook

    model.residual.net[1].register_forward_hook(hook_fn("layer1"))  # Post-SiLU capa 1 (64)
    model.residual.net[3].register_forward_hook(hook_fn("layer2"))  # Post-SiLU capa 2 (64)
    model.residual.net[4].register_forward_hook(hook_fn("output"))  # Salida lineal (2)

    # Barrido nominal central para comparar curva
    alpha_deg = np.linspace(-12.0, 12.0, 160)
    alpha_rad = np.deg2rad(alpha_deg)
    t_a = torch.tensor(alpha_rad, dtype=torch.float32, device=device)
    t_adot = torch.zeros_like(t_a)
    t_zero = torch.zeros_like(t_a)
    t_fz = torch.full_like(t_a, 700.0)
    t_p = torch.full_like(t_a, 82.7)
    t_tavg = torch.full_like(t_a, 60.0)
    t_dt = torch.zeros_like(t_a)

    with torch.no_grad():
        fy_base_sweep, _ = model.baseline(t_a, t_zero, t_fz, t_p)
        fy_base_np = fy_base_sweep.cpu().numpy()

    mask_slice = (
        (np.abs(raw_dataset.x[:, 2].numpy()) < 0.015) & 
        (raw_dataset.x[:, 3].numpy() >= 550.0) & 
        (raw_dataset.x[:, 3].numpy() <= 850.0)
    )
    alpha_ttc = np.rad2deg(raw_dataset.x[mask_slice, 0].numpy())
    fy_ttc = raw_dataset.y[mask_slice, 1].numpy()

    # Lienzo de 3 paneles
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(19, 7.5), facecolor="#06070a")
    gs = gridspec.GridSpec(2, 3, width_ratios=[1.25, 1.15, 0.9], height_ratios=[1.0, 1.0])

    ax_net = fig.add_subplot(gs[:, 0])
    ax_curve = fig.add_subplot(gs[:, 1])
    ax_metric = fig.add_subplot(gs[0, 2])
    ax_hud = fig.add_subplot(gs[1, 2])

    vis = NeuralVisualizerSingleColumn(layer_sizes=[7, 64, 64, 2])
    vis.setup_network_axis(ax_net)

    # Panel Central: Curva
    ax_curve.set_title(r"Curva Lateral $F_y$ vs $\alpha$ ($F_z = 700\ \mathrm{N},\ \gamma = 0^\circ$)", 
                       fontsize=11, color="#00ffcc", pad=12, fontweight="bold")
    ax_curve.set_facecolor("#0b0d14")
    ax_curve.grid(True, linestyle="--", alpha=0.2, color="#4a5568")
    ax_curve.set_xlabel(r"Slip Angle $\alpha$ [deg]", fontsize=10, color="white")
    ax_curve.set_ylabel(r"Fuerza Lateral $F_y$ [N]", fontsize=10, color="white")
    ax_curve.scatter(alpha_ttc, fy_ttc, color="#45a29e", alpha=0.25, s=7, label="Calspan TTC")
    ax_curve.plot(alpha_deg, fy_base_np, color="#66fcf1", linestyle="--", linewidth=1.6, label="Pacejka MF 6.2 Base")
    line_pinn, = ax_curve.plot(alpha_deg, fy_base_np, color="#ff0055", linewidth=2.4, label="PINN Híbrido")
    ax_curve.legend(loc="upper left", framealpha=0.3, fontsize=8.5)

    # Panel Métricas
    ax_metric.set_title("Evolución del Error de Validación", fontsize=11, color="#00ffcc", pad=10, fontweight="bold")
    ax_metric.set_facecolor("#0b0d14")
    ax_metric.grid(True, linestyle="--", alpha=0.2, color="#4a5568")
    ax_metric.set_ylabel("RMSE [N]", fontsize=9, color="white")
    ax_metric.axhline(base_val_fy, color="#66fcf1", linestyle="--", linewidth=1.2, label=f"Pacejka: {base_val_fy:.1f} N")
    line_rmse, = ax_metric.plot([], [], color="#ff0055", linewidth=2.0, label="PINN Val RMSE")
    ax_metric.legend(loc="upper right", framealpha=0.3, fontsize=8)

    # Panel HUD
    ax_hud.set_facecolor("#0b0d14")
    ax_hud.axis("off")
    hud_text = ax_hud.text(0.04, 0.5, "", color="white", fontsize=10, family="monospace", va="center")

    plt.tight_layout()
    plt.ion()
    plt.show(block=False)

    # Entrenamiento idéntico a 02_train_pinn.py
    criterion = PINNTireLoss(lambda_data=1.0, lambda_passivity=0.5, lambda_reg=1e-4)
    optimizer = optim.AdamW(model.residual.parameters(), lr=3e-3, weight_decay=1e-4)
    EPOCHS = 180
    steps_per_epoch = math.ceil(len(x_tr) / 1024)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-3, total_steps=EPOCHS * steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos", final_div_factor=100.0
    )

    history_epochs = []
    history_val_fy = []
    best_val = base_val_fy
    best_ep = 0

    for epoch in range(1, EPOCHS + 1):
        model.residual.train()
        total_loss = 0.0

        for b_idx, (b_in, b_tgt) in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            d_fy, d_mz = model.residual(*b_in.unbind(-1))
            fy_tot = b_tgt[:, 2] + d_fy
            loss, _, _ = criterion(d_fy, d_mz, b_tgt[:, 0], b_tgt[:, 1], fy_tot, b_in[:, 0], b_in[:, 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * len(b_in)

            # Refresco del parpadeo reactivo cada 5 batches
            if b_idx % 5 == 0:
                with torch.no_grad():
                    w1 = model.residual.net[0].weight.detach().cpu().numpy()
                    w2 = model.residual.net[2].weight.detach().cpu().numpy()
                    w3 = model.residual.net[4].weight.detach().cpu().numpy()

                    # Activación de entrada normalizada del batch actual
                    act0 = b_in.detach().abs().mean(dim=0).cpu().numpy()
                    act1 = batch_acts.get("layer1", np.zeros(64))
                    act2 = batch_acts.get("layer2", np.zeros(64))
                    act3 = batch_acts.get("output", np.zeros(2))

                    vis.update_network([w1, w2, w3], [act0, act1, act2, act3])
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    time.sleep(0.005)

        epoch_loss = total_loss / len(x_tr)

        # Validación
        model.residual.eval()
        with torch.no_grad():
            v_dfy, v_dmz = model.residual(*x_va.unbind(-1))
            val_rmse_fy = torch.sqrt(torch.mean((t_va[:, 0] - v_dfy) ** 2)).item()
            val_rmse_mz = torch.sqrt(torch.mean((t_va[:, 1] - v_dmz) ** 2)).item()

            d_fy_sw, _ = model.residual(t_a, t_adot, t_zero, t_fz, t_p, t_tavg, t_dt)
            fy_pinn_np = (fy_base_sweep + d_fy_sw).cpu().numpy()

        if val_rmse_fy < best_val:
            best_val = val_rmse_fy
            best_ep = epoch
            os.makedirs("export/weights", exist_ok=True)
            model.export_residual_weights("export/weights/pinn_residual_best.pth")

        history_epochs.append(epoch)
        history_val_fy.append(val_rmse_fy)

        line_pinn.set_ydata(fy_pinn_np)
        line_rmse.set_data(history_epochs, history_val_fy)
        ax_metric.set_xlim(1, max(epoch, 10))
        ax_metric.set_ylim(min(history_val_fy) - 3, base_val_fy + 3)

        mejora_fy = (base_val_fy - val_rmse_fy) / base_val_fy * 100.0
        mejora_mz = (base_val_mz - val_rmse_mz) / base_val_mz * 100.0
        lr_cur = optimizer.param_groups[0]["lr"]

        hud_str = (
            f"╔════════════ TELEMETRÍA PINN HÍBRIDA ════════════╗\n"
            f"  Época         : {epoch:03d} / {EPOCHS:03d}\n"
            f"  Pérdida (Loss): {epoch_loss:.5f}\n"
            f"  Learning Rate : {lr_cur:.2e}\n"
            f"  ───────────────────────────────────────────────\n"
            f"  Pacejka Base  : {base_val_fy:.2f} N\n"
            f"  PINN Val RMSE : {val_rmse_fy:.2f} N  ({mejora_fy:+.1f}%)\n"
            f"  PINN Mz RMSE  : {val_rmse_mz:.2f} N·m ({mejora_mz:+.1f}%)\n"
            f"  Mejor Época   : {best_ep:03d}  (RMSE: {best_val:.2f} N)\n"
            f"  Delta Fy Max  : {np.max(np.abs(d_fy_sw.cpu().numpy())):.1f} N\n"
            f"╚═════════════════════════════════════════════════╝"
        )
        hud_text.set_text(hud_str)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    print(f"\nEntrenamiento visual finalizado.")
    print(f"Mejor checkpoint (Época {best_ep}): Fy = {best_val:.2f} N | Mz = {val_rmse_mz:.2f} N*m")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_visual_training()