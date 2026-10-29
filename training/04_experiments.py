#!/usr/bin/env python3
"""
Experimentos de causa raíz sobre el residual PINN.
  - Ablación de entradas (¿T_avg / Delta_T / P actúan como reloj y memorizan?)
  - Regularización (weight decay, ruido en entradas, ancho)
  - Protocolos: bloques (como antes) vs entre runs (8->9, 9->8)
"""

import sys, math, copy, time
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.optim as optim
from data.loaders import TTCLoader
from models.grey_box_tire import GreyBoxTireModel
from models.network import ResidualTireMLP
from models.physics_loss import PINNTireLoss

EPOCHS, BATCH, LR, EMA_DECAY = 150, 1024, 3e-3, 0.998
BLOCK, VAL_FRAC = 256, 0.15
DATA_DIR = "data/raw_ttc/RunData_Cornering_ASCII_SI_Round9"
# Columnas: 0=alpha 1=alpha_dot 2=gamma 3=fz 4=P 5=T_avg 6=Delta_T
CONFIGS = [
    dict(name="full",             drop=[],        hidden=64,  wd=1e-4, noise=0.0),
    dict(name="sin_termico",      drop=[5, 6],    hidden=64,  wd=1e-4, noise=0.0),
    dict(name="sin_termico+P",    drop=[4, 5, 6], hidden=64,  wd=1e-4, noise=0.0),
    dict(name="sin_alpha_dot",    drop=[1],       hidden=64,  wd=1e-4, noise=0.0),
    dict(name="full_ruido0.05",   drop=[],        hidden=64,  wd=1e-4, noise=0.05),
    dict(name="full_wd1e-2",      drop=[],        hidden=64,  wd=1e-2, noise=0.0),
    dict(name="sinterm_h128_reg", drop=[5, 6],    hidden=128, wd=1e-3, noise=0.02),
]


def rmse(a, b):
    return torch.sqrt(torch.mean((a - b) ** 2)).item()


def load_run(loader, base, runs, device):
    ds = loader.load_specific_runs(DATA_DIR, run_numbers=runs)
    x = ds.x.to(device)[:, :7].contiguous()
    y = ds.y.to(device)
    with torch.no_grad():
        fy_b, mz_b = base.baseline(x[:, 0], x[:, 2], x[:, 3], x[:, 4])
    t = torch.stack([y[:, 1] - fy_b, y[:, 2] - mz_b, fy_b, mz_b], dim=-1)
    return x, t


def run(cfg, xtr, ttr, xva, tva, device, seed=0):
    torch.manual_seed(seed)
    net = ResidualTireMLP(in_features=7, hidden_dim=cfg["hidden"]).to(device)
    crit = PINNTireLoss(lambda_data=1.0, lambda_passivity=0.5, lambda_reg=1e-4)
    opt = optim.AdamW(net.parameters(), lr=LR, weight_decay=cfg["wd"])
    spe = math.ceil(len(xtr) / BATCH)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * spe, pct_start=0.1,
                                        anneal_strategy="cos", final_div_factor=100.0)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)

    keep = torch.ones(7, dtype=torch.bool, device=device)
    keep[cfg["drop"]] = False
    fill = net.x_mean.clone()
    prep = lambda x: torch.where(keep, x, fill)   # entradas descartadas -> constante

    @torch.no_grad()
    def ev(m, x, t):
        d_fy, d_mz = m(*prep(x).unbind(-1))
        return rmse(d_fy, t[:, 0]), rmse(d_mz, t[:, 1])

    best = (1e9, 0.0, 0)
    step = 0
    for epoch in range(1, EPOCHS + 1):
        net.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(perm), BATCH):
            idx = perm[i:i + BATCH]
            b_in, b_t = xtr[idx], ttr[idx]
            b = prep(b_in)
            if cfg["noise"] > 0:
                b = b + cfg["noise"] * torch.randn_like(b) * net.x_std * keep
            opt.zero_grad(set_to_none=True)
            d_fy, d_mz = net(*b.unbind(-1))
            loss, _, _ = crit(d_fy, d_mz, b_t[:, 0], b_t[:, 1], b_t[:, 2] + d_fy, b_in[:, 0], b_in[:, 2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sch.step()
            step += 1
            decay = min(EMA_DECAY, (1.0 + step) / (10.0 + step))
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), net.parameters()):
                    pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
        v_fy, v_mz = ev(ema, xva, tva)
        if v_fy < best[0]:
            best = (v_fy, v_mz, epoch)
    return dict(best_fy=best[0], best_mz=best[1], best_ep=best[2], final_fy=v_fy, final_mz=v_mz)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = TTCLoader("config/ttc_dataset.yaml")
    base = GreyBoxTireModel("config/pacejka_mf62_params.yaml").to(device).eval()

    x8, t8 = load_run(loader, base, [8], device)
    x9, t9 = load_run(loader, base, [9], device)
    xa, ta = torch.cat([x8, x9]), torch.cat([t8, t9])

    n = len(xa)
    nb = math.ceil(n / BLOCK)
    vb = torch.randperm(nb, generator=torch.Generator().manual_seed(0))[:max(1, int(VAL_FRAC * nb))]
    vm = torch.isin(torch.arange(n) // BLOCK, vb).to(device)

    folds = [
        ("bloques", xa[~vm], ta[~vm], xa[vm], ta[vm]),
        ("8->9", x8, t8, x9, t9),
        ("9->8", x9, t9, x8, t8),
    ]
    print(f"Run 8: {len(x8)} muestras | Run 9: {len(x9)} muestras")
    for name, _, _, _, tva in folds:
        print(f"Pacejka en {name:<8}: Fy {rmse(tva[:, 0], 0 * tva[:, 0]):.2f} N | Mz {rmse(tva[:, 1], 0 * tva[:, 1]):.2f} N*m")

    print(f"\n{'config':<18}{'fold':<9}{'mejor Fy':>9}{'(ep)':>6}{'final Fy':>10}{'mejor Mz':>10}{'final Mz':>10}")
    summary = {}
    for cfg in CONFIGS:
        t0 = time.time()
        for name, xtr, ttr, xva, tva in folds:
            r = run(cfg, xtr, ttr, xva, tva, device)
            summary.setdefault(cfg["name"], {})[name] = r
            print(f"{cfg['name']:<18}{name:<9}{r['best_fy']:9.2f}{r['best_ep']:6d}{r['final_fy']:10.2f}"
                  f"{r['best_mz']:10.2f}{r['final_mz']:10.2f}")
        cross = [summary[cfg['name']][k]['final_fy'] for k in ("8->9", "9->8")]
        print(f"{'':<18}-> media cruzada final Fy: {sum(cross)/2:.2f} N   ({time.time()-t0:.0f} s)\n")


if __name__ == "__main__":
    main()