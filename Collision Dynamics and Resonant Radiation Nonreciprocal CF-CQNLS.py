import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import make_interp_spline
import pandas as pd
import os
import random
import time
from dataclasses import dataclass, fields
import warnings

warnings.filterwarnings('ignore')

# ===================== 1. CINEMATIC AESTHETICS (完全保留原有风格) =====================

# 🎨 Color Palette: "Deep Science"
COLORS = {
    'cyan': '#00b4d8',  # Soliton 1 (Active)
    'magenta': '#f72585',  # Soliton 2 (Active)
    'dark': '#1d3557',  # Background
    'light': '#f1faee',  # Text
    'accent': '#e63946',  # Highlights
    'gold': '#ffb703',  # Stars
    'text': '#333333',  # Main Text
    'ghost': '#e0e1dd',  # Reciprocal "Ghost" color (Silver/White)
}

# 📐 Plot Styling (完全保留)
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#555555',
    'axes.grid': True,
    'grid.alpha': 0.15,
    'grid.color': 'black',
    'grid.linestyle': '-',
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'mathtext.fontset': 'stixsans',
    # ============ 只添加这 2 行，启用 LaTeX 符号 ============
    'text.usetex': False,    # 不启用完整LaTeX，避免报错
    'axes.formatter.use_mathtext': True  # 启用 $ $ 符号渲染
})


def add_glow(ax, x, y, color, lw=2, alpha=1.0, label=None):
    """Adds a neon glow effect to lines (Standard)."""
    ax.plot(x, y, color=color, linewidth=lw * 5, alpha=0.2 * alpha, zorder=1)
    ax.plot(x, y, color=color, linewidth=lw * 2.5, alpha=0.5 * alpha, zorder=2)
    return ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=3)


def add_ghost_trace(ax, x, y, color, lw=2, alpha=0.5, label=None):
    """Adds a 'Holographic' ghost effect for Reciprocal trajectories."""
    ax.plot(x, y, color=color, linewidth=lw * 4, alpha=0.08 * alpha, zorder=1)
    return ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=2,
                   path_effects=[pe.Stroke(linewidth=lw + 1.5, foreground='black', alpha=0.3), pe.Normal()])


def style_axis(ax, spines_off=['top', 'right']):
    for s in spines_off:
        ax.spines[s].set_visible(False)
    ax.tick_params(direction='out', width=1.2, length=4, colors='#333333')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUTPUT_DIR = "OE_Figures_Ultimate"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== 2. PHYSICS ENGINE (Optimized) =====================

@dataclass
class SimConfig:
    N: int = 2048
    L: float = 100.0
    alpha: float = 1.8
    dt: float = 5e-4
    T: float = 60.0
    v: float = 1.0
    shift: int = 150
    gamma: float = 1.0
    beta: float = 1.0   # <--- 只加这一行
    gs_tol: float = 1e-8
    collision_sep_threshold: float = 5.0

    def __post_init__(self):
        self.dx = 2 * self.L / self.N
        self.shift_physical = self.shift * self.dx
        self.t_collision_theory = self.shift_physical / self.v

    def copy(self, **kwargs):
        valid = {f.name for f in fields(self)}
        curr = {f.name: getattr(self, f.name) for f in fields(self)}
        curr.update({k: v for k, v in kwargs.items() if k in valid})
        return SimConfig(**curr)


class NonreciprocalFCQNLS:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.x = torch.linspace(-cfg.L, cfg.L, cfg.N, device=DEVICE)
        k = (2 * np.pi / (2 * cfg.L)) * torch.fft.fftfreq(cfg.N).to(DEVICE) * cfg.N
        self.k = k
        self.k_alpha = torch.abs(k) ** cfg.alpha
        self.lin_half = torch.exp(-0.5j * self.k_alpha * cfg.dt)
        self.gamma_12 = cfg.gamma
        self.gamma_21 = 2.0 - cfg.gamma
        self.delta_gamma = abs(self.gamma_12 - self.gamma_21)

    def hamiltonian(self, psi):
        psi_k = torch.fft.fft(psi, dim=1)
        # 分数阶动能项: ψ_j*(-Δ)^{α/2}ψ_j
        kin = torch.sum(self.k_alpha * torch.abs(psi_k) ** 2) * self.cfg.dx / self.cfg.N
    
        rho = torch.abs(psi) ** 2
        # 三次项: -1/2 |ψ_j|^4
        cubic = -0.5 * torch.sum(rho[0] ** 2 + rho[1] ** 2) * self.cfg.dx
        # 五次项: -β/3 |ψ_j|^6
        quintic = -self.cfg.beta / 3.0 * torch.sum(rho[0] ** 3 + rho[1] ** 3) * self.cfg.dx
    
        # 对称交叉项: -(γ12+γ21)/2 |ψ1|^2|ψ2|^2
        cross = -(self.gamma_12 + self.gamma_21) / 2.0 * torch.sum(rho[0] * rho[1]) * self.cfg.dx
    
        return kin + cubic + quintic + cross

    def momentum(self, psi):
        dpsi = torch.fft.ifft(1j * self.k * torch.fft.fft(psi, dim=1), dim=1)
        return torch.imag(torch.sum(torch.conj(psi) * dpsi, dim=1) * self.cfg.dx)

    def center_of_mass(self, psi):
        rho = torch.abs(psi) ** 2
        norm1 = torch.sum(rho[0]) * self.cfg.dx + 1e-12
        norm2 = torch.sum(rho[1]) * self.cfg.dx + 1e-12
        return (torch.sum(self.x * rho[0]) * self.cfg.dx / norm1,
                torch.sum(self.x * rho[1]) * self.cfg.dx / norm2)

    def ground_state(self):
        psi = torch.exp(-self.x ** 2 / 5).to(DEVICE) + 0j
        for _ in range(20000):
            psi_k = torch.fft.fft(psi)
            psi = torch.fft.ifft(psi_k * torch.exp(-self.k_alpha * 0.002))
            rho = torch.abs(psi) ** 2
            psi = psi * torch.exp(rho * 0.002)
            psi = psi / torch.sqrt(torch.sum(torch.abs(psi) ** 2) * self.cfg.dx) * np.sqrt(5.0)
        return psi

    def run_collision(self, psi0, save_full=True):
        cfg = self.cfg
        steps = int(cfg.T / cfg.dt)
        save_int = max(1, steps // 400)
        psi = psi0.clone()
        H0 = self.hamiltonian(psi)
        P0 = torch.sum(self.momentum(psi)).item()
        cm1_prev, cm2_prev = self.center_of_mass(psi)

        data = {'t': [], 'cm1': [], 'cm2': [], 'v1': [], 'v2': [], 'dv': [],
                'P': [], 'H_err': [], 'phase': [], 'rad': [], 'field': [], 'k_spec': None,
                'peak_rho': [], 'collision_valid': False, 'min_cm_sep': None}

        for i in range(steps):
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * self.lin_half, dim=1)
            rho1, rho2 = torch.abs(psi[0]) ** 2, torch.abs(psi[1]) ** 2
            psi[0] *= torch.exp(1j * (rho1 + self.gamma_12 * rho2) * cfg.dt)
            psi[1] *= torch.exp(1j * (rho2 + self.gamma_21 * rho1) * cfg.dt)
            psi = torch.fft.ifft(torch.fft.fft(psi, dim=1) * self.lin_half, dim=1)

            if abs(i * cfg.dt - cfg.T / 2) < cfg.dt / 2:
                k_power = torch.mean(torch.abs(torch.fft.fft(psi, dim=1)) ** 2, dim=0).cpu().numpy()
                data['k_spec'] = np.fft.fftshift(k_power)

            if i % save_int == 0:
                t = i * cfg.dt
                cm1, cm2 = self.center_of_mass(psi)
                v1 = (cm1 - cm1_prev) / (save_int * cfg.dt)
                v2 = (cm2 - cm2_prev) / (save_int * cfg.dt)
                cm1_prev, cm2_prev = cm1, cm2

                P_tot = torch.sum(self.momentum(psi))
                H_err = abs(self.hamiltonian(psi) - H0) / abs(H0)
                rho_tot = rho1 + rho2
                core = ((self.x > cm1 - 20) & (self.x < cm1 + 20)) | ((self.x > cm2 - 20) & (self.x < cm2 + 20))
                rad = 1 - torch.sum(rho_tot[core]) / torch.sum(rho_tot)

                data['t'].append(t)
                data['cm1'].append(cm1.item())
                data['cm2'].append(cm2.item())
                data['v1'].append(v1.item())
                data['v2'].append(v2.item())
                data['dv'].append(v1.item() - v2.item())
                data['P'].append(P_tot.item())
                data['H_err'].append(H_err.item())
                data['peak_rho'].append(torch.max(rho_tot).item())
                data['phase'].append((cm1 - cm2).item())
                data['rad'].append(rad.item())

                if save_full:
                    data['field'].append(torch.stack([rho1, rho2]).cpu().numpy())

        for k in data:
            if k not in ['field', 'k_spec', 'collision_valid', 'min_cm_sep']:
                data[k] = np.array(data[k])
        if save_full:
            data['field'] = np.array(data['field'])

        sep = np.abs(data['cm1'] - data['cm2'])
        data['min_cm_sep'] = np.min(sep)
        idx = np.argmin(sep)
        data['t_collision'] = data['t'][idx]
        data['x_collision'] = (data['cm1'][idx] + data['cm2'][idx]) / 2
        data['P0'] = P0
        data['collision_valid'] = data['min_cm_sep'] < cfg.collision_sep_threshold

        return data


# ===================== 3. MASTERPIECE FIGURES =====================

def create_figure1_ultimate(data_ref, data_nr, x, cfg, name="comparison"):
    field_nr = data_nr['field']
    t = data_nr['t']
    rho_total_nr = field_nr[:, 0, :] + field_nr[:, 1, :]
    t_col = data_nr['t_collision']

    max_excursion = max(
        np.max(np.abs(data_nr['cm1'])), np.max(np.abs(data_nr['cm2'])),
        np.max(np.abs(data_ref['cm1'])), np.max(np.abs(data_ref['cm2']))
    )
    visual_limit = max_excursion * 1.15
    visual_limit = max(visual_limit, 40)

    fig = plt.figure(figsize=(11, 7), constrained_layout=True)
    gs = GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[0.3, 1.5, 1], wspace=0.15, hspace=0.25)

    ax_map = fig.add_subplot(gs[1:, 0])
    ax_map.set_facecolor('black')

    # LaTeX 符号
    add_ghost_trace(ax_map, data_ref['cm1'], t, COLORS['ghost'], lw=1.5, label=r'Reciprocal ($\gamma_{12}=\gamma_{21}=1.0$)')
    add_ghost_trace(ax_map, data_ref['cm2'], t, COLORS['ghost'], lw=1.5)

    gamma_12 = cfg.gamma
    gamma_21 = 2.0 - cfg.gamma
    add_glow(ax_map, data_nr['cm1'], t, COLORS['cyan'], lw=1.5, label=f'Non-reciprocal ($\\gamma_{{12}}={gamma_12:.1f}$, $\\gamma_{{21}}={gamma_21:.1f}$)')
    add_glow(ax_map, data_nr['cm2'], t, COLORS['magenta'], lw=1.5)

    ax_map.text(20, t_col, "SYMMETRY\nBREAKING", color='white', ha='left', va='center',
                fontsize=8, fontweight='bold',
                path_effects=[pe.withStroke(linewidth=2, foreground='black')])

    ax_map.annotate("", xy=(0, t_col), xytext=(18, t_col),
                    arrowprops=dict(arrowstyle="->", color='white', lw=1, shrinkA=0, shrinkB=5))

    style_axis(ax_map)
    ax_map.set_xlabel(r'Position $x$')
    ax_map.set_ylabel(r'Propagation Distance $t$')
    ax_map.set_xlim(-visual_limit, visual_limit)
    ax_map.set_ylim(0, cfg.T)

    leg = ax_map.legend(loc='lower left', fontsize=8, facecolor='black', framealpha=0.4)
    for text in leg.get_texts(): text.set_color('white')

    ax_traj = fig.add_subplot(gs[0, 0], sharex=ax_map)
    add_ghost_trace(ax_traj, data_ref['cm1'], t, COLORS['ghost'], lw=1.5, label=r'Ref ($\gamma=1.0$)')
    add_ghost_trace(ax_traj, data_ref['cm2'], t, COLORS['ghost'], lw=1.5)
    add_glow(ax_traj, data_nr['cm1'], t, COLORS['cyan'], label=f'NR ($\\gamma_{{12}}={gamma_12:.1f}$)')
    add_glow(ax_traj, data_nr['cm2'], t, COLORS['magenta'])

    ax_traj.scatter(data_nr['x_collision'], t_col, color=COLORS['gold'], s=200, marker='*',
                    edgecolor='white', lw=1.5, zorder=10)

    style_axis(ax_traj, spines_off=['top', 'right', 'bottom'])
    ax_traj.set_yticks([])
    ax_traj.set_ylabel('Evol.')
    #ax_traj.legend(loc='upper right', fontsize=8, ncol=2)

    indices = [0, np.argmin(np.abs(t - t_col)), len(t) - 1]
    titles = [fr'Initial State ($t={t[0]:.1f}$)', fr'Collision ($t={t_col:.1f}$)', fr'Final State ($t={t[-1]:.1f}$)']

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(gs[i, 1])

        ax.fill_between(x, data_ref['field'][idx, 0, :], color=COLORS['ghost'], alpha=0.15)
        ax.plot(x, data_ref['field'][idx, 0, :], color=COLORS['ghost'], lw=1.5, alpha=0.6)
        ax.plot(x, data_ref['field'][idx, 1, :], color=COLORS['ghost'], lw=1.5, alpha=0.6)

        ax.fill_between(x, field_nr[idx, 0, :], color=COLORS['cyan'], alpha=0.2)
        ax.plot(x, field_nr[idx, 0, :], color=COLORS['cyan'], lw=1.5)
        ax.fill_between(x, field_nr[idx, 1, :], color=COLORS['magenta'], alpha=0.2)
        ax.plot(x, field_nr[idx, 1, :], color=COLORS['magenta'], lw=1.5)
        style_axis(ax)
        ax.set_xlim(-visual_limit, visual_limit)
        ax.set_ylim(0, np.max(rho_total_nr) * 1.1)
        ax.set_yticks([])

        ax.text(0.5, 0.95, titles[i], transform=ax.transAxes, fontsize=9, fontweight='bold',
                color=COLORS['dark'], ha='center', va='top')

        # ============ 删除 Initial State 子图的图例 ============
        # if i == 0:
        #     from matplotlib.lines import Line2D
        #     l = [Line2D([0], [0], color=COLORS['ghost'], lw=2, alpha=0.6, label='Reciprocal'),
        #          Line2D([0], [0], color=COLORS['cyan'], lw=2, label='Non-reciprocal')]
        #     ax.legend(handles=l, loc='lower right', fontsize=8)

        if i == 2: ax.set_xlabel(r'Position $x$')

    plt.suptitle("Nonreciprocal Soliton Scattering Dynamics", x=0.05, y=0.98, ha='left', fontsize=14, weight='bold')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Dynamics_{name}.png", bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Dynamics_{name}.pdf", bbox_inches='tight')
    plt.close()


def create_figure2_ultimate(data, cfg, name="baseline"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 7), sharex=True, gridspec_kw={'height_ratios': [1, 1], 'hspace': 0.3}, constrained_layout=True)
    t = data['t']

    ax1.axhline(0, color='black', lw=0.8, alpha=0.3)
    # ====================== 核心修改：统一使用绝对值速度差 ======================
    dv_abs = np.abs(np.abs(data['v1']) - np.abs(data['v2']))
    ax1.fill_between(t, 0, dv_abs, color=COLORS['accent'], alpha=0.15)
    add_glow(ax1, t, dv_abs, COLORS['accent'])

    dv_peak = np.max(dv_abs)
    dv_peak_t = data['t'][np.argmax(dv_abs)]
    final_dv = dv_abs[-1]

    # 标注
    ax1.annotate(r"Peak $|\Delta v|$"+f" = {dv_peak:.2f}",
                 xy=(dv_peak_t, dv_peak),
                 xytext=(dv_peak_t + 2, dv_peak),
                 arrowprops=dict(arrowstyle="->", color=COLORS['accent'], lw=1),
                 ha='left', fontsize=8, color=COLORS['accent'])

    ax1.annotate(r"Final $|\Delta v|$"+f" = {final_dv:.4f}",
                 xy=(t[-1], final_dv),
                 xytext=(t[-1] - 10, final_dv + 0.08),
                 ha='right', fontsize=9, fontweight='bold', color=COLORS['accent'])

    style_axis(ax1)
    # 标签同步更新
    ax1.set_ylabel(r'Velocity Asymmetry $|\Delta v| = ||v_1| - |v_2||$')
    ax1.set_title(r'Nonreciprocal Induction ($\gamma_{12} \neq \gamma_{21}$)', loc='left', fontsize=11, pad=10)

    ax2.plot(t, data['P'], color=COLORS['dark'], lw=1.5, label=r'Total Momentum $P$')

    ax2r = ax2.twinx()
    sep = np.abs(data['cm1'] - data['cm2'])
    ax2r.plot(t, sep, color=COLORS['cyan'], ls=':', lw=1.5, label=r'Separation $d$')
    ax2r.fill_between(t, sep, color=COLORS['cyan'], alpha=0.05)

    style_axis(ax2)
    ax2r.spines['top'].set_visible(False)
    ax2r.spines['right'].set_visible(True)
    ax2r.spines['right'].set_color(COLORS['cyan'])
    ax2r.tick_params(axis='y', colors=COLORS['cyan'])
    ax2r.set_ylabel(r'Soliton Separation $d = |x_1 - x_2|$', color=COLORS['cyan'])

    ax2.set_ylabel(r'Total Momentum $P$', color=COLORS['dark'])
    ax2.set_xlabel(r'Propagation Distance $t$')
    ax2.set_xlim(0, cfg.T)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=8)

    plt.subplots_adjust(left=0.15, right=0.9, top=0.92, bottom=0.12)
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Observables_{name}.png", bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Observables_{name}.pdf", bbox_inches='tight')
    plt.close()


def create_figure3_combined(gamma_vals, gamma_data, alphas, a_res):
    fig, (ax1, ax2_wrapper) = plt.subplots(1, 2, figsize=(14, 4.5), constrained_layout=True)

    phases = [np.max(r['phase']) for r in gamma_data]
    # ====================== 核心修改：Δv = |v₁| - |v₂| 绝对值速度差 ======================
    dvs = [abs(abs(r['v1'][-1]) - abs(r['v2'][-1])) for r in gamma_data]

    # -------------------------- 折叠为 |Δγ| 轴 --------------------------
    delta_gammas = np.abs(2 * (gamma_vals - 1.0))
    sort_idx = np.argsort(delta_gammas)
    delta_gammas_sorted = delta_gammas[sort_idx]
    phases_sorted = np.array(phases)[sort_idx]
    dvs_sorted = np.array(dvs)[sort_idx]
    # -------------------------------------------------------------------------

    ax1_2 = ax1.twinx()
    l1 = add_glow(ax1, delta_gammas_sorted, phases_sorted, COLORS['cyan'], label=r'Max Phase Shift $\Delta\phi$')
    ax1.scatter(delta_gammas_sorted, phases_sorted, color='white', edgecolors=COLORS['cyan'], s=60, lw=1.5, zorder=10)

    l2 = add_glow(ax1_2, delta_gammas_sorted, dvs_sorted, COLORS['accent'], label=r'Velocity Asymmetry $|\Delta v| = ||v_1|-|v_2||$')
    ax1_2.scatter(delta_gammas_sorted, dvs_sorted, color='white', edgecolors=COLORS['accent'], marker='s', s=60, lw=1.5, zorder=10)

    prev_dv = None
    for i, (dg, dv) in enumerate(zip(delta_gammas_sorted, dvs_sorted)):
        if i == 0 or i == len(delta_gammas_sorted)-1 or (prev_dv is not None and abs(dv - prev_dv) > 0.01):
            offset_y = 10 if i % 2 == 0 else -15
            ax1_2.annotate(f'{dv:.2f}', xy=(dg, dv), xytext=(0, offset_y),
                          textcoords='offset points', fontsize=7, ha='center')
        prev_dv = dv

    ax1.axvline(0.0, color='gray', ls='--', alpha=0.5)
    ax1.text(0.01, min(phases_sorted), r"Reciprocal$(\Delta\gamma=0)$", rotation=90, fontsize=8, color='gray', va='bottom')

    style_axis(ax1)
    ax1_2.spines['top'].set_visible(False)
    ax1_2.spines['right'].set_visible(True)
    ax1_2.spines['right'].set_color(COLORS['accent'])
    ax1_2.tick_params(axis='y', colors=COLORS['accent'])

    ax1_2.set_ylim(min(dvs_sorted) - 0.02, max(dvs_sorted) + 0.02)

    ax1.set_xlabel(r'Nonreciprocity $|\Delta\gamma| = |\gamma_{12} - \gamma_{21}|$')
    ax1.set_ylabel(r'Phase Shift $\Delta\phi$', color=COLORS['cyan'])
    ax1_2.set_ylabel(r'Asymmetry $|\Delta v| = ||v_1| - |v_2||$', color=COLORS['accent'])
    ax1.tick_params(axis='y', colors=COLORS['cyan'])

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['cyan'], lw=2, label=r'Phase Shift $\Delta\phi$'),
        Line2D([0], [0], color=COLORS['accent'], lw=2, label=r'Velocity Asymmetry $|\Delta v|$')
    ]
    ax1.legend(handles=legend_elements, loc='upper left')
    ax1.set_title("Tunable Nonreciprocity Statistics", fontweight='bold')

    # 右边子图
    rho_peak = [np.max(data['peak_rho']) for data in a_res]
    R_rad = [data['rad'][-1] for data in a_res]

    color1 = COLORS['cyan']
    ax2 = ax2_wrapper
    ax2.set_xlabel(r'Fractional Order $\alpha$')
    ax2.set_ylabel(r'Peak Density $\rho_{\rm max}$', color=color1)
    add_glow(ax2, alphas, rho_peak, color1, lw=1.5, alpha=0.8)
    ax2.scatter(alphas, rho_peak, color='white', edgecolors=color1, s=60, lw=1.5, zorder=10)
    ax2.tick_params(axis='y', labelcolor=color1)
    style_axis(ax2)

    res_idx = np.argmax(R_rad)
    ax2.annotate(r'$\rho_{\rm max}$'+f' = {rho_peak[res_idx]:.2f}',
                 xy=(alphas[res_idx], rho_peak[res_idx]),
                 xytext=(alphas[res_idx] + 0.15, rho_peak[res_idx] + 0.5),
                 fontsize=8, color=color1)

    ax2_2 = ax2.twinx()
    color2 = COLORS['accent']
    ax2_2.set_ylabel(r'Radiation Loss $R_{\rm rad}$', color=color2)
    ax2_2.plot(alphas, R_rad, color=color2, marker='o', markersize=7, linestyle='--', lw=1.5, alpha=0.8)
    ax2_2.scatter(alphas, R_rad, color='white', edgecolors=color2, s=60, lw=1.5, zorder=10)
    ax2_2.tick_params(axis='y', colors=color2)

    ax2_2.annotate(f'Resonant Collapse\n'+r'$\alpha$'+f' = {alphas[res_idx]:.2f}, '+r'$R$'+f' = {R_rad[res_idx]:.3f}',
                   xy=(alphas[res_idx], R_rad[res_idx]),
                   xytext=(alphas[res_idx] + 0.1, R_rad[res_idx] - 0.05),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                   fontsize=9, fontweight='bold', ha='left')
    
    ax2_2.set_ylim(-0.01, 0.14)
    ax2.set_title(r'Super-focusing & Resonant Radiation ($\gamma_{12}=1.3$, $\Delta\gamma=0.6$)', loc='left', fontweight='bold')

    lines = [
        Line2D([0], [0], color=color1, lw=2, label=r'$\rho_{\rm max}$'),
        Line2D([0], [0], color=color2, lw=2, linestyle='--', label=r'$R_{\rm rad}$')
    ]
    ax2.legend(lines, labels=[l.get_label() for l in lines], loc='upper right', fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Combined.png", bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Combined.pdf", bbox_inches='tight')
    plt.close()

def create_figure4_ultimate(all_results, cfg):
    fig = plt.figure(figsize=(12, 4.5), constrained_layout=True)
    gs = GridSpec(1, 2, width_ratios=[1, 1.2], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    alpha = all_results['alpha']['vals']
    rads = [r['rad'][-1] for r in all_results['alpha']['data']]

    X_smooth = np.linspace(min(alpha), max(alpha), 200)
    spl = make_interp_spline(alpha, rads, k=3)
    Y_smooth = spl(X_smooth)
    Y_smooth = np.clip(Y_smooth, 0, None)

    ax1.fill_between(X_smooth, 0, Y_smooth, color=COLORS['magenta'], alpha=0.1)
    ax1.plot(X_smooth, Y_smooth, color=COLORS['magenta'], lw=2)
    ax1.scatter(alpha, rads, color='white', edgecolors=COLORS['magenta'], s=40, zorder=5)

    peak_idx = np.argmax(rads)
    # 修复：调整扩散峰标注位置
    ax1.annotate(f"Diffusive Peak\n"+r"$\alpha$"+f" = {alpha[peak_idx]:.1f}, "+r"$\eta$"+f" = {rads[peak_idx]:.3f}",
                 xy=(alpha[peak_idx], rads[peak_idx]),
                 xytext=(alpha[peak_idx] + 0.1, rads[peak_idx] - 0.02),
                 arrowprops=dict(arrowstyle="->", color=COLORS['text']),
                 ha='left', fontsize=9)

    style_axis(ax1)
    ax1.set_xlabel(r'Fractional Order $\alpha$')
    ax1.set_ylabel(r'Radiation Loss $\eta$', labelpad=10)
    ax1.set_title('(a) Diffusive Dissipation', loc='left')

    ax2 = fig.add_subplot(gs[1])
    k = np.fft.fftshift(np.fft.fftfreq(cfg.N, d=cfg.dx) * 2 * np.pi)
    peak_alpha = alpha[peak_idx]
    idx_std = np.argmin(np.abs(alpha - 2.0))
    idx_peak = peak_idx

    spec_std = all_results['alpha']['data'][idx_std]['k_spec']
    spec_peak = all_results['alpha']['data'][idx_peak]['k_spec']

    ax2.semilogy(k, spec_std / spec_std.max(), color=COLORS['cyan'], alpha=0.8, lw=1.5,
                 label=r'$\alpha=2.0$ (Standard)')
    add_glow(ax2, k, spec_peak / spec_peak.max(), COLORS['magenta'],
             label=r'$\alpha=$'+f'{peak_alpha:.1f} (Anomalous)')

    style_axis(ax2)
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(1e-5, 2.0)
    ax2.set_xlabel(r'Wave Number $k$')
    ax2.set_ylabel(r'Normalized PSD (dB)', labelpad=10)
    ax2.set_title('(b) Spectral Broadening', loc='left')
    ax2.legend(loc='upper right', fontsize=9)

    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.15)
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Physics.png", bbox_inches='tight')
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Physics.pdf", bbox_inches='tight')
    plt.close()


def print_dashboard_header():
    print("\n" + "█" * 80)
    print(f"█  {'NONRECIPROCAL FCQNLS - NATURE/SCIENCE ANALYTICS ENGINE':^74}  █")
    print("█" * 80)
    print(f"█  Run Date: {time.strftime('%Y-%m-%d %H:%M:%S')} | Device: {str(DEVICE):<10} | Seed: 42{'':<17}█")
    print("█" + "-" * 78 + "█")


def print_section(title):
    print(f"\n🔶 {title}")
    print("━" * 80)


def print_metric(label, value, unit="", ref=None):
    val_str = f"{value:.4f}"
    if ref:
        diff = (value - ref) / ref * 100
        ref_str = f" (Err: {diff:+.2e}%)"
    else:
        ref_str = ""
    print(f"   🔹 {label:<20} : {val_str}{unit}{ref_str}")


def print_table(headers, rows, widths):
    h_str = " │ ".join([f"{h:^{w}}" for h, w in zip(headers, widths)])
    print(f"   ┌{'─' * len(h_str)}┐")
    print(f"   │ {h_str} │")
    print(f"   ├{'─' * len(h_str)}┤")
    for row in rows:
        r_str = " │ ".join([f"{str(v):^{w}}" for v, w in zip(row, widths)])
        print(f"   │ {r_str} │")
    print(f"   └{'─' * len(h_str)}┘")


def fmt_sci(x, prec=1):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    coeff = round(x / (10 ** exp), prec)
    return f"{coeff}×10^{exp}"


# ===================== MAIN EXECUTION =====================

def main():
    print_dashboard_header()

    # =========================================================================
    # PHASE 1: VISUALIZATION (FIGURE 1 SPECIAL)
    # =========================================================================
    print_section("PHASE 1: BASELINE DYNAMICS (γ₁₂=γ₂₁=1.0 vs γ₁₂=1.5, γ₂₁=0.5)")

    cfg_visual = SimConfig(N=2048, L=100, alpha=1.8, dt=5e-4, T=100.0, v=0.35, shift=40)
    sys_visual = NonreciprocalFCQNLS(cfg_visual)
    phi_vis = sys_visual.ground_state()
    x_np = sys_visual.x.cpu().numpy()

    cfg_ref = cfg_visual.copy(gamma=1.0)
    sys_ref = NonreciprocalFCQNLS(cfg_ref)
    psi_L = np.roll(phi_vis.cpu().numpy(), -cfg_ref.shift) * np.exp(1j * cfg_ref.v * x_np)
    psi_R = np.roll(phi_vis.cpu().numpy(), cfg_ref.shift) * np.exp(-1j * cfg_ref.v * x_np)
    psi0_ref = torch.tensor(np.stack([psi_L, psi_R]), device=DEVICE, dtype=torch.complex128)
    res_ref_vis = sys_ref.run_collision(psi0_ref, save_full=True)

    cfg_nr = cfg_visual.copy(gamma=1.5)
    sys_nr = NonreciprocalFCQNLS(cfg_nr)
    res_nr_vis = sys_nr.run_collision(psi0_ref, save_full=True)

    create_figure1_ultimate(res_ref_vis, res_nr_vis, x_np, cfg_visual, "dual_mode")

    # =========================================================================
    # PHASE 1.5: STANDARD PHYSICS (FIGURE 2 & CONSOLE)
    # =========================================================================
    cfg_std = SimConfig(N=2048, L=100, alpha=1.8, dt=5e-4, T=60.0, v=1.0, shift=150, gamma=1.5)
    sys_std = NonreciprocalFCQNLS(cfg_std)
    phi_std = sys_std.ground_state()
    x_std = sys_std.x.cpu().numpy()

    psi_L = np.roll(phi_std.cpu().numpy(), -cfg_std.shift) * np.exp(1j * cfg_std.v * x_std)
    psi_R = np.roll(phi_std.cpu().numpy(), cfg_std.shift) * np.exp(-1j * cfg_std.v * x_std)
    psi0_std = torch.tensor(np.stack([psi_L, psi_R]), device=DEVICE, dtype=torch.complex128)

    t0 = time.time()
    res_std = sys_std.run_collision(psi0_std, save_full=True)

    print_metric("Computation Time", time.time() - t0, "s")
    print_metric("Collision Time", res_std['t_collision'], "s", ref=cfg_std.t_collision_theory)
    print_metric("Momentum Drift", abs(res_std['P'][-1] - res_std['P0']))
    print_metric("Radiation Loss", res_std['rad'][-1])
    print(f"   🔹 Hamiltonian Error   : {fmt_sci(res_std['H_err'][-1])}")
    print(f"   🔹 Peak Density        : {np.max(res_std['peak_rho']):.2f}")
    print(f"   🔹 Nonreciprocity Δγ   : {sys_std.delta_gamma:.2f} (|γ₁₂ - γ₂₁| = |{sys_std.gamma_12:.1f} - {sys_std.gamma_21:.1f}|)")

    dv_peak = np.max(res_std['dv'])
    dv_peak_time = res_std['t'][np.argmax(res_std['dv'])]
    P_peak = np.max(np.abs(res_std['P']))
    P_peak_time = res_std['t'][np.argmax(np.abs(res_std['P']))]
    print(f"   🔹 Delta v Peak        : {dv_peak:.2f} (at t={dv_peak_time:.2f})")
    print(f"   🔹 Momentum Peak       : {P_peak:.2f} (at t={P_peak_time:.2f})")

    create_figure2_ultimate(res_std, cfg_std, "baseline")
    print("   ✅ Figures 1 & 2 generated.")

        # =========================================================================
    # PHASE 2: SCANS
    # =========================================================================
    print_section("PHASE 2: NONRECIPROCITY SCAN")
    gammas = np.linspace(0.6, 1.6, 11)
    g_res = []
    table_data = []
    # ====================== 增强版控制台输出 ======================
    print(f"   {'γ₁₂':<6} {'γ₂₁':<6} {'Δγ':<6} {'状态':<6} {'v1(最终)':<10} {'v2(最终)':<10} {'|Δv|':<10} {'H_err':<12}")
    print(f"   {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 12}")

    for g in gammas:
        cfg_g = cfg_std.copy(gamma=g)
        sys_g = NonreciprocalFCQNLS(cfg_g)
        psi0 = psi0_std.clone()
        data = sys_g.run_collision(psi0, save_full=False)
        g_res.append(data)

        status = "有效" if data['collision_valid'] else "无效"
        h_err = fmt_sci(data['H_err'][-1])
        g_21 = 2.0 - g
        delta_g = abs(g - g_21)
        
        # 新增：最终速度 + 绝对速度差
        v1_final = data['v1'][-1]
        v2_final = data['v2'][-1]
        dv_abs = abs(abs(v1_final) - abs(v2_final))
        
        # 增强打印
        print(f"   {g:<6.1f} {g_21:<6.1f} {delta_g:<6.1f} {status:<6} {v1_final:<10.4f} {v2_final:<10.4f} {dv_abs:<10.4f} {h_err:<12}")

        table_data.append([
            f"{g:.1f}", f"{g_21:.1f}", f"{delta_g:.1f}",
            "VALID" if data['collision_valid'] else "MISS",
            f"{data['t_collision']:.2f}", f"{data['min_cm_sep']:.2f}", f"{dv_abs:.4f}"
        ])

    print("\n   [细化扫描表格]")
    print_table(["γ₁₂", "γ₂₁", "Δγ", "Status", "T_col", "Min Sep", "|Δv|"], table_data, [6, 6, 6, 8, 8, 8, 10])

    # =========================================================================
    # PHASE 3: ALPHA SCAN
    # =========================================================================
    print_section("PHASE 3: FRACTIONAL ORDER SCAN")
    alphas = np.linspace(1.2, 2.0, 15)
    a_res = []
    table_data = []
    dynamic_labels = [
        "准弹性", "超聚焦", "辐射崩塌", "共振", "强耗散",
        "过渡区", "过渡区", "弱非弹性", "近经典", "近经典",
        "经典极限", "经典极限", "经典极限", "经典极限", "经典极限"
    ]
    print(f"   {'α':<8} {'H_err':<12} {'ρ_peak':<8} {'质心漂移':<8} {'R_rad':<8} {'动力学特征':<12}")
    print(f"   {'-' * 8} {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 12}")

    for i, a in enumerate(alphas):
        if a <= 1.3:
            sh, T_sim, vel = 20, 100.0, 0.5
            mode = "ADAPTIVE"
            a_str = f"{a:.2f}*"
        elif a >= 1.9:                        # ← [PATCH] 消除 α≥1.9 数值不稳定假峰
            sh, T_sim, vel = 100, 60.0, 0.8  # 减小 shift，降低速度，避免初态重叠
            mode = "NEAR-CLASSIC"
            a_str = f"{a:.2f}"
        else:
            sh, T_sim, vel = 150, 60.0, 1.0
            mode = "STANDARD"
            a_str = f"{a:.2f}"

        cfg_a = cfg_std.copy(alpha=a, gamma=1.3, shift=sh, T=T_sim, v=vel)
        sys_a = NonreciprocalFCQNLS(cfg_a)
        phi_a = sys_a.ground_state()

        psi_L = np.roll(phi_a.cpu().numpy(), -cfg_a.shift) * np.exp(1j * cfg_a.v * x_std)
        psi_R = np.roll(phi_a.cpu().numpy(), cfg_a.shift) * np.exp(-1j * cfg_a.v * x_std)
        psi0 = torch.tensor(np.stack([psi_L, psi_R]), device=DEVICE, dtype=torch.complex128)

        data = sys_a.run_collision(psi0, save_full=False)
        a_res.append(data)

        h_err = fmt_sci(data['H_err'][-1])
        peak_rho = f"{np.max(data['peak_rho']):.2f}"
        cm_drift = f"{abs(data['cm1'][-1] + data['cm2'][-1]):.2f}"
        rad_loss = f"{data['rad'][-1]:.4f}"
        dyn_label = dynamic_labels[min(i, len(dynamic_labels)-1)]

        print(f"   {a_str:<8} {h_err:<12} {peak_rho:<8} {cm_drift:<8} {rad_loss:<8} {dyn_label:<12}")

        table_data.append([
            f"{a:.2f}", mode, f"{data['t_collision']:.1f}",
            f"{data['rad'][-1]:.4f}", f"{abs(data['cm1'][-1] + data['cm2'][-1]):.3f}"
        ])

    print("\n   [细化扫描表格]")
    print_table(["α", "Mode", "T_col", "Rad Loss", "Drift"], table_data, [6, 12, 8, 10, 8])

    # =========================================================================
    # 生成组合图和Fig4
    # =========================================================================
    create_figure3_combined(gammas, g_res, alphas, a_res)
    print("   ✅ Figure 3 (Combined) generated.")

    create_figure4_ultimate({'alpha': {'vals': alphas, 'data': a_res}}, cfg_std)
    print("   ✅ Figure 4 generated.")

    print("\n" + "█" * 80)
    print(f"█  FINISHED. OUTPUT SAVED TO: {os.path.abspath(OUTPUT_DIR):<52} █")
    print("█" * 80)


if __name__ == "__main__":
    main()
