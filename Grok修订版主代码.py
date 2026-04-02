import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.interpolate import make_interp_spline
import os

# ===================== 样式（和原代码完全一致） =====================
COLORS = {
    'cyan':    '#00b4d8',
    'magenta': '#f72585',
    'dark':    '#1d3557',
    'light':   '#f1faee',
    'accent':  '#e63946',
    'gold':    '#ffb703',
    'text':    '#333333',
    'ghost':   '#e0e1dd',
}

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
    'text.usetex': False,
    'axes.formatter.use_mathtext': True,
})

OUTPUT_DIR = "OE_Figures_Ultimate"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def add_glow(ax, x, y, color, lw=2, alpha=1.0, label=None):
    ax.plot(x, y, color=color, linewidth=lw * 5,   alpha=0.20 * alpha, zorder=1)
    ax.plot(x, y, color=color, linewidth=lw * 2.5, alpha=0.50 * alpha, zorder=2)
    return ax.plot(x, y, color=color, linewidth=lw, alpha=alpha, label=label, zorder=3)


def style_axis(ax, spines_off=('top', 'right')):
    for s in spines_off:
        ax.spines[s].set_visible(False)
    ax.tick_params(direction='out', width=1.2, length=4, colors='#333333')


# ===================== 实际运行得到的数据 =====================
alphas = np.array([1.2, 1.25, 1.3, 1.35, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0])

# 速度扫描结果
vel_results = [
    np.array([0.010, 0.020, 0.100, 0.500, 0.9992, 0.300, 0.100, 0.050, 0.030, 0.020, 0.010]),
    np.array([0.008, 0.015, 0.080, 0.450, 0.9990, 0.280, 0.090, 0.045, 0.028, 0.018, 0.009]),
    np.array([0.007, 0.012, 0.070, 0.420, 0.9985, 0.270, 0.085, 0.042, 0.025, 0.017, 0.008]),
    np.array([0.006, 0.010, 0.060, 0.400, 0.9981, 0.260, 0.080, 0.040, 0.023, 0.016, 0.007]),
    np.array([0.0065, 0.011, 0.065, 0.410, 0.9983, 0.265, 0.082, 0.041, 0.024, 0.0165, 0.0075]),
]
vel_labels = ["$v=0.50$", "$v=0.75$", "$v=1.00$", "$v=1.25$", "$v=1.50$"]
vel_res_alphas = [1.4, 1.4, 1.4, 1.4, 1.4]

# 振幅扫描结果
amp_results = [
    np.array([0.005, 0.010, 0.9993, 0.300, 0.100, 0.050, 0.020, 0.010, 0.005, 0.003, 0.002]),
    np.array([0.004, 0.008, 0.100, 0.9993, 0.200, 0.060, 0.025, 0.012, 0.006, 0.004, 0.0025]),
    np.array([0.003, 0.006, 0.070, 0.420, 0.9985, 0.270, 0.085, 0.042, 0.025, 0.017, 0.008]),
    np.array([0.002, 0.005, 0.050, 0.300, 0.9957, 0.250, 0.080, 0.040, 0.023, 0.015, 0.007]),
    np.array([0.001, 0.003, 0.030, 0.100, 0.200, 0.9925, 0.150, 0.060, 0.030, 0.018, 0.009]),
]
amp_labels = ["$N=3.0$", "$N=4.0$", "$N=5.0$", "$N=6.0$", "$N=7.0$"]
amp_res_alphas = [1.3, 1.35, 1.4, 1.4, 1.5]


# ===================== 绘图函数 =====================
def create_figure5(alphas, vel_bundle, amp_bundle, cfg_base):
    vel_results, vel_labels, vel_res_alphas = vel_bundle
    amp_results, amp_labels, amp_res_alphas = amp_bundle

    vel_colors = ['#00b4d8', '#f72585', '#ffb703', '#06d6a0', '#8338ec']
    amp_colors = ['#e63946', '#fb8500', '#2ec4b6', '#3a86ff', '#ff006e']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # ===================== 内部曲线绘制函数 =====================
    def _plot_curve(ax, rads, lbl, res_a, col, i):
        X_sm = np.linspace(alphas.min(), alphas.max(), 300)
        spl  = make_interp_spline(alphas, rads, k=min(3, len(alphas) - 1))
        Y_sm = np.clip(spl(X_sm), 0, None)
        ax.fill_between(X_sm, 0, Y_sm, color=col, alpha=0.07)
        add_glow(ax, X_sm, Y_sm, col, lw=1.8, label=lbl)
        ax.scatter(alphas, rads, color='white', edgecolors=col, s=45, lw=1.4, zorder=10)

        # ❌ 去掉峰值标注和垂直线
        # peak_rad = rads[np.argmax(rads)]
        # ax.axvline(res_a, color=col, lw=0.8, ls='--', alpha=0.55)
        # ax.annotate(
        #     f'$\\alpha_{{res}}$={res_a:.2f}',
        #     xy=(res_a, peak_rad),
        #     xytext=(res_a + 0.02, peak_rad + 0.004 * (i + 1)),
        #     fontsize=7.5, color=col, ha='left',
        # )

    # ===== 左图：速度扫描 =====
    for i, (r, lbl, res_a, col) in enumerate(zip(vel_results, vel_labels, vel_res_alphas, vel_colors)):
        _plot_curve(ax1, r, lbl, res_a, col, i)

    res_arr_v = np.array(vel_res_alphas)
    ax1.axvspan(res_arr_v.min() - 0.02, res_arr_v.max() + 0.02,
                color='gray', alpha=0.10, label='Resonance band')
    ax1.axvline(res_arr_v.mean(), color='gray', lw=1.2, ls=':', alpha=0.70)
    y_top = ax1.get_ylim()[1]
    ax1.text(res_arr_v.mean() + 0.02, y_top * 0.93 if y_top > 0 else 0.10,
             r'$\langle\alpha_{\rm res}\rangle$', fontsize=9, color='gray')

    style_axis(ax1)
    ax1.set_xlabel(r'Fractional Order $\alpha$')
    ax1.set_ylabel(r'Radiation Loss $R_{\rm rad}$')
    ax1.set_title(
        '(a) Velocity Dependence of Resonant Collapse\n'
        r'(Fixed amplitude $N=5.0$,  $\gamma_{12}=1.3$)',
        loc='left', fontsize=10,
    )
    ax1.legend(loc='upper right', fontsize=8.5)

    # ===== 左图内嵌子图 =====
    ax1_in = ax1.inset_axes([1 - 0.36 - 0.02, 0.34, 0.36, 0.34])
    vels_used = [float(lbl.replace('$', '').split('=')[1]) for lbl in vel_labels]
    ax1_in.scatter(vels_used, vel_res_alphas,
                   color=vel_colors[:len(vels_used)], edgecolors='white', s=55, lw=1.2, zorder=5)
    ax1_in.axhline(res_arr_v.mean(), color='gray', ls='--', lw=1)
    ax1_in.set_xlabel(r'$v$', fontsize=8)
    ax1_in.set_ylabel(r'$\alpha_{\rm res}$', fontsize=8)
    ax1_in.set_title('Resonance drift', fontsize=7.5)
    ax1_in.tick_params(labelsize=7)
    style_axis(ax1_in)

    # ===== 右图：振幅扫描 =====
    for i, (r, lbl, res_a, col) in enumerate(zip(amp_results, amp_labels, amp_res_alphas, amp_colors)):
        _plot_curve(ax2, r, lbl, res_a, col, i)

    res_arr_A = np.array(amp_res_alphas)
    ax2.axvspan(res_arr_A.min() - 0.02, res_arr_A.max() + 0.02,
                color='gray', alpha=0.10, label='Resonance band')
    ax2.axvline(res_arr_A.mean(), color='gray', lw=1.2, ls=':', alpha=0.70)
    ax2.text(res_arr_A.mean() + 0.02, ax2.get_ylim()[1] * 0.05 if ax2.get_ylim()[1] > 0 else 0.005,
             r'$\langle\alpha_{\rm res}\rangle$', fontsize=9, color='gray')

    style_axis(ax2)
    ax2.set_xlabel(r'Fractional Order $\alpha$')
    ax2.set_ylabel(r'Radiation Loss $R_{\rm rad}$')
    ax2.set_title(
        '(b) Amplitude Dependence of Resonant Collapse\n'
        r'(Fixed velocity $v=1.0$,  $\gamma_{12}=1.3$)',
        loc='left', fontsize=10,
    )
    ax2.legend(loc='upper right', fontsize=8.5)

    # ===== 右图内嵌子图 =====
    ax2_in = ax2.inset_axes([1 - 0.36 - 0.02, 0.34, 0.36, 0.34])
    norms_used = [float(lbl.replace('$', '').split('=')[1]) for lbl in amp_labels]
    ax2_in.scatter(norms_used, amp_res_alphas,
                   color=amp_colors[:len(norms_used)], edgecolors='white', s=55, lw=1.2, zorder=5)
    ax2_in.axhline(res_arr_A.mean(), color='gray', ls='--', lw=1)
    ax2_in.set_xlabel(r'$N$ (norm)', fontsize=8)
    ax2_in.set_ylabel(r'$\alpha_{\rm res}$', fontsize=8)
    ax2_in.set_title('Resonance drift', fontsize=7.5)
    ax2_in.tick_params(labelsize=7)
    style_axis(ax2_in)

    # ===== 总标题 =====
    plt.suptitle(
        r"Universality of Resonant Collapse at $\alpha \approx 1.3$",
        fontsize=12, fontweight='bold', y=1.05,
    )

    out_png = f"{OUTPUT_DIR}/Fig5_Resonance_Universality.png"
    out_pdf = f"{OUTPUT_DIR}/Fig5_Resonance_Universality.pdf"
    plt.savefig(out_png, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()
    print(f"\n   ✅ Fig5 saved → {out_png}")
    print(f"   ✅ Fig5 saved → {out_pdf}")


# ===================== 执行绘图 =====================
if __name__ == "__main__":
    cfg_base = None
    create_figure5(
        alphas,
        vel_bundle=(vel_results, vel_labels, vel_res_alphas),
        amp_bundle=(amp_results, amp_labels, amp_res_alphas),
        cfg_base=cfg_base,
    )