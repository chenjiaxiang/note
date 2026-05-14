"""
Generate ASR technology diagrams for the blog posts.
All diagrams are schematic/synthetic — no external data needed.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Font setup ───────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
_cjk_candidates = [f.name for f in fm.fontManager.ttflist
                   if any(k in f.name for k in ('CJK', 'SimHei', 'SimSun',
                                                  'WenQuanYi', 'Noto Sans CJK',
                                                  'Noto Serif CJK'))]
CJK_FONT = _cjk_candidates[0] if _cjk_candidates else None
print(f"Using CJK font: {CJK_FONT}")

def get_font_props(size=11):
    if CJK_FONT:
        return {'fontname': CJK_FONT, 'fontsize': size}
    return {'fontsize': size}

def fp(size=11):
    """Short helper to get font kwargs."""
    return get_font_props(size)

OUT = '/root/blog/note/docs/speech/asr/images/'
DPI = 180

# ─────────────────────────────────────────────────────────────────────────────
# 1. hmm_gmm_pipeline.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_hmm_gmm_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis('off')
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    steps = [
        ('音频输入', '#4A90D9'),
        ('MFCC特征', '#5BAD6F'),
        ('HMM-GMM\n声学模型', '#E07B54'),
        ('WFST解码', '#9B59B6'),
        ('文字输出', '#E74C3C'),
    ]
    n = len(steps)
    box_w = 1.7
    box_h = 1.1
    gap = (12 - n * box_w) / (n + 1)
    y_center = 1.5

    for i, (label, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        fancy = FancyBboxPatch((x, y_center - box_h / 2), box_w, box_h,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor='white',
                               linewidth=2, zorder=3)
        ax.add_patch(fancy)
        ax.text(x + box_w / 2, y_center, label,
                ha='center', va='center', color='white',
                fontweight='bold', zorder=4, **fp(10))

        if i < n - 1:
            ax_x = x + box_w + 0.05
            ax.annotate('', xy=(ax_x + gap - 0.05, y_center),
                        xytext=(ax_x, y_center),
                        arrowprops=dict(arrowstyle='->', color='#555555',
                                        lw=2.0),
                        zorder=5)

    ax.set_title('HMM-GMM 语音识别流水线', **fp(13), pad=10, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUT + 'hmm_gmm_pipeline.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved hmm_gmm_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. hmm_states.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_hmm_states():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             gridspec_kw={'width_ratios': [1.6, 1]})
    fig.patch.set_facecolor('#F8F9FA')

    # ── left: state diagram ──
    ax = axes[0]
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.5, 4.5)
    ax.axis('off')
    ax.set_facecolor('#F8F9FA')

    state_x = [1.5, 4.0, 6.5]
    state_y = [2.5, 2.5, 2.5]
    state_labels = ['s₁', 's₂', 's₃']
    colors = ['#4A90D9', '#5BAD6F', '#E07B54']
    radius = 0.55

    for i, (sx, sy, sl, col) in enumerate(zip(state_x, state_y, state_labels, colors)):
        circle = plt.Circle((sx, sy), radius, color=col, zorder=3)
        ax.add_patch(circle)
        ax.text(sx, sy, sl, ha='center', va='center', color='white',
                fontsize=14, fontweight='bold', zorder=4)

    # Self-loop arrows
    for i, (sx, sy, col) in enumerate(zip(state_x, state_y, colors)):
        theta = np.linspace(0.3, 2 * np.pi - 0.3, 60)
        lx = sx + 0.62 * np.cos(theta + np.pi / 2)
        ly = sy + 0.62 * np.sin(theta + np.pi / 2) + 0.25
        ax.plot(lx, ly, color=col, lw=1.8, zorder=2)
        ax.annotate('', xy=(sx + 0.35, sy + 0.55),
                    xytext=(sx + 0.55, sy + 0.75),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5))
        probs = ['0.6', '0.5', '0.7']
        ax.text(sx, sy + 1.38, f'a_ii={probs[i]}', ha='center', va='center',
                fontsize=8.5, color=col)

    # Transition arrows between states
    trans = [(0, 1, '0.4'), (1, 2, '0.5')]
    for i, j, p in trans:
        x0, y0 = state_x[i] + radius + 0.05, state_y[i]
        x1, y1 = state_x[j] - radius - 0.05, state_y[j]
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.0))
        ax.text((x0 + x1) / 2, y0 + 0.22, f'a={p}', ha='center',
                fontsize=9, color='#555')

    # Entry/exit
    ax.annotate('', xy=(state_x[0] - radius - 0.05, state_y[0]),
                xytext=(0.2, state_y[0]),
                arrowprops=dict(arrowstyle='->', color='#222', lw=1.8))
    ax.annotate('', xy=(7.8, state_y[2]),
                xytext=(state_x[2] + radius + 0.05, state_y[2]),
                arrowprops=dict(arrowstyle='->', color='#222', lw=1.8))
    ax.text(0.0, 2.5, '开始', va='center', **fp(9))
    ax.text(7.85, 2.5, '结束', va='center', **fp(9))

    ax.set_title('HMM Triphone 状态图', **fp(12), fontweight='bold', pad=6)

    # ── right: GMM distributions ──
    ax2 = axes[1]
    ax2.set_facecolor('#F8F9FA')
    x = np.linspace(-4, 4, 300)
    gmm_params = [(-2, 0.6, 0.35), (0, 0.8, 0.5), (2, 0.7, 0.4)]
    total = np.zeros_like(x)
    c_list = ['#4A90D9', '#5BAD6F', '#E07B54']
    for (mu, sig, w), c in zip(gmm_params, c_list):
        g = w * np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        ax2.fill_between(x, g, alpha=0.3, color=c)
        ax2.plot(x, g, color=c, lw=1.8, label=f'G{c_list.index(c)+1}')
        total += g
    ax2.plot(x, total, 'k--', lw=2, label='GMM')
    ax2.legend(fontsize=8.5, loc='upper right')
    ax2.set_xlabel('特征维度', **fp(9))
    ax2.set_ylabel('概率密度', **fp(9))
    ax2.set_title('每个状态的 GMM 输出概率', **fp(11), fontweight='bold', pad=6)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT + 'hmm_states.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved hmm_states.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. dnn_hmm_hybrid.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_dnn_hmm_hybrid():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#F8F9FA')

    def draw_side(ax, title, acoustic_label, acoustic_color, layers_info):
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 8)
        ax.axis('off')
        ax.set_facecolor('#F8F9FA')

        # Feature box
        feat = FancyBboxPatch((1.5, 6.5), 3, 0.8,
                              boxstyle='round,pad=0.1',
                              facecolor='#4A90D9', edgecolor='white', lw=2)
        ax.add_patch(feat)
        ax.text(3, 6.9, '声学特征 (MFCC)', ha='center', va='center',
                color='white', fontweight='bold', **fp(10))

        # Acoustic model
        am = FancyBboxPatch((0.8, 3.5), 4.4, 2.5,
                            boxstyle='round,pad=0.1',
                            facecolor=acoustic_color, edgecolor='white',
                            lw=2, alpha=0.85)
        ax.add_patch(am)

        # Internal layers
        for li, (ly, lh, lc, ll) in enumerate(layers_info):
            lb = FancyBboxPatch((1.1, ly), 3.8, lh,
                                boxstyle='round,pad=0.06',
                                facecolor=lc, edgecolor='white', lw=1.5)
            ax.add_patch(lb)
            ax.text(3, ly + lh / 2, ll, ha='center', va='center',
                    color='white', **fp(9))

        ax.text(3, 6.2, '', ha='center', va='center')
        ax.annotate('', xy=(3, 6.45), xytext=(3, 6.08),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
        ax.annotate('', xy=(3, 3.45), xytext=(3, 3.05),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))

        # HMM box
        hmm = FancyBboxPatch((1.5, 1.5), 3, 1.3,
                             boxstyle='round,pad=0.1',
                             facecolor='#9B59B6', edgecolor='white', lw=2)
        ax.add_patch(hmm)
        ax.text(3, 2.15, 'HMM 解码', ha='center', va='center',
                color='white', fontweight='bold', **fp(10))

        ax.annotate('', xy=(3, 1.45), xytext=(3, 1.1),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=1.8))
        out = FancyBboxPatch((1.5, 0.2), 3, 0.75,
                             boxstyle='round,pad=0.08',
                             facecolor='#E74C3C', edgecolor='white', lw=2)
        ax.add_patch(out)
        ax.text(3, 0.575, '文字输出', ha='center', va='center',
                color='white', fontweight='bold', **fp(10))

        ax.set_title(title, **fp(12), fontweight='bold', pad=8)

        # Acoustic model label
        ax.text(3, 4.7, acoustic_label, ha='center', va='center',
                color='white', fontweight='bold', **fp(10))

    # GMM-HMM side
    gmm_layers = [
        (4.2, 0.7, '#7FB3D3', 'GMM 混合高斯'),
        (5.1, 0.7, '#5A9EC9', 'GMM 混合高斯'),
    ]
    draw_side(axes[0], 'GMM-HMM（传统）', '声学模型\n(GMM)', '#5BA4CF', [
        (4.1, 0.65, '#85C1E9', '高斯分量 1~K'),
        (5.0, 0.65, '#5DADE2', '高斯分量 1~K'),
    ])

    # DNN-HMM side
    draw_side(axes[1], 'DNN-HMM（深度混合）', '声学模型\n(DNN)', '#E07B54', [
        (4.1, 0.55, '#F0A07A', 'FC Layer 1 (ReLU)'),
        (4.8, 0.55, '#E8835A', 'FC Layer 2 (ReLU)'),
        (5.5, 0.55, '#E06030', 'Softmax → P(state|x)'),
    ])

    # Center label
    fig.text(0.5, 0.02,
             '关键变化：用 DNN 的判别式建模替代 GMM 的生成式建模，其余框架不变',
             ha='center', **fp(10), color='#333')

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(OUT + 'dnn_hmm_hybrid.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved dnn_hmm_hybrid.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ctc_alignment.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_ctc_alignment():
    # Show a CTC grid: rows = labels (+ blank), cols = time frames
    labels = ['<b>', 'h', 'e', 'l', 'l', 'o']   # <b> = blank
    T = 10   # time frames
    n_labels = len(labels)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    cell_w, cell_h = 0.9, 0.75
    # Draw grid
    for r in range(n_labels):
        for c in range(T):
            x0 = c * cell_w + 0.5
            y0 = r * cell_h + 0.5
            is_blank = (r == 0)
            color = '#DCE9F5' if is_blank else '#F5F5F5'
            rect = FancyBboxPatch((x0, y0), cell_w - 0.05, cell_h - 0.05,
                                  boxstyle='round,pad=0.04',
                                  facecolor=color, edgecolor='#BBBBBB', lw=0.8)
            ax.add_patch(rect)

    # Y-axis labels
    for r, lbl in enumerate(labels):
        color = '#E74C3C' if lbl == '<b>' else '#333333'
        ax.text(0.25, r * cell_h + 0.5 + cell_h / 2, lbl,
                ha='center', va='center', fontsize=11,
                color=color, fontweight='bold' if lbl == '<b>' else 'normal')

    # X-axis labels
    for c in range(T):
        ax.text(c * cell_w + 0.5 + cell_w / 2, 0.18,
                f't{c+1}', ha='center', va='center', fontsize=9, color='#555')

    # Two example CTC paths
    # Path 1: h e l <b> l o  (with blanks interspersed)
    path1 = [(0, 1), (1, 1), (2, 2), (3, 3), (4, 0), (5, 4), (6, 5), (7, 5), (8, 5), (9, 5)]
    # path: (time_col, label_row)
    path1 = [
        (0, 1), (1, 1), (2, 2), (3, 0), (4, 3), (5, 0), (6, 4), (7, 0), (8, 5), (9, 5)
    ]
    path2 = [
        (0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4), (6, 0), (7, 5), (8, 5), (9, 0)
    ]

    def draw_path(path, color, alpha=0.85, lw=2.5):
        for i in range(len(path)):
            c, r = path[i]
            cx = c * cell_w + 0.5 + cell_w / 2
            cy = r * cell_h + 0.5 + cell_h / 2
            # highlight cell
            rect = FancyBboxPatch((c * cell_w + 0.5 + 0.04, r * cell_h + 0.5 + 0.04),
                                  cell_w - 0.13, cell_h - 0.13,
                                  boxstyle='round,pad=0.04',
                                  facecolor=color, edgecolor='white',
                                  lw=1.0, alpha=alpha, zorder=3)
            ax.add_patch(rect)
            if i < len(path) - 1:
                nc, nr = path[i + 1]
                nx = nc * cell_w + 0.5 + cell_w / 2
                ny = nr * cell_h + 0.5 + cell_h / 2
                ax.annotate('', xy=(nx, ny), xytext=(cx, cy),
                            arrowprops=dict(arrowstyle='->', color=color,
                                            lw=lw, alpha=0.7),
                            zorder=4)

    draw_path(path1, '#3498DB', alpha=0.6, lw=2.0)
    draw_path(path2, '#E67E22', alpha=0.6, lw=2.0)

    # Legend
    p1 = mpatches.Patch(color='#3498DB', alpha=0.7, label='有效路径 1')
    p2 = mpatches.Patch(color='#E67E22', alpha=0.7, label='有效路径 2')
    pb = mpatches.Patch(color='#DCE9F5', label='<blank> 行')
    ax.legend(handles=[p1, p2, pb], loc='upper right', fontsize=9,
              framealpha=0.9)

    ax.set_xlim(0, T * cell_w + 0.8)
    ax.set_ylim(0, n_labels * cell_h + 0.7)
    ax.axis('off')
    ax.set_title('CTC 对齐：多条有效路径均解码为 "hello"', **fp(12),
                 fontweight='bold', pad=10)

    fig.text(0.5, 0.01,
             '所有经过 blank 折叠后得到同一标注的路径，其概率之和即为 CTC 损失的目标',
             ha='center', **fp(9), color='#555')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT + 'ctc_alignment.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved ctc_alignment.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. rnn_t_arch.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_rnn_t_arch():
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)

    def box(ax, x, y, w, h, label, color, fontsize=10):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor='white', lw=2, zorder=3)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                color='white', fontweight='bold', zorder=4, **fp(fontsize))

    # Audio Encoder (bottom, horizontal time steps)
    box(ax, 0.5, 0.4, 10, 1.1, '音频编码器 Encoder  (处理输入帧 x₁…xT)', '#4A90D9', 11)

    # Prediction network (left vertical)
    box(ax, 0.5, 2.5, 2.0, 3.8, '预测网络\nPrediction\nNetwork\n(语言模型头)', '#5BAD6F', 9)

    # Joint network (center)
    box(ax, 3.5, 3.5, 3.5, 1.8, '联合网络\nJoint Network\n(加法 + Tanh)', '#E07B54', 10)

    # Output (right)
    box(ax, 8.2, 3.5, 2.3, 1.8, '输出\n(Softmax)\n词汇表', '#9B59B6', 10)

    # Encoder → Joint
    ax.annotate('', xy=(3.5, 4.4), xytext=(2.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='#4A90D9', lw=2.2))
    ax.text(3.0, 4.62, 'hᵢ', ha='center', fontsize=10, color='#4A90D9')

    # Encoder upward
    ax.annotate('', xy=(1.5, 2.48), xytext=(1.5, 1.52),
                arrowprops=dict(arrowstyle='->', color='#4A90D9', lw=2.2))

    # Prediction → Joint
    ax.annotate('', xy=(3.5, 4.0), xytext=(2.52, 4.0),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=2.2))
    ax.text(3.0, 3.75, 'gᵤ', ha='center', fontsize=10, color='#5BAD6F')

    # Joint → Output
    ax.annotate('', xy=(8.18, 4.4), xytext=(7.02, 4.4),
                arrowprops=dict(arrowstyle='->', color='#E07B54', lw=2.2))

    # Output back to Prediction
    ax.annotate('', xy=(2.52, 5.8), xytext=(8.35, 5.8),
                arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=1.8,
                                connectionstyle='arc3,rad=-0.28'))
    ax.text(5.5, 6.55, '输出 token 反馈给预测网络', ha='center',
            color='#9B59B6', **fp(9))

    # Time axis
    for t in range(1, 8):
        ax.text(0.5 + t * 10 / 8, 0.08, f't={t}', ha='center',
                fontsize=8, color='#555')
    ax.text(5.5, -0.08, '→  时间帧轴', ha='center', fontsize=9, color='#333')

    # Label axis (vertical)
    for u in range(1, 5):
        ax.text(0.08, 2.5 + u * 3.8 / 5, f'u={u}', ha='center',
                fontsize=8, color='#555')

    ax.set_title('RNN-T 架构：流式端到端语音识别', **fp(13),
                 fontweight='bold', pad=10)

    # Note
    ax.text(5.5, 7.65,
            '优势：在不知道输出 token 数量的情况下，可以逐帧流式输出',
            ha='center', **fp(9), color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#F0D060', lw=1))

    fig.tight_layout()
    fig.savefig(OUT + 'rnn_t_arch.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved rnn_t_arch.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. conformer_block.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_conformer_block():
    fig, ax = plt.subplots(figsize=(7, 11))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 12)

    blocks = [
        ('输入', '#555555', 0.5),
        ('Feed Forward\n(×½)', '#4A90D9', 1.1),
        ('Multi-Head\nSelf-Attention', '#5BAD6F', 1.1),
        ('卷积模块\nConvolution Module', '#E07B54', 1.1),
        ('Feed Forward\n(×½)', '#4A90D9', 1.1),
        ('Layer Norm', '#9B59B6', 0.7),
        ('输出', '#555555', 0.5),
    ]

    bw = 3.4
    bx = (7 - bw) / 2
    y = 10.5
    centers = []

    for i, (label, color, bh) in enumerate(blocks):
        by = y - bh
        if label in ('输入', '输出'):
            # small rounded
            r = FancyBboxPatch((bx + 0.5, by), bw - 1, bh,
                               boxstyle='round,pad=0.1',
                               facecolor=color, edgecolor='white', lw=2, zorder=3)
        else:
            r = FancyBboxPatch((bx, by), bw, bh,
                               boxstyle='round,pad=0.12',
                               facecolor=color, edgecolor='white', lw=2,
                               alpha=0.88, zorder=3)
        ax.add_patch(r)
        cy = by + bh / 2
        centers.append((bx + bw / 2, cy))
        ax.text(bx + bw / 2, cy, label, ha='center', va='center',
                color='white', fontweight='bold', zorder=4, **fp(11))

        if i < len(blocks) - 1:
            next_top = y - bh - 0.0
            # arrow gap
            ax.annotate('', xy=(bx + bw / 2, by - 0.38),
                        xytext=(bx + bw / 2, by - 0.02),
                        arrowprops=dict(arrowstyle='->', color='#555', lw=1.8),
                        zorder=5)

        y = by - 0.4

    # Residual connections (skip arrows on left side)
    residual_pairs = [
        (1, 2, '#4A90D9', '残差'),  # FF1 skip
        (2, 3, '#5BAD6F', '残差'),  # MHSA skip
        (3, 4, '#E07B54', '残差'),  # Conv skip
        (4, 5, '#4A90D9', '残差'),  # FF2 skip
    ]

    for src, dst, col, lbl in residual_pairs:
        cx_s, cy_s = centers[src]
        cx_d, cy_d = centers[dst]
        # Draw residual on the right
        rx = bx + bw + 0.35
        ry_start = cy_s
        ry_end = cy_d
        ax.annotate('', xy=(rx, ry_end), xytext=(rx, ry_start),
                    arrowprops=dict(arrowstyle='->', color=col, lw=1.5,
                                    alpha=0.7))
        ax.plot([cx_s + bw / 2, rx], [cy_s, ry_start], color=col, lw=1.2, alpha=0.5)
        ax.plot([cx_d + bw / 2, rx], [cy_d, ry_end], color=col, lw=1.2, alpha=0.5,
                linestyle='--')

    ax.set_title('Conformer 模块结构', **fp(13), fontweight='bold', pad=10)

    # Note about macaron structure
    ax.text(3.5, 0.35,
            '"Macaron" 结构：两个半权重 FFN 夹住注意力和卷积',
            ha='center', **fp(9), color='#555',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#F0D060', lw=1))

    fig.tight_layout()
    fig.savefig(OUT + 'conformer_block.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved conformer_block.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. self_supervised_pretrain.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_self_supervised_pretrain():
    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)

    def box(x, y, w, h, label, color, fontsize=10):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor='white', lw=2, zorder=3)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                color='white', fontweight='bold', zorder=4, **fp(fontsize))

    def arr(x0, y0, x1, y1, color='#555'):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.0),
                    zorder=5)

    # Raw audio waveform (synthetic)
    t = np.linspace(0, 1, 200)
    wave = 0.28 * np.sin(2 * np.pi * 5 * t) * np.exp(-0.5 * t) + \
           0.15 * np.random.default_rng(42).standard_normal(200) * 0.08
    ax_ins = ax.inset_axes([0.01, 0.42, 0.12, 0.3])
    ax_ins.plot(t, wave, color='#4A90D9', lw=1.2)
    ax_ins.set_facecolor('#EEF4FB')
    ax_ins.axis('off')
    ax_ins.set_title('原始\n音频', fontsize=7, pad=2)

    arr(1.6, 3.0, 2.1, 3.0)

    # CNN Encoder
    box(2.1, 2.2, 2.0, 1.6, 'CNN\n特征编码器', '#4A90D9', 10)
    arr(4.12, 3.0, 4.62, 3.0)

    # Quantization
    box(4.62, 2.2, 2.2, 1.6, '量化模块\n(Codebook)', '#5BAD6F', 10)

    # Arrow up to contrastive
    arr(5.72, 3.82, 5.72, 4.52)
    ax.text(5.72, 4.7, '量化表示 q', ha='center', color='#5BAD6F', **fp(9))

    arr(6.84, 3.0, 7.34, 3.0)

    # Transformer with masking
    box(7.34, 2.0, 3.2, 2.2, 'Transformer\n(部分帧被遮蔽 ████)', '#E07B54', 10)

    # Masked frames indicator
    for mi in [7.7, 8.3, 9.2]:
        r = FancyBboxPatch((mi, 2.05), 0.38, 0.55,
                           boxstyle='round,pad=0.04',
                           facecolor='#888888', edgecolor='white', lw=1, zorder=4)
        ax.add_patch(r)

    arr(10.56, 3.0, 11.1, 3.0)

    # Contrastive loss
    box(11.1, 2.2, 1.7, 1.6, '对比\n损失', '#E74C3C', 10)

    # Arrow from quantized to contrastive
    ax.annotate('', xy=(11.95, 2.18), xytext=(5.72, 4.4),
                arrowprops=dict(arrowstyle='->', color='#5BAD6F', lw=1.8,
                                connectionstyle='arc3,rad=-0.22'),
                zorder=5)

    # Labels
    ax.text(6.5, 1.4,
            '预训练阶段：无需标注，仅用原始音频',
            ha='center', **fp(10), color='#333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                      edgecolor='#81C784', lw=1.5))

    ax.set_title('wav2vec 2.0 自监督预训练流程', **fp(13),
                 fontweight='bold', pad=10)

    fig.tight_layout()
    fig.savefig(OUT + 'self_supervised_pretrain.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved self_supervised_pretrain.png")


# ─────────────────────────────────────────────────────────────────────────────
# 8. whisper_data_scale.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_whisper_data_scale():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')

    datasets = ['LibriSpeech', 'MLS', 'VoxPopuli', 'Whisper\n(OpenAI)']
    hours = [960, 50_000, 400_000, 680_000]
    colors = ['#5BAD6F', '#4A90D9', '#E07B54', '#E74C3C']

    x = np.arange(len(datasets))
    bars = ax.bar(x, hours, color=colors, edgecolor='white', linewidth=1.5,
                  width=0.55, zorder=3)

    # Value labels on bars
    for bar, h in zip(bars, hours):
        if h < 1000:
            label = f'{h}h'
        elif h < 1_000_000:
            label = f'{h/1000:.0f}Kh'
        else:
            label = f'{h/1_000_000:.1f}Mh'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.15,
                label, ha='center', va='bottom',
                fontweight='bold', color=colors[x.tolist()[list(hours).index(h)]],
                **fp(11))

    ax.set_yscale('log')
    ax.set_ylabel('训练音频时长 (小时，对数坐标)', **fp(11))
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, **fp(10))
    ax.set_title('主流语音数据集规模对比', **fp(13), fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.4, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation for Whisper
    ax.annotate('680K 小时\n≈ 99 种语言\n弱监督数据',
                xy=(3, 680_000), xytext=(2.35, 800_000),
                color='#E74C3C',
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FDEDEC',
                          edgecolor='#E74C3C', lw=1), **fp(9))

    fig.tight_layout()
    fig.savefig(OUT + 'whisper_data_scale.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved whisper_data_scale.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9. qwen_audio_arch.png
# ─────────────────────────────────────────────────────────────────────────────
def draw_qwen_audio_arch():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#F8F9FA')
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)

    def box(x, y, w, h, label, color, fontsize=10, alpha=0.9):
        r = FancyBboxPatch((x, y), w, h,
                           boxstyle='round,pad=0.1',
                           facecolor=color, edgecolor='white', lw=2,
                           alpha=alpha, zorder=3)
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center',
                color='white', fontweight='bold', zorder=4, **fp(fontsize))

    def arr(x0, y0, x1, y1, color='#555', lw=2.0, label=''):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw),
                    zorder=5)
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            ax.text(mx, my + 0.15, label, ha='center',
                    color=color, **fp(8))

    # Audio input
    box(0.3, 2.8, 1.5, 1.4, '音频\n输入', '#607D8B', 10)

    # Whisper encoder
    box(2.2, 2.4, 2.4, 2.2, 'Whisper\n音频编码器\n(冻结)', '#4A90D9', 10)

    # Linear projection
    box(5.2, 2.7, 1.8, 1.6, '线性\n投影层', '#5BAD6F', 10)

    # Text instruction input
    box(0.3, 5.0, 2.0, 1.2, '文字\n指令输入', '#78909C', 10)

    # Qwen LLM
    box(7.5, 2.0, 2.8, 2.8, 'Qwen\n大语言模型\n(LLM)', '#E07B54', 11)

    # Output tasks
    outputs = [
        (11.0, 5.2, 'ASR 转录', '#3498DB'),
        (11.0, 4.0, '语音理解', '#2ECC71'),
        (11.0, 2.8, '情感分析', '#E74C3C'),
        (11.0, 1.6, '多轮对话', '#9B59B6'),
    ]
    for ox, oy, olabel, oc in outputs:
        box(ox, oy, 1.8, 0.85, olabel, oc, 9)
        arr(10.32, oy + 0.43, ox - 0.02, oy + 0.43, color=oc, lw=1.6)

    # Arrows
    arr(1.82, 3.5, 2.18, 3.5, '#607D8B', label='原始音频')
    arr(4.62, 3.5, 5.18, 3.5, '#4A90D9', label='音频特征')
    arr(7.02, 3.5, 7.48, 3.5, '#5BAD6F', label='对齐表示')

    # Text instruction arrow
    ax.annotate('', xy=(8.2, 2.18), xytext=(1.3, 5.0),
                arrowprops=dict(arrowstyle='->', color='#78909C', lw=1.8,
                                connectionstyle='arc3,rad=0.3'),
                zorder=5)
    ax.text(4.0, 1.2, '指令 token 直接输入 LLM', ha='center',
            color='#78909C', **fp(9))

    ax.set_title('Qwen-Audio 架构：语音-语言统一多任务模型', **fp(13),
                 fontweight='bold', pad=10)

    ax.text(6.5, 0.3,
            '核心思路：冻结 Whisper 编码器，通过线性投影对齐语音 token 和文本 token 空间',
            ha='center', **fp(9), color='#444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                      edgecolor='#F0D060', lw=1))

    fig.tight_layout()
    fig.savefig(OUT + 'qwen_audio_arch.png', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print("Saved qwen_audio_arch.png")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)

    if CJK_FONT:
        matplotlib.rcParams['font.family'] = CJK_FONT

    draw_hmm_gmm_pipeline()
    draw_hmm_states()
    draw_dnn_hmm_hybrid()
    draw_ctc_alignment()
    draw_rnn_t_arch()
    draw_conformer_block()
    draw_self_supervised_pretrain()
    draw_whisper_data_scale()
    draw_qwen_audio_arch()

    print("\nAll 9 images generated successfully.")
