"""Draw the model grid's error figures from the two tables beside this file.

sim.csv is what this simulator measured, v6e.csv the TPU row each point is compared against; a point whose
v6e row is marked use=0 is left out, with the reason in that row. Writes one PNG per family group under docs/.
"""
import argparse
import csv
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SS = [128, 256, 512, 1024, 2048, 4096]
LABEL = {128: "128", 256: "256", 512: "512", 1024: "1K", 2048: "2K", 4096: "4K"}
GRID, AXIS, SEP, EDGE = "#d0d0d0", "#000000", "#808080", "#000000"
plt.rcParams.update({"text.color": AXIS, "axes.labelcolor": AXIS, "xtick.color": AXIS,
                     "ytick.color": AXIS, "axes.edgecolor": AXIS, "font.size": 10.0})

NAME = {"llama2_7b": "Llama-2-7B", "llama3_8b": "Llama-3-8B", "llama31_8b": "Llama-3.1-8B",
        "llama4_scout": "Llama-4-Scout", "mixtral_8x7b": "Mixtral-8x7B", "gemma_7b": "Gemma-7B",
        "gemma2_2b": "Gemma-2-2B", "gemma3_1b": "Gemma-3-1B", "qwen3_8b": "Qwen3-8B",
        "qwen35_9b": "Qwen3.5-9B", "qwen15_moe": "Qwen1.5-MoE", "qwen3_30b": "Qwen3-30B-A3B",
        "qwen3_vl_moe": "Qwen3-VL-MoE"}
COLOR = {"llama2_7b": "#2f55c8", "llama3_8b": "#3d7cc9", "llama31_8b": "#7fa6de",
         "llama4_scout": "#aebdf0", "mixtral_8x7b": "#8a3b2b", "gemma_7b": "#1f8a7d",
         "gemma2_2b": "#53b0a3", "gemma3_1b": "#9fd6cd", "qwen3_8b": "#b8406e",
         "qwen35_9b": "#d4768f", "qwen15_moe": "#eaaac2", "qwen3_30b": "#6b4fa0",
         "qwen3_vl_moe": "#a98fd0"}
FAMILY = [("Llama · Mistral", ["llama2_7b", "llama3_8b", "llama31_8b", "llama4_scout", "mixtral_8x7b"]),
          ("Gemma", ["gemma_7b", "gemma2_2b", "gemma3_1b"]),
          ("Qwen", ["qwen3_8b", "qwen35_9b", "qwen15_moe", "qwen3_30b", "qwen3_vl_moe"])]
#: Each figure: the file it is written to, the families it holds, its tick step and how far it is squeezed.
FIGURES = [("grid-qwen", [FAMILY[2]], 10, 1.0),
           ("grid-llama-gemma", FAMILY[:2], 20, 0.6)]

LABEL_PT = 10.5
PER_PERCENT = 0.024
SIZES = (10.5, 10.0, 9.5, 9.0, 8.5, 8.0)
FRAME = {"frameon": True, "edgecolor": AXIS, "facecolor": "#ffffff", "framealpha": 1.0}


def read_errors(sim_path, v6e_path):
    """{(model, mode): {S: error percent}}, the modes named as the figures name them."""
    ref = {}
    for r in csv.DictReader(open(v6e_path)):
        if r["use"] == "1":
            ref[(r["model"], r["mode"], int(r["S"]))] = float(r["tpu_us"])
    out = {}
    for r in csv.DictReader(open(sim_path)):
        k = (r["model"], r["mode"], int(r["S"]))
        if k in ref and r["sim_us"]:
            mode = "prefill" if r["mode"] == "seq" else "decode"
            out.setdefault((k[0], mode), {})[k[2]] = (float(r["sim_us"]) / ref[k] - 1) * 100
    return out


def _text_in(text, size):
    """How many inches of the figure a run of text takes, measured rather than guessed."""
    fig = plt.figure(figsize=(1, 1), dpi=72)
    t = fig.text(0, 0, text, fontsize=size)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width / fig.dpi
    plt.close(fig)
    return w


#: A panel shorter than its own axis label leaves that label running past the frame, so the span grows
#: until it clears the longest of them -- the label is rotated, so its width is the height it needs.
MIN_PANEL_IN = _text_in("Decode error (%)", LABEL_PT) + 0.15


def _own_span(vals, step, scale):
    """The ticks and limits that hold these values, grown a step at a time until the panel they ask
    for is tall enough for its own axis label at `scale` inches per percent."""
    lo = math.floor(min(min(vals), -step) / step) * step
    hi = math.ceil(max(max(vals), step) / step) * step
    while (hi - lo) * scale < MIN_PANEL_IN:
        lo, hi = lo - step, hi + step
    return list(range(lo, hi + 1, step)), (lo - step * 0.3, hi + step * 0.3)


def _frame(ax, ticks, ylim):
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.set_ylim(*ylim)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v}%" for v in ticks], fontsize=9.5, color=AXIS)
    ax.tick_params(length=2.5, width=0.6, labelsize=9.5, labelcolor=AXIS, color=AXIS)
    for side in ax.spines.values():
        side.set_linewidth(0.5)
        side.set_color(AXIS)


def _fits(leg, ax):
    """Whether the legend is no wider than the panel it sits over, which is what keeps the figure
    from growing sideways when it is saved with a tight bounding box."""
    r = ax.figure.canvas.get_renderer()
    return leg.get_window_extent(r).width <= ax.get_window_extent().width


def _box(leg):
    leg.get_frame().set_linewidth(0.6)
    leg.get_frame().set_boxstyle("square", pad=0.3)


def _legend_above(ax, fams):
    """The legend over the panel, outside it: one row per family, shrunk until it fits the panel's
    width. The families are not named -- the colours group them and the caption says which they are."""
    cols = max(len(ms) for _, ms in fams)
    blank = lambda: plt.Line2D([], [], linestyle="none", marker="")
    cell = {}
    for r, (_, ms) in enumerate(fams):
        for c, m in enumerate(ms):
            cell[(r, c)] = (plt.Rectangle((0, 0), 1, 1, color=COLOR[m], ec=EDGE, lw=0.4), NAME[m])
    order = [cell.get((r, c), (blank(), "")) for c in range(cols) for r in range(len(fams))]
    for size in SIZES:
        leg = ax.legend([h for h, _ in order], [t for _, t in order], ncol=cols, loc="lower left",
                        bbox_to_anchor=(0, 1.03), fontsize=size, handlelength=0.85,
                        columnspacing=0.65, handletextpad=0.22, labelspacing=0.3,
                        borderpad=0.3, **FRAME)
        _box(leg)
        ax.figure.canvas.draw()
        if _fits(leg, ax):
            return


def figure_models(data, fams, step, squeeze, path):
    """Two panels, prefill over decode: a bar per model inside each sequence length, families set
    apart. Both panels take the same inches per percent, so one tick step is one distance."""
    names, slot, pos = [], 0.0, {}
    for _, fam in fams:
        for m in fam:
            if (m, "prefill") in data or (m, "decode") in data:
                names.append(m)
                pos[m] = slot
                slot += 1.0
        slot += 0.6
    span = slot - 0.6
    width = 0.9 / span
    vspan = {mo: _own_span([v for m in names for v in data.get((m, mo), {}).values()],
                           step, PER_PERCENT * squeeze) for mo in ("prefill", "decode")}
    tall = [(vspan[mo][1][1] - vspan[mo][1][0]) * squeeze for mo in ("prefill", "decode")]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 1.15 + PER_PERCENT * sum(tall)), sharex=True,
                             dpi=72, gridspec_kw={"height_ratios": tall})
    for ax, mode in zip(axes, ("prefill", "decode")):
        for m in names:
            vals = data.get((m, mode), {})
            off = (pos[m] - (span - 1) / 2) * width
            xs = [j + off for j, s in enumerate(SS) if s in vals]
            ax.bar(xs, [vals[s] for s in SS if s in vals], width=width * 0.9, color=COLOR[m],
                   edgecolor=EDGE, linewidth=0.4)
        for j in range(len(SS) - 1):
            ax.axvline(j + 0.5, color=SEP, lw=0.5)
        _frame(ax, *vspan[mode])
        ax.set_xlim(-0.55, len(SS) - 0.45)
        ax.set_ylabel(f"{mode.capitalize()} error (%)", fontsize=LABEL_PT, color=AXIS)
    axes[1].set_xticks(range(len(SS)))
    axes[1].set_xticklabels([LABEL[s] for s in SS], fontsize=9.5, rotation=-45, ha="left",
                            rotation_mode="anchor", color=AXIS)
    axes[1].set_xlabel("Sequence length", fontsize=LABEL_PT, labelpad=9, color=AXIS)
    _legend_above(axes[0], fams)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sim", default=os.path.join(HERE, "sim.csv"))
    ap.add_argument("--v6e", default=os.path.join(HERE, "v6e.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "docs"))
    args = ap.parse_args()
    os.makedirs(os.path.abspath(args.out), exist_ok=True)
    data = read_errors(args.sim, args.v6e)
    every = [v for d in data.values() for v in d.values()]
    for name, fams, step, squeeze in FIGURES:
        path = os.path.abspath(os.path.join(args.out, name + ".png"))
        figure_models(data, fams, step, squeeze, path)
        print("wrote", path)
    inside = sum(1 for v in every if abs(v) <= 15)
    print(f"{len(every)} points, {inside} within +-15%, MAE {sum(map(abs, every)) / len(every):.1f}%")


main()
