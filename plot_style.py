import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator


STYLE = {
    "font_family": "Microsoft YaHei",
    "font_size": 12,
    "title_size": 16,
    "label_size": 13,
    "tick_size": 11,
    "spine_width": 1.2,
    "line_width": 1.8,
    "figure_dpi": 120,
    "save_dpi": 600,
}


def get_red_blue_cmap(name="thesis_red_blue"):
    # negative -> zero -> positive: blue -> white -> red
    colors = ["#2C7BB6", "#FFFFFF", "#D7191C"]
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def get_signed_color_kwargs(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"cmap": get_red_blue_cmap(), "vmin": None, "vmax": None, "center": 0}

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return {"cmap": get_red_blue_cmap(), "vmin": None, "vmax": None, "center": 0}

    abs_max = float(np.max(np.abs(finite_values)))
    if abs_max == 0:
        return {"cmap": get_red_blue_cmap(), "vmin": None, "vmax": None, "center": 0}

    return {
        "cmap": get_red_blue_cmap(),
        "vmin": -abs_max,
        "vmax": abs_max,
        "center": 0,
    }


def apply_plot_style():
    mpl.rcParams.update(
        {
            "font.family": STYLE["font_family"],
            "font.size": STYLE["font_size"],
            "axes.titlesize": STYLE["title_size"],
            "axes.labelsize": STYLE["label_size"],
            "axes.linewidth": STYLE["spine_width"],
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "black",
            "xtick.labelsize": STYLE["tick_size"],
            "ytick.labelsize": STYLE["tick_size"],
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.major.width": STYLE["spine_width"],
            "ytick.major.width": STYLE["spine_width"],
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "lines.linewidth": STYLE["line_width"],
            "figure.dpi": STYLE["figure_dpi"],
            "savefig.dpi": STYLE["save_dpi"],
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    if title is not None:
        ax.set_title(title, pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_linewidth(STYLE["spine_width"])
    ax.spines["right"].set_linewidth(STYLE["spine_width"])
    ax.spines["left"].set_linewidth(STYLE["spine_width"])
    ax.spines["bottom"].set_linewidth(STYLE["spine_width"])
    ax.spines["top"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")


def style_colorbar(cbar, label=None, ticks=None, tick_format=None):
    if label is not None:
        cbar.set_label(label, fontsize=STYLE["label_size"])
    if ticks is not None:
        cbar.locator = FixedLocator(ticks)
        cbar.set_ticks(ticks)
    if tick_format is not None:
        cbar.formatter = FormatStrFormatter(tick_format)
        cbar.update_ticks()
    if ticks is not None:
        ticklabels = [tick_format % tick if tick_format is not None else str(tick) for tick in ticks]
        axis = cbar.ax.xaxis if getattr(cbar, "orientation", "horizontal") == "horizontal" else cbar.ax.yaxis
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_minor_locator(NullLocator())
        axis.set_ticks(ticks)
        axis.set_ticklabels(ticklabels)
        cbar.ax.minorticks_off()
    cbar.ax.tick_params(labelsize=STYLE["tick_size"], width=STYLE["spine_width"])
    for spine in cbar.ax.spines.values():
        spine.set_linewidth(STYLE["spine_width"])
    # 只显示最大值和最小值刻度
    if ticks is not None and len(ticks) >= 2:
        min_tick = ticks[0]
        max_tick = ticks[-1]
        if getattr(cbar, "orientation", "vertical") == "horizontal":
            cbar.ax.set_xticks([min_tick, max_tick])
            cbar.ax.set_xticklabels([tick_format % min_tick if tick_format else str(min_tick),
                                      tick_format % max_tick if tick_format else str(max_tick)])
            cbar.ax.set_yticks([])
        else:
            cbar.ax.set_yticks([min_tick, max_tick])
            cbar.ax.set_yticklabels([tick_format % min_tick if tick_format else str(min_tick),
                                      tick_format % max_tick if tick_format else str(max_tick)])
            cbar.ax.set_xticks([])
        cbar.ax.minorticks_off()


def force_colorbar_end_ticks(cbar, ticks, tick_format="%.3f"):
    if ticks is None:
        return

    # 只取最大值和最小值
    if len(ticks) >= 2:
        ticks = [ticks[0], ticks[-1]]

    ticklabels = [tick_format % tick for tick in ticks]
    cbar.set_ticks(ticks)

    if getattr(cbar, "orientation", "vertical") == "horizontal":
        cbar.ax.set_xticks(ticks)
        cbar.ax.set_xticklabels(ticklabels)
        cbar.ax.set_yticks([])
        cbar.ax.xaxis.set_minor_locator(NullLocator())
    else:
        cbar.ax.set_yticks(ticks)
        cbar.ax.set_yticklabels(ticklabels)
        cbar.ax.set_xticks([])
        cbar.ax.yaxis.set_minor_locator(NullLocator())

    cbar.ax.minorticks_off()


def save_figure(fig, path):
    fig.savefig(path, dpi=STYLE["save_dpi"], bbox_inches="tight")
