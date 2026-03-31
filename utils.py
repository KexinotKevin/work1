from joypy import joyplot
import os
import seaborn as sns
import matplotlib.pyplot as plt

from plot_style import apply_plot_style, get_red_blue_cmap, style_axes, save_figure


def draw_ridge_distrib(df_long, dataset_type, res_dir):
    apply_plot_style()
    rb_cmap = get_red_blue_cmap()
    fig, axes = joyplot(
        df_long,
        by="intell_category",
        column="scores",
        figsize=(12, 6),
        linecolor="white",
        colormap=rb_cmap,
        background="#FFFFFF",
    )
    fig.set_facecolor("#FFFFFF")
    plt.xlabel("Scores' Distribution")
    plt.yticks(rotation=30)
    plt.title("Intelligence Scores Distribution of dataset {}".format(dataset_type))
    plt.subplots_adjust(top=0.95, bottom=0.1)
    for ax in fig.axes:
        style_axes(ax)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    save_figure(fig, os.path.join(res_dir, "joyplot_{}_cogdistrib.png".format(dataset_type)))


def draw_violin_distrib(df_long, dataset_type, labellist, res_dir):
    apply_plot_style()
    plt.figure(figsize=(10, 6))
    rb_cmap = get_red_blue_cmap()
    ax = sns.violinplot(
        x="intell_category",
        y="scores",
        hue="gender",
        data=df_long,
        order=labellist,
        scale="count",
        split=True,
        palette=[rb_cmap(0.1), rb_cmap(0.9)],
    )
    ax.set_facecolor("#FFFFFF")
    plt.xlabel("Scores' Distribution")
    plt.xticks(rotation=30)
    plt.title("Intelligence Scores Sex Distribution of dataset {}".format(dataset_type))
    plt.subplots_adjust(top=0.95, bottom=0.1)
    style_axes(ax)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    out_path = os.path.join(res_dir, "joyplot_sex_{}_cogdistrib.png".format(dataset_type))
    save_figure(ax.figure, out_path)
