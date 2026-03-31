import os

import matplotlib.pyplot as plt
import nibabel
import numpy as np
import seaborn as sns
from nilearn.plotting import (
    find_xyz_cut_coords,
    plot_connectome,
    plot_img_on_surf,
    plot_roi,
    plot_stat_map,
)

from plot_style import (
    apply_plot_style,
    force_colorbar_end_ticks,
    get_red_blue_cmap,
    get_signed_color_kwargs,
    style_axes,
    style_colorbar,
)


def _load_img(img_or_path):
    if isinstance(img_or_path, str):
        return nibabel.load(img_or_path)
    return img_or_path


def _save_display(display, path, dpi=600):
    if hasattr(display, "savefig"):
        display.savefig(path, dpi=dpi, bbox_inches="tight")
        return
    if hasattr(display, "figure") and hasattr(display.figure, "savefig"):
        display.figure.savefig(path, dpi=dpi, bbox_inches="tight")
        return
    if isinstance(display, np.ndarray) and display.size > 0:
        first_item = display.flat[0]
        if hasattr(first_item, "figure") and hasattr(first_item.figure, "savefig"):
            first_item.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            return
    if isinstance(display, (list, tuple)) and display:
        first_item = display[0]
        if hasattr(first_item, "figure") and hasattr(first_item.figure, "savefig"):
            first_item.figure.savefig(path, dpi=dpi, bbox_inches="tight")
            return
        if hasattr(first_item, "savefig"):
            first_item.savefig(path, dpi=dpi, bbox_inches="tight")
            return
    current_fig = plt.gcf()
    if current_fig is not None and hasattr(current_fig, "savefig"):
        current_fig.savefig(path, dpi=dpi, bbox_inches="tight")
        return
    raise TypeError(f"Unsupported display type for saving: {type(display)!r}")


def get_coords(atlas_img, label_ids=None):
    atlas_img = _load_img(atlas_img)
    if label_ids is None:
        values = np.unique(atlas_img.get_fdata())
        values = values[values != 0]
    else:
        values = np.asarray(label_ids).astype(float)
    coords = []
    for v in values:
        data = np.zeros_like(atlas_img.get_fdata())
        data[atlas_img.get_fdata() == v] = 1
        xyz = find_xyz_cut_coords(nibabel.Nifti1Image(data, atlas_img.affine))
        coords.append(xyz)
    return np.asarray(coords)


def compute_roi_values(edge_contrib, normalize=True):
    edge_contrib = np.asarray(edge_contrib)
    if edge_contrib.ndim != 2 or edge_contrib.shape[0] != edge_contrib.shape[1]:
        raise ValueError("edge_contrib must be a square 2D array.")

    contrib = edge_contrib.astype(np.float64, copy=True)
    np.fill_diagonal(contrib, 0.0)

    roi_values = contrib.sum(axis=1, keepdims=True)
    if normalize:
        total = roi_values.sum()
        if total != 0:
            roi_values = roi_values / total

    return roi_values


def draw_connectome(
    adjacency_matrix,
    coords=None,
    atlas_img=None,
    edge_threshold="90%",
    node_size=18,
    title="Brain Connectome",
    save_dir=None,
    filename="connectome.pdf",
    edge_vmin=None,
    edge_vmax=None,
    node_color="auto",
):
    adj = np.asarray(adjacency_matrix)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adjacency_matrix must be a square 2D array.")

    if coords is None:
        if atlas_img is None:
            raise ValueError("Provide either coords or atlas_img.")
        coords = get_coords(atlas_img)
    coords = np.asarray(coords)

    if coords.shape[0] != adj.shape[0] or coords.shape[1] != 3:
        raise ValueError("coords shape must be (num_roi, 3) and match adjacency_matrix.")

    apply_plot_style()
    display = plot_connectome(
        adjacency_matrix=adj,
        node_coords=coords,
        edge_threshold=edge_threshold,
        node_size=node_size,
        title=title,
        node_color=node_color,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        _save_display(display, os.path.join(save_dir, filename), dpi=600)

    return display


def draw_atlas_roi(
    roi_values,
    atlas_img,
    label_ids=None,
    bg_img=None,
    view="stat",
    title="Atlas ROI Values",
    cmap=None,
    threshold=None,
    cut_coords=None,
    display_mode="ortho",
    cbar_tick_format="%.3f",
    save_dir=None,
    filename="atlas_roi.pdf",
):
    atlas_img = _load_img(atlas_img)
    atlas_data = atlas_img.get_fdata()

    roi_values = np.asarray(roi_values).squeeze()
    if roi_values.ndim != 1:
        raise ValueError("roi_values must be 1D (num_roi,) or (num_roi, 1).")

    if label_ids is None:
        labels = np.unique(atlas_data)
        labels = labels[labels != 0]
    else:
        labels = np.asarray(label_ids).astype(float)

    if roi_values.size != labels.size:
        raise ValueError(
            f"roi_values length ({roi_values.size}) does not match label count ({labels.size})."
        )

    value_img_data = np.zeros_like(atlas_data, dtype=np.float32)
    for idx, label in enumerate(labels):
        value_img_data[atlas_data == label] = roi_values[idx]

    value_img = nibabel.Nifti1Image(value_img_data, atlas_img.affine, atlas_img.header)

    apply_plot_style()
    if cmap is None:
        cmap = get_red_blue_cmap()
    abs_max = float(np.max(np.abs(roi_values))) if roi_values.size else 0.0
    if abs_max > 0:
        vmin = -abs_max
        vmax = abs_max
    else:
        vmin = None
        vmax = None

    if view == "roi":
        display = plot_roi(
            roi_img=value_img,
            bg_img=bg_img,
            title=title,
            cmap=cmap,
            threshold=threshold,
            cut_coords=cut_coords,
            display_mode=display_mode,
            colorbar=True,
            cbar_tick_format=cbar_tick_format,
            vmin=vmin,
            vmax=vmax,
        )
    elif view == "surf":
        try:
            display = plot_img_on_surf(
                stat_map=value_img,
                title=title,
                cmap=cmap,
                threshold=threshold,
                colorbar=True,
                cbar_tick_format=cbar_tick_format,
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=True,
            )
        except ValueError as exc:
            if "Unknown projection '3d'" not in str(exc):
                raise
            display = plot_stat_map(
                stat_map_img=value_img,
                bg_img=bg_img,
                title=f"{title} (fallback)",
                cmap=cmap,
                threshold=threshold,
                cut_coords=cut_coords,
                display_mode=display_mode,
                colorbar=True,
                cbar_tick_format=cbar_tick_format,
                vmin=vmin,
                vmax=vmax,
                symmetric_cbar=True,
            )
    else:
        display = plot_stat_map(
            stat_map_img=value_img,
            bg_img=bg_img,
            title=title,
            cmap=cmap,
            threshold=threshold,
            cut_coords=cut_coords,
            display_mode=display_mode,
            colorbar=True,
            cbar_tick_format=cbar_tick_format,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=True,
        )

    if hasattr(display, "_cbar") and display._cbar is not None:
        ticks = [vmin, vmax] if vmin is not None and vmax is not None else None
        style_colorbar(display._cbar, ticks=ticks, tick_format=cbar_tick_format)
        force_colorbar_end_ticks(display._cbar, ticks=ticks, tick_format=cbar_tick_format)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, filename)
        _save_display(display, out_path, dpi=600)

    return value_img, display


def draw_heatmap(mat, save_dir=None, show=True):
    apply_plot_style()
    color_kwargs = get_signed_color_kwargs(mat)
    ax = sns.heatmap(mat, cbar=True, **color_kwargs)
    style_axes(
        ax,
        title="Resampled Heatmap of Brain Connectivity",
        xlabel="Subjects",
        ylabel="Significance-chosen Rearranged Connectivity",
    )
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            vmin = color_kwargs.get("vmin")
            vmax = color_kwargs.get("vmax")
            if vmin is not None and vmax is not None:
                ticks = [vmin, vmax]
                style_colorbar(cbar, ticks=ticks, tick_format="%.3f")
            else:
                style_colorbar(cbar)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "heatmap.pdf")
        ax.figure.savefig(save_path, dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(ax.figure)
