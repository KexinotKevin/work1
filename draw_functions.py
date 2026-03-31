import numpy as np
import nibabel
import os
import seaborn as sns
import matplotlib.pyplot as plt
from nilearn.plotting import find_xyz_cut_coords, plot_connectome
from plot_style import apply_plot_style, get_signed_color_kwargs, style_axes, style_colorbar

def get_coords(atlas_img):
    # atlas_img = atlas['maps']
    # atlas_img = nibabel.load(atlas_img)
    # all ROIs except the background
    values = np.unique(atlas_img.get_fdata())[1:]
    # iterate over Harvard-Oxford ROIs
    coords = []
    for v in values:
        data = np.zeros_like(atlas_img.get_fdata())
        data[atlas_img.get_fdata() == v] = 1
        xyz = find_xyz_cut_coords(nibabel.Nifti1Image(data, atlas_img.affine))
        coords.append(xyz)
    return coords

def draw_heatmap(mat, save_dir=None,show=True):
    apply_plot_style()
    color_kwargs = get_signed_color_kwargs(mat)
    ax = sns.heatmap(mat, cbar=True, **color_kwargs)
    style_axes(
        ax,
        title='Resampled Heatmap of Brain Connectivity',
        xlabel='Subjects',
        ylabel='Significance-chosen Rearranged Connectivity',
    )
    if ax.collections:
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            style_colorbar(cbar)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    filename="heatmap.png"
    save_path=os.path.join(save_dir,filename)
    ax.figure.savefig(save_path, dpi=600, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(ax.figure)
