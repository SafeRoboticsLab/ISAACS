
"""Code for visualising the save collections."""

import pdb

import os
import pickle
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def process_args():
    """Read and interpret command line arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    # parser = argparse.ArgumentParser(
    #   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadpath', required=True)
    parser.add_argument('--savepath', default=None)
    return parser.parse_args()


def main_visualisation(args, dpi=200):
    """Main function to load and visualise the collection and store to file."""
    filepath = os.path.join(args.loadpath, 'trained_collection.pkl')
    filename = filepath.split('/')[-2]

    # TODO: Change colormap and make it midpoint
    cmap = cm.CMRmap_r

    with open(filepath, "rb") as f:
        dimensions, grid_collection = pickle.load(f)
    # Extract all fitness values
    if len(dimensions) == 2:
        grid_plot = -np.inf * np.ones(dimensions)
        for (i, j), v in grid_collection.items():
            grid_plot[i, j] = v.fitness
    elif len(dimensions) == 6:
        grid_plot = -np.inf * np.ones(
            (np.prod(dimensions[0::2]), np.prod(dimensions[1::2])))
        factors = np.ones(len(dimensions))
        for x in range(0, 2):
            for i in range(x + 2, len(dimensions), 2):
                factors[i] = np.prod(dimensions[x:i:2])
        for bd, v in grid_collection.items():
            A = int(np.sum(np.multiply(bd, factors)[0::2]))
            B = int(np.sum(np.multiply(bd, factors)[1::2]))
            grid_plot[A, B] = v.fitness
    else:
        raise AttributeError(
            "Collection size not supported! (only 2 and 6 dimensional grids)")

    # Plot and save collection
    ratio = grid_plot.shape[0] / grid_plot.shape[1]
    fig = plt.figure(figsize=(5, int(5 * ratio)))
    ax = fig.add_subplot(111)
    # im = ax.imshow(
    #     grid_plot, interpolation='none', origin='lower', cmap=cmap, vmin = -0.2, vmax = 0.31)
    im = ax.imshow(
        grid_plot, interpolation='none', origin='lower', cmap=cmap)
    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("right", size="5%", pad=0.25)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Fitness')
    ax.set_xticks(np.arange(-0.5, grid_plot.shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_plot.shape[1], 1), minor=True)
    ax.grid(which='minor', color='gray', ls='-', lw=0.5, alpha=0.5)
    ax.set_xlim(-0.5, grid_plot.shape[0] + 0.5)
    ax.set_ylim(-0.5, grid_plot.shape[1] + 0.5)
    ax.set_xlabel('dim 0')
    ax.set_ylabel('dim 1')

    # Save file
    ax.set_title('File: {}\nBehaviour coverage: {}/{} ({:.2f}%)'.format(
        filename, len(grid_collection), np.prod(dimensions),
        len(grid_collection) / np.prod(dimensions) * 100))
    savepath = '/'.join(filepath.split('/')[:-1]) \
        if args.savepath is None else args.savepath
    if savepath is not None:
        plt.savefig(
            os.path.join(savepath, '{}__collection.png'.format(filename)),
            format="png",
            dpi=dpi)
    else:
        plt.show()


if __name__ == "__main__":
    args = process_args()
    main_visualisation(args)
