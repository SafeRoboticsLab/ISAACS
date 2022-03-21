
"""Code for visualising the save collections"""

import pdb

import os
import pickle
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def process_args():
    """Read and interpret command line arguments"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    # parser = argparse.ArgumentParser(
    #   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loadpath', required=True)
    parser.add_argument(
        '--plot_type', nargs='+',
        default=['behaviours', 'maxfitness', 'qdscore', 'avgfitness'],
        help="Select plot type(s):\n"
             "'behaviours'\n"
             "'maxfitness'\n"
             "'avgfitness'\n"
             "'qdscore'\n")
    return parser.parse_args()


def main_visualisation(args, dpi=200):
    """Main function to load and visualise the collection and store to file."""
    filepath = os.path.join(args.loadpath, 'training_data.pkl')
    filename = filepath.split('/')[-2]

    with open(filepath, "rb") as f:
        training_data = pickle.load(f)

    # Extract all values
    total_samples = 0
    total_behaviours = [0]
    total_max_fitness = []
    total_avg_fitness = []
    total_qd_score = []
    for iter_ in training_data:
        total_samples += iter_['n_sampled']
        total_behaviours.extend([iter_['collection_size']] * iter_['n_sampled'])
        total_max_fitness.extend([iter_['max_fitness']] * iter_['n_sampled'])
        total_avg_fitness.extend([iter_['avg_fitness']] * iter_['n_sampled'])
        total_qd_score.extend([iter_['qd_score']] * iter_['n_sampled'])

    if 'behaviours' in args.plot_type:
        plt.plot(np.arange(total_samples + 1), total_behaviours)
        # Save file
        plt.title('File: {}'.format(filename))
        savepath = '/'.join(filepath.split('/')[:-1])
        plt.savefig(
            os.path.join(savepath, '{}__{}.png'.format(filename, 'bd_plot')),
            format="png",
            dpi=dpi)
        plt.cla()
        
    if 'maxfitness' in args.plot_type:
        plt.plot(np.arange(total_samples), total_max_fitness)
        # Save file
        plt.title('File: {}'.format(filename))
        savepath = '/'.join(filepath.split('/')[:-1])
        plt.savefig(
            os.path.join(
                savepath, '{}__{}.png'.format(filename, 'maxfit_plot')),
            format="png",
            dpi=dpi)
        plt.cla()
        
    if 'avgfitness' in args.plot_type:
        plt.plot(np.arange(total_samples), total_avg_fitness)
        # Save file
        plt.title('File: {}'.format(filename))
        savepath = '/'.join(filepath.split('/')[:-1])
        plt.savefig(
            os.path.join(
                savepath, '{}__{}.png'.format(filename, 'avgfit_plot')),
            format="png",
            dpi=dpi)
        plt.cla()

    if 'qdscore' in args.plot_type:
        plt.plot(np.arange(total_samples), total_avg_fitness)
        # Save file
        plt.title('File: {}'.format(filename))
        savepath = '/'.join(filepath.split('/')[:-1])
        plt.savefig(
            os.path.join(
                savepath, '{}__{}.png'.format(filename, 'qdscore_plot')),
            format="png",
            dpi=dpi)
        plt.cla()


if __name__ == "__main__":
    args = process_args()
    main_visualisation(args)
