import os
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import matplotlib as mpl

plt.switch_backend('agg')
plt.style.use('seaborn-paper')

mpl.font_manager.fontManager.addfont(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Helvetica-Regular.ttf'))

plt.rcParams['font.family'] = 'Helvetica Neue LT'
plt.rcParams['legend.fontsize'] = 12.0
plt.rcParams['axes.labelsize'] = 14.0
plt.rcParams['xtick.labelsize'] = 14.0
plt.rcParams['ytick.labelsize'] = 14.0
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['savefig.dpi'] = 200.0

color_pool = ['blue', 'red', 'darkorange', 'purple', 'black', 'cyan', 'lime', 'gold']

def main():
    df = pd.read_csv('../data/all_target_acc.csv', sep='\t')
    
    fig, ax = plt.subplots()  #figsize=(4, 4))
    x, y = df['proteinmpnn_blosum'], df['mrf_blosum']
    ax.scatter(x, y, color='blue')
    ax.set_xlabel('ProteinMPNN')
    ax.set_ylabel('NeuralMRF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.grid(linestyle='--')
    
    x = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 100)
    y = x
    ax.plot(x, y, linestyle='--', color='black')
    ax.set_title('BLOSUM score')
    plt.savefig('./figures/blosum_proteinmpnn_vs_prodesign.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

    # plot
    fig, ax = plt.subplots()  #figsize=(4, 4))
    x, y = df['proteinmpnn'], df['mrf']
    ax.scatter(x, y, color='blue')
    ax.set_xlabel('ProteinMPNN')
    ax.set_ylabel('NeuralMRF')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal')
    ax.grid(linestyle='--')
    
    x = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 100)
    y = x
    ax.plot(x, y, linestyle='--', color='black')
    ax.set_title('Sequence Recovery Rate')

    plt.savefig('./figures/acc_proteinmpnn_vs_prodesign.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

    # plot bar for acc
    fig, ax = plt.subplots(figsize=(3, 4))
    names = ['NeuralMRF', 'ProteinMPNN']
    y_pos = [0.2, 0.5]
    height = [df['mrf'].mean(), df['proteinmpnn'].mean()]
    
    ax.bar(y_pos, height, color='purple', width=0.08)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(names)

    ax.set_ylabel('Sequence Recovery Rate')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--')
    plt.savefig('./figures/acc_bar.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)
    
    # plot bar for BLOSUM
    fig, ax = plt.subplots(figsize=(3,4))  #figsize=(4, 4))
    names = ['NeuralMRF', 'ProteinMPNN']
    y_pos = [0.2, 0.5]
    height = [df['mrf_blosum'].mean(), df['proteinmpnn_blosum'].mean()]
    
    ax.bar(y_pos, height, color='purple', width=0.08)
    ax.set_xticks(y_pos)
    ax.set_xticklabels(names)

    ax.set_ylabel('BLOSUM score')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(linestyle='--')
    plt.savefig('./figures/blosum_bar.pdf',
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

if __name__ == '__main__':
    main()
