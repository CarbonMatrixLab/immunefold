import os
from collections import defaultdict
import argparse

import numpy as np

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

def parse_log(path, fields):
    data = defaultdict(list)
    def _parse_line(line):
        if 'Loss/train' not in line:
            return
        flags = line.split(':')
        kv =  line.strip().split('Loss/')[1]
        k, v = kv.split(':')
        v = float(v.strip())

        if k in fields:
            data[k].append(v)
            
    with open(path) as f:
        for line in f:
            _parse_line(line)

    return data

def get_data(log_dir, fields, rank=None):
    data = []
    for f in os.listdir(log_dir):
        if not (f.endswith('.log') and f.startswith('rank')):
            continue

        f_data = parse_log(os.path.join(log_dir, f), fields)
        data.append(f_data)
    
    data = dict(zip(fields, [[b[k] for b in data] for k in fields]))
    for k, v in data.items():
        size = min(map(len, v))
        data[k] = np.mean(np.array([vv[:size] for vv in v]), axis=0)
        
    return data

def plot_curve(fields, data, figure_path, agg_factor=1):
    #y = data[data < 10.0]
    fig, ax = plt.subplots()  #figsize=(4, 4))
    
    for idx, name in enumerate(fields):
        y = data[name]
        if y.shape[0] < agg_factor:
            continue
        y[y > 50.0] = 50. 
        print('steps=', y.shape[0])
        if agg_factor > 1:
            num_segs = (y.shape[0] - 1) // agg_factor + 1
            z = []
            for n in range(num_segs):
                z.append(np.mean(y[n * agg_factor : (n+1) * agg_factor]))
            z = np.array(z)
        else:
            z = y

        print(name)
        for i, yy in enumerate(z):
            print(name, i, yy)
        x = np.arange(z.shape[0]) * agg_factor
        
        name = name.replace('@', '_')
        plt.plot(x, z, color=color_pool[idx], label=name)

    ax.legend(frameon=False, loc='upper right')
    
    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(linestyle='--')

    plt.savefig(figure_path,
                format='pdf',
                bbox_inches='tight',
                pad_inches=0.01)
    plt.close(fig)

def main(args):
    
    fields = ['train@all', 
            'train@mrf@loss',
            'train@mrf@pll_loss', 
            'train@mrf@pair_norm_loss',
            'train@mrf@site_norm_loss', 
            'train@pair@loss',
            'train@seq@loss']
    fields = [
              "train@all",
              "train@folding@loss",
              "train@distogram@loss",
              "train@folding@backbone_fape_loss",
              "train@folding@sidechain_fape_loss",
              "train@folding@chi_loss",
              "train@folding@sq_chi_loss",
              "train@folding@angle_norm_loss",
              "train@predicted_lddt@loss",
              "train@predicted_lddt@lddt_ca_loss",
              "train@sde@loss",
              "train@sde@rot_score_loss",
              "train@sde@trans_score_loss",
              "train@sde@rot_axis_loss",
              "train@sde@rot_angle_loss",
            ]
    data = get_data(args.log_dir, fields)
    
    agg_factor = 2

    f = [fields[k] for k in [0]]
    plot_curve(f, data,
            os.path.join(args.output_dir, 'all_loss.pdf'), 
            agg_factor=agg_factor)
    
    
    f = [fields[k] for k in range(3, 5)]
    plot_curve(f, data,
            os.path.join(args.output_dir, 'folding_loss.pdf'), 
            agg_factor=agg_factor)

    f = [fields[k] for k in range(10, 15)]
    plot_curve(f, data,
            os.path.join(args.output_dir, 'sde_loss.pdf'), 
            agg_factor=agg_factor)

    f = [fields[k] for k in range(9, 10)]
    plot_curve(f, data,
            os.path.join(args.output_dir, 'plddt_loss.pdf'), 
            agg_factor=agg_factor)

    f = [fields[k] for k in range(2,3)]
    plot_curve(f, data,
            os.path.join(args.output_dir, 'distogram_loss.pdf'), 
            agg_factor=agg_factor)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    main(args)
