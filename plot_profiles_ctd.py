"""
Author: lgarzio on 6/11/2025
Last modified: lgarzio on 6/11/2025
Plot CTD profiles
"""

import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def main(fdir):
    savedir = os.path.join(fdir, 'profiles')
    os.makedirs(savedir, exist_ok=True)

    fmod_savedir = os.path.join(fdir, 'modified_csv')
    os.makedirs(fmod_savedir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(fdir, '*.csv')))
    vars = ['TEMP', 'CNDC', 'PSAL', 'density', 'flSP']
    for f in files:
        df = pd.read_csv(f)

        # remove rows where depth is negative
        df = df[df['DEPTH'] >= 0]
        df = df.reset_index(drop=True)

        # find max depth minus 0.05m, find the median index at that depth and assign down/up profiles
        deep = np.nanmax(df['DEPTH'])
        deep_range = deep - .05

        deep_idxs = np.where(np.logical_and(df['DEPTH'] >= deep_range, df['DEPTH'] <= deep))[0]
        deep_idx = int(np.median(deep_idxs))
        df['profile'] = ''
        df.loc[0:deep_idx+1, 'profile'] = 'down'
        df.loc[deep_idx+1:, 'profile'] = 'up'

        # save a modified version of the original file
        fname = f.split('/')[-1]
        fmod = fname.replace('.csv', '_modified.csv')
        df.to_csv(os.path.join(fmod_savedir, fmod), index=False)

        cast = dict(
            down='tab:blue',
            up='tab:red'
        )
        for v in vars:
            fig, ax = plt.subplots(figsize=(8, 10))
            for c, color in cast.items():
                depth = df.loc[df['profile'] == c, 'DEPTH'].values
                data = df.loc[df['profile'] == c, v].values

                ax.plot(data, depth, color=color)  # plot lines
                ax.scatter(data, depth, color=color, s=10, label=c)
            
            ax.invert_yaxis()
            ax.set_ylabel('Depth (m)')
            ax.set_xlabel(f'{v}')

            if v == 'density':  # round ticks for density
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            plt.legend(loc='best')
            #ax.ticklabel_format(useOffset=False)  # don't use scientific notation for ticks

            ttl = f'{np.unique(df.station)[0]}:  {np.unique(df.date)[0]}'
            ax.set_title(ttl)

            save_filename = fname.replace('.csv', f'_{v}.png')
            sfile = os.path.join(savedir, save_filename)
            plt.savefig(sfile, dpi=300)
            plt.close()


if __name__ == '__main__':
    filedir = '/Users/garzio/Documents/rucool/Saba/CostaRica/costa_rica_CTD_data/csv'
    main(filedir)
