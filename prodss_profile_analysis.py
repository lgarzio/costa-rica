"""
Author: lgarzio on 6/13/2025
Last modified: lgarzio on 6/13/2025
Figure out if up/down casts for pH and DO can be averaged.
Bin data into 0.25m depth bins
For pH, if up and down casts are within 0.05 pH units, data can be averaged.
For DO, the accuracy is +/-8% of reading, if values are within that range (8% of the lower reading), data can be averaged.
"""

import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def bin_data(dataframe, depthvar='DEP m', binsize=0.25):
    """
    Bin the data (calculate average) into specified depth bins
    :param dataframe: pandas DataFrame containing the data to be binned
    :param depthvar: the name of the depth variable in the dataframe, default is 'DEP m'
    :param binsize: the size of the depth bins, default is 0.25
    :return: binned DataFrame
    """
    depth_max = np.nanmax(dataframe[depthvar])
    bins = np.arange(0, depth_max + binsize, binsize)  # Generate array of depths you want to bin at
    cut = pd.cut(dataframe[depthvar], bins)
    dataframe_binned = dataframe.groupby(cut, observed=False).mean()
    dataframe_binned = dataframe_binned.dropna(subset=dataframe.columns, how='all')
    dataframe_binned.rename(columns=lambda col: f"{col}_avg", inplace=True)
    return dataframe_binned


def main(fdir):
    pHyes = 0
    pHno = 0
    pHnan = 0
    DOyes = 0
    DOno = 0
    DOnan = 0
    files = sorted(glob.glob(os.path.join(fdir, '*_modified.csv')))
    vars = ['DO mg/L', 'pH']
    depthvar = 'DEP m'  # depth variable
    vars.append(depthvar)
    for f in files:
        df = pd.read_csv(f)

        # bin the data into 0.25m depth bins for each down/up cast and merge the dataframes
        down = df.loc[df['profile'] == 'down'].copy()
        up = df.loc[df['profile'] == 'up'].copy()

        down = down[vars].dropna(how='all')
        up = up[vars].dropna(how='all')

        down_binned = bin_data(down)
        up_binned = bin_data(up)
        
        # merge the down and up binned dataframes
        df_binned = pd.merge(down_binned, up_binned, left_index=True, right_index=True, suffixes=('_down', '_up'), how='outer')
        df_binned.sort_index(inplace=True)
        df_binned.reset_index(inplace=True)


        # calculate differences between up and down casts
        df_binned['DO_diff'] = np.abs(df_binned['DO mg/L_avg_down'] - df_binned['DO mg/L_avg_up'])
        df_binned['pH_diff'] = np.abs(df_binned['pH_avg_down'] - df_binned['pH_avg_up'])

        # determine if the up and down casts are within the specified thresholds
        df_binned['DO_threshold'] = 0.08 * df_binned[['DO mg/L_avg_down', 'DO mg/L_avg_up']].min(axis=1)
        df_binned['pH_threshold'] = 0.05  # 0.05 pH units for pH

        df_binned['can_avg_DO'] = ''
        idx = np.where(np.logical_and(df_binned['DO_diff'] <= df_binned['DO_threshold'], ~df_binned['DO_diff'].isna()))[0]
        df_binned.loc[idx, 'can_avg_DO'] = 'yes'
        DOyes += len(idx)
        idx = np.where(np.logical_and(df_binned['DO_diff'] > df_binned['DO_threshold'], ~df_binned['DO_diff'].isna()))[0]
        df_binned.loc[idx, 'can_avg_DO'] = 'no'
        DOno += len(idx)
        DOnan += df_binned['DO_diff'].isna().sum()
        df_binned['can_avg_pH'] = ''
        idx = np.where(np.logical_and(df_binned['pH_diff'] <= df_binned['pH_threshold'], ~df_binned['pH_diff'].isna()))[0]
        df_binned.loc[idx, 'can_avg_pH'] = 'yes'
        pHyes += len(idx)
        idx = np.where(np.logical_and(df_binned['pH_diff'] > df_binned['pH_threshold'], ~df_binned['pH_diff'].isna()))[0]
        df_binned.loc[idx, 'can_avg_pH'] = 'no'
        pHno += len(idx)
        pHnan += df_binned['pH_diff'].isna().sum()
        
        df_binned.rename(columns={'DEP m': 'depth_bin'}, inplace=True)

        # save a modified version of the original file
        fname = f.split('/')[-1]
        fbinned = fname.replace('.csv', '_binned.csv')
        df_binned.to_csv(os.path.join(fdir, fbinned), index=False)
    
    print(f"Total depth bins analyzed: {pHyes + pHno + pHnan}")
    print(f"pH can be averaged yes: {pHyes}, no: {pHno}, nan: {pHnan}")
    print(f"DO can be averaged yes: {DOyes}, no: {DOno}, nan: {DOnan}")


if __name__ == '__main__':
    filedir = '/Users/garzio/Documents/rucool/Saba/CostaRica/ProDSS/modified_csv'
    main(filedir)
