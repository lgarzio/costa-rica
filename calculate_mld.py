"""
Author: lgarzio on 6/13/2025
Last modified: lgarzio on 6/13/2025
Calculate MLD
For CTD data: upcasts only
For ProDSS data: in progress
"""

import glob
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})


def profile_mld(df, mld_var='density', zvar='pressure', qi_threshold=0.5):
    """
    Written by Sam Coakley and Lori Garzio, Jan 2022
    Calculates the Mixed Layer Depth (MLD) for a single profile as the depth of max Brunt‐Vaisala frequency squared
    (N**2) from Carvalho et al 2016 (https://doi.org/10.1002/2016GL071205). Calculates a Quality Index to determine
    if the water column is well-mixed, hence calculating MLD is not appropriate and the function will return
    MLD = np.nan, from Lorbacher et al, 2006 doi:10.1029/2003JC002157. When "QI > 0.8 a well defined MLD results.
    For QI in the range 0.5– 0.8, increased uncertainty of the profile interpretation becomes evident and with
    QI < 0.5 no mixed layer interpretation is possible." (from Section 3.4)
    :param df: depth profile in the form of a pandas dataframe
    :param mld_var: the name of the variable for which MLD is calculated, default is 'density'
    :param zvar: the name of the depth variable in the dataframe, default is 'pressure'
    :param qi_threshold: quality index threshold for determining well-mixed water, default is 0.5
    :return: the depth of the mixed layer in the units of zvar and the max buoyancy frequency in units of s-2
    """
    df.dropna(subset=[mld_var], inplace=True)
    pN2 = np.sqrt(9.81 / np.nanmean(df[mld_var]) * np.diff(df[mld_var], prepend=np.nan) / np.diff(df[zvar], prepend=np.nan)) ** 2
    if len(df) < 5:  # if there are <5 data bins, don't calculate MLD
        mld = np.nan
        maxN2 = np.nan
        qi = np.nan
    elif np.sum(~np.isnan(pN2)) < 3:  # if there are <3 values calculated for pN2, don't calculate MLD
        mld = np.nan
        maxN2 = np.nan
        qi = np.nan
    # elif np.nanmax(np.diff(df[zvar])) > gap(np.nanmax(df[zvar]) - np.nanmin(df[zvar])):
    #     # if there is a gap in the profile that exceeds the defined threshold, don't calculate MLD
    #     mld = np.nan
    #     maxN2 = np.nan
    #     qi = np.nan
    else:
        pressure_range = [np.nanmin(df[zvar]), np.nanmax(df[zvar])]

        maxN2 = np.nanmax(pN2)
        mld_idx = np.where(pN2 == maxN2)[0][0]

        # if the code finds the first or last data point to be the max pN2, return nan
        if np.logical_or(mld_idx == 0, mld_idx == len(df) - 1):
            mld = np.nan
            maxN2 = np.nan
            qi = np.nan
        else:
            #mld = np.nanmean([df[zvar][mld_idx], df[zvar][mld_idx + 1]])
            mld = np.nanmean([df[zvar].iloc[mld_idx], df[zvar].iloc[mld_idx + 1]])

            if np.logical_or(mld < pressure_range[0] + 1, mld > pressure_range[1] - 1):
                # if MLD is within 1 dbar of the top or bottom of the profile, return nan
                mld = np.nan
                maxN2 = np.nan
                qi = np.nan
            else:
                if qi_threshold:
                    # find MLD  1.5
                    mld15 = mld * 1.5
                    mld15_idx = np.argmin(np.abs(df[zvar] - mld15))

                    # Calculate Quality index (QI) from Lorbacher et al, 2006 doi:10.1029/2003JC002157
                    surface_mld_values = df[mld_var][0:mld_idx]  # values from the surface to MLD
                    surface_mld15_values = df[mld_var][0:mld15_idx]  # values from the surface to MLD * 1.5

                    qi = 1 - (np.std(surface_mld_values - np.nanmean(surface_mld_values)) /
                                np.std(surface_mld15_values - np.nanmean(surface_mld15_values)))

                    if qi < qi_threshold:
                        # if the Quality Index is < the threshold, this indicates well-mixed water so don't return MLD
                        mld = np.nan
                        maxN2 = np.nan

    return mld, maxN2, qi


def main(fdir, sensor):
    savedir = os.path.join(fdir, 'profiles_mld')
    os.makedirs(savedir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(fdir, '*.csv')))
    vars = ['TEMP', 'CNDC', 'PSAL', 'density', 'flSP']
    for f in files:
        fname = f.split('/')[-1]
        df = pd.read_csv(f)

        if sensor == 'CTD':
            # filter for upcasts only
            df = df.loc[df['profile'] == 'up'].copy()
            zvar = 'prSM'
            mld_var = 'density'
            depthvar = 'DEPTH'
        elif sensor == 'ProDSS':
            # figure out how to do this
            print('test')

        # bin the data into 0.1 dbar pressure bins
        df2 = df[[mld_var, zvar, depthvar]].dropna(how='all')
        depth_max = np.nanmax(df2[zvar])
        bins = np.arange(0, depth_max+0.1, 0.1)  # Generate array of depths you want to bin at
        cut = pd.cut(df2[zvar], bins)
        df3 = df2.groupby(cut, observed=False).mean()
        
        # calculate MLD using the binned dataframe
        kwargs = {'zvar': zvar, 'mld_var': mld_var}
        mld, n2, qi = profile_mld(df3, **kwargs)

        # plot the full res data (df) and the binned data (df3, density only) that was used to calculate MLD
        for v in vars:
            fig, ax = plt.subplots(figsize=(8, 10))
            ax.plot(df[v], df[depthvar], color='tab:blue')  # plot lines
            ax.scatter(df[v], df[depthvar], color='tab:blue', s=10, label='upcast')

            if v == mld_var:
                ax.plot(df3[mld_var], df3[depthvar], color='tab:orange')
                ax.scatter(df3[mld_var], df3[depthvar], color='tab:orange', label='binned density (0.1 dbar)')
            
            # plot MLD if it's not NaN
            if not np.isnan(mld):
                ax.axhline(mld, color='k', linestyle='--')
            
            ax.invert_yaxis()
            ax.set_ylabel('Depth (m)')
            ax.set_xlabel(f'{v}')

            if v == 'density':  # round ticks for density
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2f}'))

            plt.legend(loc='best')
            #ax.ticklabel_format(useOffset=False)  # don't use scientific notation for ticks

            ttl = f'{np.unique(df.station)[0]}:  {np.unique(df.date)[0]}\nMLD: {mld:.2f} m, N2: {n2:.4f} s-2, QI: {qi:.2f}'
            ax.set_title(ttl)

            save_filename = fname.replace('.csv', f'_{v}.png')
            sfile = os.path.join(savedir, save_filename)
            plt.savefig(sfile, dpi=300)
            plt.close()


if __name__ == '__main__':
    filedir = '/Users/garzio/Documents/rucool/Saba/CostaRica/costa_rica_CTD_data/csv/modified_csv'
    sensor = 'CTD'  # CTD ProDSS
    main(filedir, sensor)
