#!/usr/bin/python3

import sys
import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt

model = 'big'
tickfontsize = 16
labelfontsize = 18
linewidth = 3

def produce_cdf_plot(df, key, xlabel, color, filename):

    ax = df[key].hist(cumulative=True, density=True, bins=100, histtype='step',
        xlabelsize=tickfontsize, ylabelsize=tickfontsize,
        color=color, linewidth=linewidth)

    plt.xlabel(xlabel, weight='bold', fontsize=labelfontsize)
    plt.ylabel("CDF (0-1)", weight='bold', fontsize=labelfontsize)

    fig = ax.get_figure()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def produce_hist_plot(df, key, xlabel, color, filename):

    ax = df[key].hist(cumulative=False, density=True, bins=100, histtype='step',
        xlabelsize=tickfontsize, ylabelsize=tickfontsize,
        color=color, linewidth=linewidth)

    plt.xlabel(xlabel, weight='bold', fontsize=labelfontsize)
    plt.ylabel("Density (0-1)", weight='bold', fontsize=labelfontsize)

    fig = ax.get_figure()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def merge_data(edf, tdf):

    # create df
    columns = ['Duration', 'CPU', 'Memory']
    df = pd.DataFrame(index=tdf.index, columns=columns)
    df.index.name = 'Participant'

    # add Duration and Data Size
    df['Duration'] = tdf['Duration']
    df['Data Size'] = tdf['Data Size']

    # Add CPU and Memory
    for index, row in df.iterrows():

        start = tdf.loc[index]['Start Time']
        end = tdf.loc[index]['End Time']

        # Add CPU
        df.loc[index, 'CPU'] = edf[((edf.index > start) & (edf.index < end))]['%CPU'].mean()

        # # Add VIRT
        # virt = edf[((edf.index > start) & (edf.index < end))]['VIRT'].tail(1).values[0]
        # if virt[len(virt)-1] is not 'g':
        #     print("Warning: '%s'" % virt[len(virt)-1])

        # df.loc[index, 'VIRT'] = float(virt[:-1])

        # Add Memory
        mem_start = start - datetime.timedelta(0, 2)  # -2 seconds
        mem_end = end
        df_selected = edf[((edf.index > mem_start) & (edf.index < mem_end))]['%MEM']
        df.loc[index, 'Memory'] = df_selected.iloc[-1] - df_selected.iloc[0]

    return df


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':

    global_path = "New Evals/%s" % model

    # get paths
    evaluation_file_path = os.path.join(global_path, 'evals.csv')
    timestamps_file_path = os.path.join(global_path, 'federated.csv')

    # load evaluation_df
    evaluation_df = pd.read_csv(evaluation_file_path, index_col=0)
    evaluation_df.index = pd.to_datetime(evaluation_df.index) - pd.Timedelta(days=1)
    evaluation_df['%CPU'] = evaluation_df['%CPU'] / 4

    # load timestamps_df
    timestamps_df = pd.read_csv(timestamps_file_path, index_col=0)
    # in case we have duplicate index (users)
    timestamps_df = timestamps_df.loc[~timestamps_df.index.duplicated(keep='first')]
    timestamps_df['Start Time'] = pd.to_datetime(timestamps_df['Start Time'], unit='s')
    timestamps_df['End Time'] = pd.to_datetime(timestamps_df['End Time'], unit='s')

    if model == 'big':
        # super dirty fix
        evaluation_df.index -= pd.Timedelta(days=9)

        evaluation_df['tmp'] = evaluation_df.index
        evaluation_df.tmp.iloc[0:16666] = evaluation_df.iloc[0:16666].index - pd.Timedelta(days=1)
        evaluation_df.index = evaluation_df['tmp']
        evaluation_df.index.name = 'Timestamp'

    elif model == 'small':
        evaluation_df.index -= pd.Timedelta(days=8)

    else:
        print('Wrong model')
        sys.exit()

    # aggregate data into a single df
    df = merge_data(evaluation_df, timestamps_df)

    # df['VIRT'] = tdf['VIRT'] / 1024.0 / 1024.0

    # negative to 0
    df[df['Memory'] < 0] = 0

    print("Report stats for the '%s' model" % model)
    tte_mean = df.Duration.mean()
    tte_std = df.Duration.std()
    cpu_mean = df.CPU.mean()
    cpu_std = df.CPU.std()
    mem_mean = df.Memory.mean()
    mem_std = df.Memory.std()
    print("TTE: %.2f (%.2f)\tCPU: %.2f (%.2f)\tMemory: %.2f (%.2f)" %
      (tte_mean, tte_std, cpu_mean, cpu_std, mem_mean, mem_std))

    # produce cdf plots
    produce_cdf_plot(df,
                     key='Duration',
                     xlabel='Time to Execute (sec)',
                     color='tab:red',
                     filename='cdf_duration_model_%s.pdf' % model)

    produce_cdf_plot(df,
                     key='Data Size',
                     xlabel='Data size',
                     color='tab:orange',
                     filename='cdf_data_model_%s.pdf' % model)

    produce_cdf_plot(df,
                     key='CPU',
                     xlabel='CPU %',
                     color='tab:green',
                     filename='cdf_cpu_model_%s.pdf' % model)

    produce_cdf_plot(df,
                     key='Memory',
                     xlabel='Memory %',
                     color='tab:blue',
                     filename='cdf_memory_model_%s.pdf' % model)

    # produce hist plots
    produce_hist_plot(df,
                      key='Duration',
                      xlabel='Time to Execute (sec)',
                      color='tab:red',
                      filename='hist_duration_model_%s.pdf' % model)

    produce_hist_plot(df,
                      key='Data Size',
                      xlabel='Data size',
                      color='tab:orange',
                      filename='hist_data_model_%s.pdf' % model)

    produce_hist_plot(df,
                      key='CPU',
                      xlabel='CPU %',
                      color='tab:green',
                      filename='hist_cpu_model_%s.pdf' % model)

    produce_hist_plot(df,
                      key='Memory',
                      xlabel='Memory %',
                      color='tab:blue',
                      filename='hist_memory_model_%s.pdf' % model)
