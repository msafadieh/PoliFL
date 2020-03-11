#!/usr/bin/python3

import os
# import sys
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

tickfontsize = 14
labelfontsize = 16


def produce_cdf_plot(df, key, xlabel, color, filename, linewidth=1, in_k=False):

    ax = df[key].hist(cumulative=True, density=True, bins=100, histtype='step',
        xlabelsize=tickfontsize, ylabelsize=tickfontsize,
        color=color, linewidth=linewidth)

    if in_k:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x/1000) + 'k'))

    plt.xlabel(xlabel, weight='bold', fontsize=labelfontsize)
    plt.ylabel("CDF (0-1)", weight='bold', fontsize=labelfontsize)

    fig = ax.get_figure()
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()


def produce_hist_plot(df, key, xlabel, color, filename, linewidth=1, in_k=False):

    ax = df[key].hist(cumulative=False, density=True, bins=100, histtype='step',
        xlabelsize=tickfontsize, ylabelsize=tickfontsize,
        color=color, linewidth=linewidth)

    if in_k:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f}'.format(x/1000) + 'k'))

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
    df['Words Count'] = tdf['Words Count']
    df['Filtered Words Count'] = tdf['Filtered Words Count']

    # Add CPU and Memory
    for index, row in df.iterrows():

        start = tdf.loc[index]['Start Time']
        end = tdf.loc[index]['End Time']

        # Add CPU
        df.loc[index, 'CPU'] = edf[((edf.index > start) & (edf.index < end))]['%CPU'].mean()

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

    global_path = "New Evals/words2"

    # Stats
    anew_stats = os.path.join(global_path, 'anew_stats.csv')
    dfs = pd.read_csv(anew_stats, index_col=0)

    # Results
    anew_results = os.path.join(global_path, 'anew_results.csv')
    dfr = pd.read_csv(anew_results, index_col=0)
    # in case we have duplicate index (users)
    dfr = dfr.loc[~dfr.index.duplicated(keep='first')]
    dfr['Start Time'] = pd.to_datetime(dfr['Start Time'], unit='s')
    dfr['End Time'] = pd.to_datetime(dfr['End Time'], unit='s')

    # merge dfs and dfr
    df = pd.concat([dfs, dfr], axis=1, sort=False)

    # load evaluation_df
    evaluation_file_path = os.path.join(global_path, 'evals.csv')
    evaluation_df = pd.read_csv(evaluation_file_path, index_col=0)
    evaluation_df.index = pd.to_datetime(evaluation_df.index)
    evaluation_df['%CPU'] = evaluation_df['%CPU'] / 4

    # aggregate data into a single df
    df = merge_data(evaluation_df, df)

    # negative to
    df[df['Memory'] < 0] = 0

    # filter outliers
    # df = df[df['Words Count'] < 250000]

    tte_mean = df.Duration.mean()
    tte_std = df.Duration.std()
    cpu_mean = df.CPU.mean()
    cpu_std = df.CPU.std()
    mem_mean = df.Memory.mean()
    mem_std = df.Memory.std()
    print("TTE: %.2f (%.2f)\tCPU: %.2f (%.2f)\tMemory: %.2f (%.2f)" %
      (tte_mean, tte_std, cpu_mean, cpu_std, mem_mean, mem_std))

    # Stats
    produce_cdf_plot(df,
                     key='Words Count',
                     xlabel='Total Words per User',
                     color='tab:orange',
                     filename='cdf_filtering_task_words_count.pdf',
                     linewidth=3,
                     in_k=True)

    produce_hist_plot(df,
                      key='Words Count',
                      xlabel='Total Words per User',
                      color='tab:orange',
                      filename='hist_filtering_task_words_count.pdf',
                      linewidth=2,
                      in_k=True)

    produce_cdf_plot(df,
                     key='Filtered Words Count',
                     xlabel='Filtered Words per User',
                     color='tab:orange',
                     filename='cdf_filtering_task_filtered_words.pdf',
                     linewidth=3,
                     in_k=True)

    produce_hist_plot(df,
                      key='Filtered Words Count',
                      xlabel='Filtered Words per User',
                      color='tab:orange',
                      filename='hist_filtering_task_filtered_words.pdf',
                      linewidth=2,
                      in_k=True)

    # produce cdf plots
    produce_cdf_plot(df,
                     key='Duration',
                     xlabel='Time to Execute (sec)',
                     color='tab:red',
                     linewidth=3,
                     filename='cdf_duration_filtering_task.pdf')

    produce_cdf_plot(df,
                     key='CPU',
                     xlabel='CPU %',
                     color='tab:green',
                     linewidth=3,
                     filename='cdf_cpu_filtering_task.pdf')

    produce_cdf_plot(df,
                     key='Memory',
                     xlabel='Memory %',
                     color='tab:blue',
                     linewidth=3,
                     filename='cdf_memory_filtering_task.pdf')

    # produce hist plots
    produce_hist_plot(df,
                     key='Duration',
                     xlabel='Time to Execute (sec)',
                     color='tab:red',
                     linewidth=3,
                     filename='hist_duration_filtering_task.pdf')

    produce_hist_plot(df,
                      key='CPU',
                      xlabel='CPU %',
                      color='tab:green',
                      linewidth=3,
                      filename='hist_cpu_filtering_task.pdf')

    produce_hist_plot(df,
                      key='Memory',
                      xlabel='Memory %',
                      color='tab:blue',
                      linewidth=3,
                      filename='hist_memory_filtering_task.pdf')
