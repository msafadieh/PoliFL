#!/usr/bin/python3

import os
# import sys
import time
import re
import random

import pandas as pd

from glob import glob
from tqdm import tqdm

from datetime import datetime, timedelta

SLEEP_TIME = 5


def produce_stats(user_files, words, output='anew_stats.csv'):

    print('Producing stats...')

    # create df
    columns = ['Filename', 'Words Count', 'Filtered Words Count']
    df = pd.DataFrame(index=range(len(user_files)), columns=columns)
    df.index.name = 'Index'

    # per user
    for idx, file in enumerate(tqdm(user_files, ncols=80)):

        # counter
        filtered_word_count = 0

        # read file
        with open(file, 'r') as myfile:
            datastring = myfile.read()

            # per word
            for word in words:

                # count instances of 'word'
                filtered_word_count += datastring.count(word)

        # cound words
        words_len = len(re.findall(r'\w+', datastring))

        # add to df
        df.loc[idx] = (os.path.basename(file), words_len, filtered_word_count)

    df.to_csv(output)


def evaluate(user_files, words, output='anew_results.csv'):

    print('Evaluating...')

    # create df
    columns = ['Filename', 'Start Time', 'End Time', 'Duration']
    df = pd.DataFrame(index=range(len(user_files)), columns=columns)
    df.index.name = 'Index'

    # per user
    for idx, file in enumerate(tqdm(user_files, ncols=80)):

        # read file
        with open(file, 'r') as myfile:
            datastring = myfile.read()

            # << Evaluation Start >>
            time.sleep(SLEEP_TIME)
            start_time = time.time()

            # per word
            for word in words:

                # remove instances of 'word'
                datastring = datastring.replace(word, "")

            end_time = time.time()
            time.sleep(SLEEP_TIME)
            # << Evaluation End >>

            delta = float(end_time - start_time)

            # add to df
            df.loc[idx] = (os.path.basename(file), start_time, end_time, delta)

    df.to_csv(output)


##################################################################
# MAIN
##################################################################

if __name__ == '__main__':

    print("Process ID: %d" % os.getpid())

    print("Starting in 10 seconds...\n")
    time.sleep(10)

    print("Started at: %s" % (datetime.now()))
    start_time = time.time()

    anew_path = 'ANEW2010All.txt'
    reddit_folder = 'shard_by_author'

    column = 'ValMn'
    threshold = 5.0

    # read ANEW
    anew_df = pd.read_csv(anew_path, sep='\t')
    # keep valid entries
    anew_df = anew_df[anew_df[column] > threshold]
    words = anew_df['Word']

    user_files = glob(reddit_folder + '/*')
    user_files = random.sample(user_files, 1000)

    # per user from reddit_folder
    produce_stats(user_files, words)
    evaluate(user_files, words)

    # save and report elapsed time
    elapsed_time = time.time() - start_time
    print("\nSuccess! Duration: %s" % str(timedelta(seconds=int(elapsed_time))))
