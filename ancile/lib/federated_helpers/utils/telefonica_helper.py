import torch
from collections import defaultdict

# system
from sklearn.utils import shuffle

# external libs
import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm

# scripts from this project

# initialization of libraries and global variables
COMPRESSED = True
#NOISE_STD = 0.05
NUM_EMBEDDING_UNITS = 50
NUM_HIDDEN_UNITS = 500
#NUM_CLASSIF_UNITS = 500
#DROPOUT_PROBABILITY = 0.5
MAX_TRAINING_EPOCHS = 100
MAX_PATIENCE = 5
OPTIMIZER = 'adam'      # ['rmsprop', 'adam', 'adadelta']
LEARNING_RATE = 1e-3
CLIP_NORM = None

NUM_NON_ZERO_IMPUTATIONS = 1

DATA_FILE = 'data.h5'

from ancile.lib.federated_helpers.utils.helper import Helper
import logging

from ancile.lib.federated_helpers.models.tf_pytorch_classifier import *


# from models.word_model import RNNModel
# from utils.nlp_dataset import NLPDataset
# from utils.text_load import *
from torchvision import datasets, transforms
import numpy as np
logger = logging.getLogger("logger")


class TelefonicaHelper(Helper):
    NUM_EMBEDDING_UNITS = 50

    def create_model(self):
        batch_size_train, seq_len, num_features = (self.batch_size, 50, 60)
        num_features -= 2

        self.target_model = Model(self.name, self.current_time, batch_size_train, seq_len, num_features).to(self.device)
        self.local_model = Model(self.name, self.current_time,  batch_size_train, seq_len, num_features).to(self.device)

        return

    def create_one_model(self):

        return self.create_model()


    def load_data(self):
        # Load dataset
        # DATA_FOLDERS = [('train', f'data/{self.data_path}/train/'),
        #                 ('valid', f'data/{self.data_path}/validation/'),
        #                 ('test', f'data/{self.data_path}/test/'),
        #                 ('test_unk', f'data/{self.data_path}/test_unk/')]
        # model_folder = os.path.dirname('../models/%s/' % time.strftime('%Y%m%d-%H%M%S'))
        # os.makedirs(model_folder)
        #
        # # Load data and loop for every set
        # data = {}
        # self.uids = {}
        # column_names = None
        # for mode, bin_folder in DATA_FOLDERS:
        #     bin_folder = os.path.join(bin_folder, 'compressed')
        #     data_file_path = os.path.join(bin_folder, DATA_FILE)
        #     h5f = h5py.File(data_file_path, 'r', libver='latest')
        #     num_buckets = h5f.attrs['num_buckets']
        #     self.uids[mode] = h5f.attrs['uuids']
        #     if column_names is None:
        #         column_names = h5f.attrs['column_names'].tolist()
        #     '''
        #     meta_data_entries = h5f.attrs.keys()
        #     print("Using %s data file %s, %d meta data entries:" % (mode, data_file_path, len(meta_data_entries)))
        #     for item in meta_data_entries:
        #         print("\t", item + " : ", h5f.attrs[item])
        #     print
        #     #'''
        #
        #     print("loading %d %s buckets FROM %s" % (num_buckets, mode, data_file_path))
        #     bucket_list = []
        #     for bucket_num in tqdm(range(1, num_buckets + 1), ncols=100):
        #         name = 'bucket_%d' % bucket_num
        #         bucket = h5f[name][:]
        #         tqdm.write("\t%s: %s" % (name, bucket.shape))
        #         bucket_list.append(bucket)
        #     h5f.close()
        #     data[mode] = bucket_list[:]
        #
        # self.train_data = data['train']
        # self.val_data = data['valid']
        # # self.test_data = data['test']
        # self.test_data = [data['test'], data['test_unk']]
        # self.column_names = column_names
        self.train_data = np.load('data/train_data.npy')


        return

    def get_iterator(self, train_buckets_copy):
        train_buckets_copy = shuffle(train_buckets_copy)
        train_iterator = tqdm(train_buckets_copy, desc='Bucket', leave=False, ncols=100)
        return train_iterator




    def get_batch(self, train_data, batch, evaluation=False):
        x = torch.from_numpy(batch[:, :, :-2]).to(self.device)
        y = torch.from_numpy(batch[:, :, -2:-1]).to(self.device)
        w = torch.from_numpy(batch[:, :, -1]).to(self.device)

        # Make sure the bucket has at least some non-zero weight
        if True or np.sum(w) <= 0:
            for _ in range(NUM_NON_ZERO_IMPUTATIONS):
                pi, pj = np.random.randint(0, w.shape[0]), np.random.randint(0, w.shape[1])
                if w[pi, pj] <= 0:
                    y[pi, pj], w[pi, pj] = 0, 1



        return x, (y, w)