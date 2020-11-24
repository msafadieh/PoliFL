##############################################################################################
#
# Deep Learning classification script
#
##############################################################################################

# system
import argparse
import datetime
import os
import sys
import time

# pytorch
import torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on: ", DEVICE)
import torch.nn as nn
from torch.autograd import Variable

# external libs
import numpy as np
# from numba import jit

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

from ancile.lib.federated_helpers.models.simple import SimpleNet

NUM_NON_ZERO_IMPUTATIONS = 10
DATA_FOLDERS = [('train', 'data/data/train/'), ('valid', 'data/data/validation/'), ('test', 'data/data/test/'),
                ('test_unk', 'data/data/test_unk/')]
DATA_FILE = 'data.h5'


def var(tensor):
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)


# Note: It uses a Stateful LSTM. Info about how we did it:
# - https://discuss.pytorch.org/t/stateful-rnn-example/10912
# - https://discuss.pytorch.org/t/solved-training-a-simple-rnn/9055/5
class Model(SimpleNet):
    def __init__(self, name, current_time, batch_size, seq_len, num_features, num_layers=2):
        super(Model, self).__init__(name, current_time)

        # save parameters
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_features = num_features
        self.num_layers = num_layers
        self.hidden_size = NUM_HIDDEN_UNITS

        # Fully Connected Layer
        self.fc1 = nn.Linear(in_features=num_features, out_features=NUM_EMBEDDING_UNITS)

        # LSTMs (batch, seq, feature)
        self.lstm = nn.LSTM(input_size=NUM_EMBEDDING_UNITS, hidden_size=NUM_HIDDEN_UNITS, num_layers=num_layers, batch_first=True)

        # Fully Connected Layer
        self.fc2 = nn.Linear(in_features=NUM_HIDDEN_UNITS, out_features=1)

        # activation functions
        self.sigmoid = nn.Sigmoid()
        self.prelu = nn.PReLU()

        # init hidden
        self.reset_states()

    def forward(self, x):
        # input: torch.Size([18, 50, 58])

        # TODO: Check if TimeDistributed is equivalent to this implementation.

        x = self.fc1(x)  # torch.Size([18, 50, 58])
        x = self.prelu(x)  # torch.Size([18, 50, 58])
        x, hidden = self.lstm(x, self.hidden)  # torch.Size([18, 50, 500]) (batch, seq, feature)
        self.hidden = (hidden[0].detach(), hidden[1].detach())
        x = self.fc2(x)  # torch.Size([18, 50, 1])
        x = self.sigmoid(x)  # torch.Size([18, 50, 1])

        return x

    def reset_states(self):
        self.hidden = (var(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                       var(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    # reset states with custom batch_size (needed when switching between training / validation)
    def reset_states_custom(self, batch_size):
        self.hidden = (var(torch.zeros(self.num_layers, batch_size, self.hidden_size)),
                       var(torch.zeros(self.num_layers, batch_size, self.hidden_size)))

#
# def evaluation(y, yhat, w, epsilon=1e-6):
#     # calculate Binary cross entropy
#     yhat = np.clip(yhat, epsilon, 1 - epsilon)
#     # bce = -np.sum(w * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat))) / np.sum(w)
#     bce = -np.mean(w * (y * np.log(yhat) + (1 - y) * np.log(1 - yhat)))
#     return bce
#
#
# def show_loss_report(training_loss_scores, validation_loss_scores):
#
#     tqdm.write("\tTrain Loss: ")
#     tqdm.write('\t[' + ', '.join(str('%.4f' % (x)) for x in training_loss_scores) + ']')
#
#     tqdm.write("\tValid Loss: ")
#     tqdm.write('\t[' + ', '.join(str('%.4f' % (x)) for x in validation_loss_scores) + ']')
#
#
# def plot_loss(training_loss_scores, validation_loss_scores, plot_file):
#
#     # create plot
#     x = range(0, len(training_loss_scores))
#     plt.xticks(x)
#     plt.plot(x, training_loss_scores, 'b^-', label='Train loss')
#
#     if validation_loss_scores is not None:
#         plt.plot(x, validation_loss_scores, 'g^-', label='Validation loss')
#
#     plt.legend(loc=1)
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#
#     # save into file
#     plt.savefig(plot_file)
#
#     # clear plot
#     plt.clf()
#     plt.close()
#
# #
# # def train(model_folder, train_buckets, validation_buckets, column_names, uids_train, uids_valid, max_epochs=MAX_TRAINING_EPOCHS, max_patience=MAX_PATIENCE):
# #
# #     # create model
# #     batch_size_train, seq_len, num_features = train_buckets[0][0].shape
# #     num_features -= 2
# #
# #     print('Train: Found batch_size=%d, seq_len=%d, num_features=%d' % (batch_size_train, seq_len, num_features))
# #     model = Model(None, None, batch_size_train, seq_len, num_features).to(DEVICE)
# #     batch_size_valid, _, _ = validation_buckets[0][0].shape
# #
# #     optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# #     loss_function = nn.BCELoss(reduction='none')
# #
# #     # stat init
# #     training_loss_scores = []
# #     validation_loss_scores = []
# #     best_loss_so_far = np.inf
# #     best_model_so_far = model.state_dict()
# #     patience = max_patience
# #
# #     # We will use a copy for training, as it will be suffled on each epoch
# #     train_buckets_copy = train_buckets[:]
# #
# #     try:
# #         # epoch iteration
# #         for epoch in range(0, max_epochs):
# #             tqdm.write("Epoch %d" % epoch)
# #             tqdm.write('Training ...')
# #             model.train()
# #
# #             # Shuffle the users at each epoch.
# #             train_buckets_copy = shuffle(train_buckets_copy)
# #
# #             # bucket iteration
# #             epoch_loss = []
# #             tqdm_bucket = tqdm(train_buckets_copy, desc='Bucket', leave=False, ncols=100)
# #             for bucket in tqdm_bucket:
# #
# #                 # clear gradients
# #                 model.zero_grad()
# #                 model.reset_states()
# #
# #                 # batch iteration
# #                 tqdm_batch = tqdm(bucket, desc='>> Loss (-.----)', leave=False, ncols=100)
# #                 for batch in tqdm_batch:
# #
# #                     x = torch.from_numpy(batch[:, :, :-2]).to(DEVICE)
# #                     y = torch.from_numpy(batch[:, :, -2:-1]).to(DEVICE)
# #                     w = torch.from_numpy(batch[:, :, -1]).to(DEVICE)
# #
# #                     # Make sure the bucket has at least some non-zero weight
# #                     if True or np.sum(w) <= 0:
# #                         for _ in range(NUM_NON_ZERO_IMPUTATIONS):
# #                             pi, pj = np.random.randint(0, w.shape[0]), np.random.randint(0, w.shape[1])
# #                             if w[pi, pj] <= 0:
# #                                 y[pi, pj], w[pi, pj] = 0, 1
# #
# #                     yhat = model(x)  # forward
# #                     loss = loss_function(yhat, y)
# #
# #                     # apply weights
# #                     # info: https://discuss.pytorch.org/t/per-class-and-per-sample-weighting/25530/3
# #                     loss = loss.view(-1, NUM_EMBEDDING_UNITS)  # reshape to match w
# #                     loss = loss * w
# #                     # loss = (loss * w / w.sum()).sum()
# #                     loss = loss.mean()
# #
# #                     loss.backward()
# #                     optimizer.step()
# #
# #                     epoch_loss.append(loss.item())
# #
# #                     tqdm_batch.set_description('>> Loss (%.4f)' % np.mean(epoch_loss))
# #                     tqdm_batch.refresh()
# #
# #             training_loss_scores.append(np.mean(epoch_loss))
# #
# #             tqdm.write('Predict on validation ...')
# #             prediction_matrix = predict(model, validation_buckets, column_names, 'Validation loss')
# #             bce = evaluation(prediction_matrix[:, 2], prediction_matrix[:, 3], prediction_matrix[:, 4])
# #             validation_loss_scores.append(bce)
# #
# #             # Report losses
# #             # tqdm.write('Report losses')
# #             show_loss_report(training_loss_scores, validation_loss_scores)
# #
# #             # loss plot
# #             # tqdm.write("Plot BCS loss" )
# #             plot_file = os.path.join(model_folder, 'loss_plot.png')
# #             plot_loss(training_loss_scores, validation_loss_scores, plot_file)
# #
# #             # Early stopping
# #             if validation_loss_scores[-1] < best_loss_so_far:
# #                 best_loss_so_far = validation_loss_scores[-1]
# #                 best_model_so_far = model.state_dict()
# #                 patience = max_patience
# #             else:
# #                 patience -= 1
# #                 if patience == 0:
# #                     tqdm.write("End of patience reached. Breaking.")
# #                     break
# #
# #     except (KeyboardInterrupt, SystemExit):
# #         # if Ctrl-C is used, stop training and continue with validation
# #         tqdm.write("Warning: Training was interrupted using Ctrl-c.")
# #
# #     # Set best model
# #     model.load_state_dict(best_model_so_far)
# #
# #     # save the weights in models folder
# #     tqdm.write('Save the weights in models folder')
# #     model_file = os.path.join(model_folder, 'best_model_weights_so_far.pth')
# #     torch.save(best_model_so_far, model_file)
# #
# #     # Evaluate with Training and Validation data, using best model.
# #     tqdm.write('Final predict on training ...')
# #     prediction_matrix = predict(model, train_buckets, column_names, 'Train loss')
# #     output_array_training = produce_output_array('training', prediction_matrix, uids_train, column_names)
# #
# #     tqdm.write('Final predict on validation ...')
# #     prediction_matrix = predict(model, validation_buckets, column_names, 'Validation loss')
# #     output_array_validation = produce_output_array('validation', prediction_matrix, uids_valid, column_names)
# #
# #     return model, output_array_training, output_array_validation

#
# @jit(nopython=True)
# def np_find_first_one(vec):
#     '''return the index of the first occurence of 1 in vec.
#        There is a numpy but checks for all occurences, so this should be faster.
#     '''
#     for i in range(len(vec)):
#         if vec[i] == 1:
#             return i
#     return -1


# def produce_matrix(x, user_start, app_category_start, app_category_size):
#     '''return a matrix with the length of x,
#     having user_index and app_category_index as columns.'''
#
#     # 1. create list of user_ids (user_id is the user_id index only)
#     user_ids = np.repeat(range(user_start, user_start + len(x)), x.shape[1])
#     # e.g. for user_start = 0, user_size = 3 and seq_length = 4:
#     #      user_ids = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
#
#     # 2. create list of app_categories (app_category is app_category index only)
#     x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
#     app_category_features = x[:, app_category_start:app_category_start + app_category_size]
#     app_categories = np.apply_along_axis(np_find_first_one, 1, app_category_features)
#
#     # return as a matrix
#     return np.column_stack((user_ids, app_categories))
#
#
# def predict(model, predict_buckets, column_names, descr):
#     # Model Evaluation
#     # we are going to do the predictions in batches of batch_size users as well.
#
#     model.eval()
#
#     # get batch_size (use 1st bucket for that)
#     batch_size = predict_buckets[0].shape[1]
#     # user_size = batch_size * len(predict_buckets)
#
#     # App Categories (index_start and size)
#     app_categories = [name for name in column_names if name.startswith('Notif#')]
#     app_category_start = column_names.index(app_categories[0])
#     app_category_size = len(app_categories)
#
#     # list that holds the matrix per batch
#     all_matrices = []
#
#     for bucket_index, bucket in enumerate(tqdm(predict_buckets, desc=descr, leave=False, ncols=100)):
#
#         model.reset_states()
#
#         user_start = bucket_index * batch_size
#
#         # Iterrate through bytes
#         tqdm_batch = tqdm(bucket, desc='>> Batch', leave=False, ncols=100)
#         for batch in tqdm_batch:
#
#             with torch.no_grad():
#
#                 x = torch.from_numpy(batch[:, :, :-2]).to(DEVICE)
#                 y = torch.from_numpy(batch[:, :, -2:-1]).to(DEVICE)
#                 w = torch.from_numpy(batch[:, :, -1]).to(DEVICE)
#
#                 model.reset_states_custom(batch_size)
#
#                 # predict this batch
#                 yhat = model(x)  # predict all elements for these users.
#                 # loss = loss_function(yhat, y)
#
#                 # prepare matrix (columns user_index, app_category_index)
#                 prediction_matrix = produce_matrix(x.cpu().numpy(), user_start, app_category_start, app_category_size)
#
#                 # flatten data per row
#                 y = y.cpu().numpy().flatten()
#                 yhat = yhat.cpu().numpy().flatten()
#                 w = w.cpu().numpy().flatten()
#
#                 # append y, yhat, w columns
#                 prediction_matrix = np.column_stack((prediction_matrix, y, yhat, w))
#
#                 # Columns are:
#                 #   user_index, app_category_index, y, yhat, w
#                 # only keep rows with app_category != - 1 and w != 0
#                 prediction_matrix = prediction_matrix[np.where(prediction_matrix[:, 1] != -1)[0], :]
#                 prediction_matrix = prediction_matrix[np.where(prediction_matrix[:, 4] != 0)[0], :]
#
#                 # append to list
#                 all_matrices.append(prediction_matrix)
#
#         # close tqdm since it is an external variable
#         tqdm_batch.close()
#
#     # return in one np.matrix (user_index, app_category_index, y, yhat, w)
#     return np.vstack(all_matrices)
#
#
# def test(model, model_folder, test_buckets, column_names, uids, desc):
#
#     batch_size_test, seq_len, num_features = test_buckets[0][0].shape
#     num_features -= 2
#     print('%s: Found batch_size=%d, seq_len=%d, num_features=%d' % (desc, batch_size_test, seq_len, num_features))
#
#     prediction_matrix = predict(model, test_buckets, column_names, desc)
#     bce = evaluation(prediction_matrix[:, 2], prediction_matrix[:, 3], prediction_matrix[:, 4])
#
#     # print report
#     tqdm.write("Performance on test set")
#     tqdm.write("%s Loss: %.4f" % (desc, bce))
#     tqdm.write("")
#
#     # produce output_array
#     dateset_type = desc.replace(" ", "_").lower()
#     output_array = produce_output_array(dateset_type, prediction_matrix, uids, column_names)
#
#     return output_array
#
#
# def produce_output_array(dateset_type, prediction_matrix, uuids, column_names):
#
#     # only keep app categories (without the 'Notif#' prefix)
#     app_categories = [name[len("Notif#"):] for name in column_names if name.startswith('Notif#')]
#
#     # flatten uuids (it is a list, so no .flatten())
#     uuids = [item for sublist in uuids for item in sublist]
#
#     # prepare matrix
#     dateset_type = np.repeat(dateset_type, len(prediction_matrix))
#     app_category_array = map(lambda x, app_categories=app_categories: app_categories[x], prediction_matrix[:, 1].astype(int))
#     uuid_array = map(lambda x, uuids=uuids: uuids[x], prediction_matrix[:, 0].astype(int))
#     ypred_array = map(lambda x: 0 if x < 0.5 else 1, prediction_matrix[:, 3])
#
#     output_array = np.array(list(zip(uuid_array, app_category_array, dateset_type, prediction_matrix[:, 2], prediction_matrix[:, 3], ypred_array, prediction_matrix[:, 4])),
#                             dtype=[('uid', 'S32'), ('app_category', 'S32'), ('data_set', 'S32'), ('y', int), ('yhat', 'float32'), ('ypred', int), ('w', 'float32')])
#
#     return output_array
#
#
# def write_output(output_arrays, filename):
#
#     # get filename
#     # output_path = util.ensure_exists('../output')
#     # filepath = os.path.join(output_path, filename)
#     #
#     # # merge output_arrays
#     # output_table = np.concatenate(output_arrays)
#     #
#     # # save matrix into csv
#     # header = "uid,app_category,data_set,y,yhat,ypred,w"
#     # np.savetxt(filepath, output_table, delimiter=",", fmt="%s,%s,%s,%i,%f,%i,%f", header=header, comments='')
#     return
#
# #############
#
# def main(args):
#
#     # create the folder to store the model
#     model_folder = os.path.dirname('../models/%s/' % time.strftime('%Y%m%d-%H%M%S'))
#     os.makedirs(model_folder)
#
#     # Load data and loop for every set
#     data = {}
#     uids = {}
#     column_names = None
#     for mode, bin_folder in DATA_FOLDERS:
#
#         # choose between compressed / uncompressed dataset
#         if COMPRESSED:
#             bin_folder = os.path.join(bin_folder, 'compressed')
#         else:
#             bin_folder = os.path.join(bin_folder, 'uncompressed')
#
#         # Load dataset
#         data_file_path = os.path.join(bin_folder, DATA_FILE)
#         h5f = h5py.File(data_file_path, 'r', libver='latest')
#         num_buckets = h5f.attrs['num_buckets']
#         uids[mode] = h5f.attrs['uuids']
#         if column_names is None:
#             column_names = h5f.attrs['column_names'].tolist()
#         '''
#         meta_data_entries = h5f.attrs.keys()
#         print("Using %s data file %s, %d meta data entries:" % (mode, data_file_path, len(meta_data_entries)))
#         for item in meta_data_entries:
#             print("\t", item + " : ", h5f.attrs[item])
#         print
#         #'''
#
#         print("loading %d %s buckets FROM %s" % (num_buckets, mode, data_file_path))
#         bucket_list = []
#         for bucket_num in tqdm(range(1, num_buckets + 1), ncols=100):
#             name = 'bucket_%d' % bucket_num
#             bucket = h5f[name][:]
#             tqdm.write("\t%s: %s" % (name, bucket.shape))
#             bucket_list.append(bucket)
#         h5f.close()
#         data[mode] = bucket_list[:]
#
#     # Start training...
#     model, output_array_training, output_array_validation = train(model_folder, data['train'], data['valid'], column_names, uids['train'], uids['valid'])
#
#     # Start validation with test dataset...
#     output_array_test = test(model, model_folder, data['test'], column_names, uids['test'], desc='Test')
#
#     # Start validation with test_unk dataset...
#     output_array_test_unk = test(model, model_folder, data['test_unk'], column_names, uids['test_unk'], desc='Test unknown')
#
#     # write output into one csv file
#     tqdm.write("Writing output file ...")
#     output_arrays = [output_array_training, output_array_validation, output_array_test, output_array_test_unk]
#
#     output_filename = PREFIX + 'dlpt_results.csv'
#     write_output(output_arrays, output_filename)
#
#
# def parse_arguments(args):
#
#     parser = argparse.ArgumentParser(description='Creates data frames for deep learning.')
#     parser.add_argument('-tb', '--test-batch', dest='test_batch', action='store_const', const=True, default=False, help="use test batch instead of full batch")
#     parser.add_argument('-s', '--seed', dest='seed', action='store', default=42, help="choose a specific seed for the randomisation process")
#     parser.add_argument('-p', '--prefix', dest='output_prefix', required=False, default='', help="Add a prefix in the output (useful for reports with data from custom seeds).")
#     parsed = vars(parser.parse_args(args))
#
#     return parsed
#
#
# # main
# # -------------------------------------------------------
# if __name__ == '__main__':
#     try:
#         print("Started at: %s" % (datetime.datetime.now()))
#         global start_time
#         start_time = time.time()
#
#         # parse args
#         args = parse_arguments(sys.argv[1:])
#
#         SEED = int(args['seed'])
#         np.random.seed(SEED)
#         torch.manual_seed(SEED)
#         print("Seed: %d" % SEED)
#
#         PREFIX = args['output_prefix']
#         print("Prefix: %s" % PREFIX)
#
#         main(args)
#
#         # save and report elapsed time
#         elapsed_time = time.time() - start_time
#
#         print()
#         print("Completed at: %s. Duration: %s" % (datetime.datetime.now(), str(datetime.timedelta(seconds=int(elapsed_time)))))
#         print()
#
#     except (KeyboardInterrupt, SystemExit):
#         print()
#         print("interrupted - exiting on request")
#         print()
