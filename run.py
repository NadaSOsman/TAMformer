from pie_data import PIE
from jaad_data import JAAD
from data_generator import DataGenerator, DataGetter
from tamformer import TAMformer
import os
import sys
import yaml
import numpy as np
import getopt
import pickle
import cv2
import tensorflow as tf
import random as rn
from argparse import ArgumentParser
import copy
from tensorflow.compat.v1.keras import backend as K
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
K.set_session(sess)

from tensorflow.keras.metrics import AUC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow import keras


def run(config_path, auxiliary_loss, test, resume):
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)

    print(configs['model_opts']['dataset'], '--------------------------------------')
    tte = configs['model_opts']['time_to_event']
    configs['data_opts']['min_track_size'] = configs['model_opts']['obs_length'] + 2*tte

    if configs['model_opts']['dataset'] == 'jaad':
        imdb = JAAD(data_path= configs['data_opts']['path_to_dataset'])
    else:
        imdb = PIE(data_path= configs['data_opts']['path_to_dataset'])

    data_raw_train = imdb.generate_data_trajectory_sequence('train', **configs['data_opts'])
    data_raw_test = imdb.generate_data_trajectory_sequence('test', **configs['data_opts'])
    data_raw_val = imdb.generate_data_trajectory_sequence('val', **configs['data_opts'])

    data_getter_train = DataGetter('train', data_raw_train, configs['model_opts'])
    data_getter_test = DataGetter('test', data_raw_test, configs['model_opts'])
    data_getter_val = DataGetter('val', data_raw_test, configs['model_opts'])

    data_train = data_getter_train.get_data()
    test_data = data_getter_test.get_data()
    val_data = data_getter_val.get_data()

    tamformer = TAMformer(configs['model_opts'], auxiliary_loss).tamformer()
    model_name = configs['model_opts']['model_path']\
                 +'/tamformer_'+configs['model_opts']['dataset']+'_'\
                 +'_'.join(configs['model_opts']['obs_input_type'])+'_'\
                 +str(configs['model_opts']['lr'])+'.h5'

    if test or resume:
        print("Lodaing "+model_name+" ...")
        tamformer.load_weights(model_name, by_name=False, skip_mismatch=False)
    if not test:
        class_w = class_weights(configs['model_opts']['apply_class_weights'],
                                     data_train['count'],
                                     configs['model_opts']['negative_weight'],
                                     configs['model_opts']['positive_weight'])
        optimizer = get_optimizer(configs['model_opts']['optimizer'])(learning_rate=configs['model_opts']['lr'])
        w = [class_w[0], class_w[1]]
        tamformer.compile(loss=weighted_binary_crossentropy(weights=w),
                          optimizer=optimizer,
                          metrics=['accuracy'])

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_name,
                                                                 save_weights_only=True,
                                                                 monitor='val_loss',
                                                                 mode='min',
                                                                 save_best_only=True)
        history = tamformer.fit(x=data_train['data'][0],
                                y=None,
                                batch_size=configs['model_opts']['batch_size'],
                                epochs=configs['model_opts']['epochs'],
                                validation_data=val_data['data'][0],
                                verbose=1,
                                callbacks=[checkpoint_callback])

        tamformer = TAMformer(configs['model_opts'], auxiliary_loss).tamformer()
        tamformer.load_weights(model_name)

    print("Testing ...")
    test_results = tamformer.predict(test_data['data'][0], verbose=1)
    best_perf_acc = [0 for i in range(40)]
    best_perf_auc = [0 for i in range(40)]
    best_perf_f1 = [0 for i in range(40)]
    AT = np.flip(np.arange(0, 4.1, 0.1))
    for t in np.arange(0.01, 1.0, 0.01):
       test_results_array = np.array([np.where(test_results[i]>=t, 1, 0) for i in range(40)])
       average = 'binary'
       multi_class = 'raise'
       count = 0
       index = int(configs['model_opts']['interval']/configs['model_opts']['step'])
       masking_index = (test_data['data'][2]/configs['model_opts']['step']).astype(int)
       for i in range(len(test_results)):
           rev_index = int((configs['model_opts']['seq_len']-configs['model_opts']['obs_length'])/configs['model_opts']['step'])\
                            + int(configs['model_opts']['obs_length']/configs['model_opts']['step']) - i
           acc = accuracy(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index)
           f1 = score_f1(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)
           auc = score_auc(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, multi_class=multi_class)
           precision = score_precision(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)
           recall = score_recall(test_data['data'][1][0][i], test_results_array[i], rev_index, masking_index, average=average)

           if best_perf_f1[count]+best_perf_auc[count]<=f1+auc:
               best_perf_f1[count] = f1
               best_perf_auc[count] = auc
               best_perf_acc[count] = acc
           count += 1
    count = 0
    for i in range(len(test_results)):
        print(AT[count],':' ,'acc:', best_perf_acc[count], '- auc:', best_perf_auc[count], '- f1:', best_perf_f1[count])
        count += 1

def accuracy(true, pred, index, masking_index):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return accuracy_score(y_true, y_pred)

def score_f1(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return f1_score(y_true, y_pred, average=average)

def score_auc(true, pred, index, masking_index, multi_class):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return roc_auc_score(y_true, y_pred, multi_class=multi_class)

def score_precision(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return precision_score(y_true, y_pred, average=average)

def score_recall(true, pred, index, masking_index, average):
    masking_index = masking_index >= index
    y_true =  np.array([true[i] for i in range(len(masking_index)) if masking_index[i]==1])
    y_pred =  np.array([pred[i] for i in range(len(masking_index)) if masking_index[i]==1])
    return recall_score(y_true, y_pred, average=average)


def class_weights(apply_weights, sample_count, w_neg, w_pos):
    if not apply_weights:
        return None

    total = sample_count['neg_count'] + sample_count['pos_count']
    neg_weight = w_neg #sample_count['pos_count']/total
    pos_weight = w_pos #sample_count['neg_count']/total

    print("### Class weights: negative {:.3f} and positive {:.3f} ###".format(neg_weight, pos_weight))
    return {0: neg_weight, 1: pos_weight}


def weighted_binary_crossentropy(weights, out_weight=1.0):
    def loss_func(y_true, y_pred):
        tf_y_true = tf.cast(y_true, dtype=y_pred.dtype)
        tf_y_pred = tf.cast(y_pred, dtype=y_pred.dtype)
        weights_v = tf.where(tf.equal(tf_y_true, 1), weights[1], weights[0])
        ce = K.binary_crossentropy(y_pred, y_true)
        loss = K.mean(tf.multiply(ce, weights_v))
        return loss*out_weight
    return loss_func


def get_optimizer(optimizer):
    assert optimizer.lower() in ['adam', 'sgd', 'rmsprop'], \
    "{} optimizer is not implemented".format(optimizer)
    if optimizer.lower() == 'adam':
        return Adam
    elif optimizer.lower() == 'sgd':
        return SGD
    elif optimizer.lower() == 'rmsprop':
        return RMSprop



if __name__ == '__main__':
    parser = ArgumentParser(description="Train-Test program for TAMformer")
    parser.add_argument('--config_file', type=str, help="Path to the directory to load the config file")
    parser.add_argument('--auxiliary_loss', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', action='store_true')

    args = parser.parse_args()
    run(args.config_file, args.auxiliary_loss, args.test, args.resume)
