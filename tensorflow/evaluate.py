import numpy as np
import tensorflow as tf
import math
import time
import os
import logging

import config
import input_data_manager as idm
import table as tb
import utils as ut
import classifier


def train():
    image_file_list = []
    label_file_list = []

    image_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/librispeech.test/test_clean_feat_l30_r30.npy']
    label_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/librispeech.test/test_clean_lab_l30_r30.npy']
    test_image_manager = idm.InputDataManager(image_file_list, read_separate=False, is_label=False,
                                               class_num=config.OUTPUT_DIM)
    test_label_manager = idm.InputDataManager(label_file_list, read_separate=False, is_label=True,
                                               class_num=config.OUTPUT_DIM)

    # valid_data_manager = None
    # model_file = '/data/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/tensorflow/model/libri_build_hn_npsrnn_bidirectional_feed_concat_multilayer_n2000_l30_r30_nodrop_sgd_start0_1/model-4'
    model_file = '/data/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/tensorflow/model/libri_test_build_model2_n3000_l30_r30_nodrop_sgd_start0_1/model-6'


    # model_file = None
    # new_model_file = 'timit.build_model.l5.r5.model'
    # new_model_file = 'libri.build_hn_npsrnn_bidirectional_feed.l30.r30.sgd.start0.001'
    # new_model_file = None

    device_id = '/gpu:1'

    batch_size = 500
    learning_rate = 0.1
    rms_learning_rate = 0.001

    # mnist = input_data.read_data_sets('./', one_hot=False)
    # valid_data_manager.convert_to_one_hot_label()

    input_dim = config.INPUT_DIM
    output_dim = config.OUTPUT_DIM
    # with tf.Graph().as_default():
    with tf.device(device_id):
        if True:
            pl_images = tf.placeholder(tf.float32, [batch_size, input_dim])
            # pl_images = tf.placeholder(tf.float32, [batch_size, config.N_INPUT_FRAME, config.HEIGHT])
            pl_labels = tf.placeholder(tf.int32, shape=(batch_size), name='pl_label')
            pl_keep_prob = tf.placeholder(tf.float32, name='pl_keep_prob')
            pl_learning_rate = tf.placeholder(tf.float32, name='pl_learning_rate')

            # Model Build - Block 1
            # logits, saver = classifier.build_model(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=2000, hidden_2=2000)
            logits, saver = classifier.build_model2(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=3000, hidden_2=3000)
            hiddenlist = [2000] * config.N_INPUT_FRAME
            # logits, saver = classifier.build_npsrnn(pl_images, pl_keep_prob, config.N_INPUT_FRAME, config.HEIGHT, config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_npsrnn_bidirectional(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                         config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_npsrnn_bidirectional_feed(pl_images, pl_keep_prob, 5, 5, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_concat_multilayer(pl_images, pl_keep_prob,
            #                                                                          config.N_LEFT_CONTEXT,
            #                                                                          config.N_RIGHT_CONTEXT,
            #                                                                          config.HEIGHT,
            #                                                                          config.OUTPUT_DIM, hiddenlist)

            # logits, saver = classifier.build_seqnet(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=2000,
            #                                        hidden_2=2000)
            # logits, saver = classifier.build_seqnet3(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=1000,
            #                                        hidden_2=1000)
            # logits, saver = classifier.build_seqnet_npsrnn(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=2000,
            #                                          hidden_2=2000)
            # logits, saver = classifier.build_seqnet_npsrnn8(pl_images, pl_keep_prob, input_dim, output_dim,
            #                                                hidden_1=2000,
            #                                                hidden_2=2000)
            # logits, saver = classifier.build_model2(pl_images, pl_keep_prob, input_dim, output_dim)
            # logits, saver = classifier.build_model_cnn3(pl_images, pl_keep_prob, batch_size, output_dim)


            # Loss for update - Block 2
            loss = classifier.cross_entropy_loss(logits=logits, labels=pl_labels)

            # Parameter Update operator - Block 3
            # optimizer = tf.train.RMSPropOptimizer(rms_learning_rate, 0.6)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=pl_learning_rate)
            train_op = optimizer.minimize(loss)

            # Evaluation Operator
            softmax = tf.nn.softmax(logits)
            num_corrects = tf.nn.in_top_k(softmax, pl_labels, 1)
            eval_op = tf.reduce_sum(tf.cast(num_corrects, tf.int32))  # how many

            # Session
            tfconfig = tf.ConfigProto()
            tfconfig.gpu_options.allow_growth = True
            tfconfig.allow_soft_placement = True
            sess = tf.Session(config=tfconfig)

            # Init all variable
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, model_file)

            b_num_examples, b_true_counts = classifier.check_accuracy(sess, eval_op, pl_images, pl_labels, pl_keep_prob,
                                                                      test_image_manager.images,
                                                                      test_label_manager.images,
                                                                      batch_size, device_id)
            all_num_examples = b_num_examples
            total_true_counts = np.sum(b_true_counts)
            print np.mean(np.asarray(b_true_counts))
            print np.std(np.asarray(b_true_counts))
            prec = float(total_true_counts) / float(all_num_examples)
            print(
                '\tNum examples: %d  Num correct: %d  Test Accuracy @ 1: %0.04f (BEST)' % (
                    all_num_examples, total_true_counts, prec))



if __name__ == '__main__':
    train()
