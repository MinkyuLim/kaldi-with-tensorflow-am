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
    model_dir = '/home/lmkhi/work/project/pycharm_transfer/hn-pns-rnn/tensorflow/model/libri_build_lstm_rnn_n1000_layer1_l7_r7'
    epoch = 9
    model_epoch = 'model-'+str(epoch)
    model_file = model_dir + '/' + model_epoch

    os.system('cp ./export_model.py %s' % model_dir + '/')

    device_id = '/cpu:0'

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
            ################
            pl_images = tf.placeholder(tf.float32, [batch_size, input_dim])
            # pl_images = tf.placeholder(tf.float32, [batch_size, config.N_INPUT_FRAME, config.HEIGHT])
            pl_labels = tf.placeholder(tf.int32, shape=(batch_size), name='pl_label')
            pl_keep_prob = tf.placeholder(tf.float32, name='pl_keep_prob')
            pl_learning_rate = tf.placeholder(tf.float32, name='pl_learning_rate')

            # Model Build - Block 1
            # logits, saver = classifier.build_model(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=2400, hidden_2=2400)
            # logits, saver = classifier.build_model_nodrop(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=1000, hidden_2=1000)
            # logits, saver = classifier.build_model2(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=5000, hidden_2=5000, nodrop=True)
            # logits, saver = classifier.build_model2_batch(pl_images, output_dim, is_training=True)
            logits, saver = classifier.build_lstm_rnn(pl_images, config.N_INPUT_FRAME, config.HEIGHT, output_dim, 1000)
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
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_concat(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_multilayer(pl_images, pl_keep_prob,
            #                                                                      config.N_LEFT_CONTEXT,
            #                                                                      config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                                                      config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_concat_multilayer(pl_images, pl_keep_prob,
            #                                                                                 config.N_LEFT_CONTEXT,
            #                                                                                 config.N_RIGHT_CONTEXT,
            #                                                                                 config.HEIGHT,
            #                                                                                 config.OUTPUT_DIM,
            #                                                                                 hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_concat_ffnnfeat(pl_images, pl_keep_prob,
            #                                                                               config.N_LEFT_CONTEXT,
            #                                                                               config.N_RIGHT_CONTEXT,
            #                                                                               config.HEIGHT,
            #                                                                               config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional_feed_concat_ffnnfeat_sameUt(pl_images,
            #                                                                                      pl_keep_prob,
            #                                                                                      config.N_LEFT_CONTEXT,
            #                                                                                      config.N_RIGHT_CONTEXT,
            #                                                                                      config.HEIGHT,
            #                                                                                      config.OUTPUT_DIM,
            #                                                                                      hiddenlist)
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

    builder = tf.saved_model.builder.SavedModelBuilder(model_dir+'/'+str(epoch))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(pl_images)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(softmax)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'input': tensor_info_x},
                    outputs={'output': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        },
        legacy_init_op=tf.group(tf.tables_initializer(), name='legacy_init_op'))

    builder.save()
            #######################

if __name__ == '__main__':
    train()


