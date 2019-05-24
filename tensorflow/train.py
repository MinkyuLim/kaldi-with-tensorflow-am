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

    # wsj
    # for i in range(10):
    #     image_file_list.append(
    #         '/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/train.%d_feat_l7_r7.npy' % i)
    #     label_file_list.append(
    #         '/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/train.%d_lab_l7_r7.npy' % i)

    # rm
    image_file_list.append(
        '/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/rm/train_feat_l4_r4.npy')
    label_file_list.append(
        '/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/rm/train_lab_l4_r4.npy')
    train_image_manager = idm.InputDataManager(image_file_list, read_separate=False, is_thread=True, is_label=False,
                                               class_num=config.OUTPUT_DIM)
    train_label_manager = idm.InputDataManager(label_file_list, read_separate=False, is_thread=True, is_label=True,
                                               class_num=config.OUTPUT_DIM)

    # wsj
    # image_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/valid_7_7_feat_l7_r7.npy']
    # label_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/valid_7_7_lab_l7_r7.npy']

    # rm
    image_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/rm/valid_feat_l4_r4.npy']
    label_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/rm/valid_lab_l4_r4.npy']
    valid_image_manager = idm.InputDataManager(image_file_list, read_separate=False, is_thread=False, is_label=False,
                                               class_num=config.OUTPUT_DIM)
    valid_label_manager = idm.InputDataManager(label_file_list, read_separate=False, is_thread=False, is_label=True,
                                               class_num=config.OUTPUT_DIM)

    # image_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/test_feat_l30_r30.npy']
    # label_file_list = ['/home/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/data_preparation/wsj/test_lab_l30_r30.npy']
    # test_data_manager = idm.InputDataManager(image_file_list, label_file_list, read_separate=False,
    #                                           class_num=config.OUTPUT_DIM)
    # test_data_manager.images = test_data_manager.images.reshape(
    #     (test_data_manager.images.shape[0], config.INPUT_DIM))

    # valid_data_manager = None
    model_file = None
    # model_file = '/data/lmkhi/work/project/HN-NPS-RNN/HN-NPS-RNN/tensorflow/model/wsj.build_model.l30.r30.startlr0.1/model-25'
    # new_model_file = 'timit.build_model.l5.r5.model'
    # new_model_file = 'wsj.build_model_nodrop.n2400.l7.r7.startlr0.1'
    new_model_file = 'rm.build_model_nodrop.l2.n1000.l4.r4.startlr0.1'
    # new_model_file = None

    device_id = '/gpu:3'

    batch_size = 500
    learning_rate = 0.1

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
            # logits, saver = classifier.build_model(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=2400, hidden_2=2400)
            logits, saver = classifier.build_model_nodrop(pl_images, pl_keep_prob, input_dim, output_dim, hidden_1=1000, hidden_2=1000)
            hiddenlist = [1000] * config.N_INPUT_FRAME
            # logits, saver = classifier.build_npsrnn(pl_images, pl_keep_prob, config.N_INPUT_FRAME, config.HEIGHT, config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_npsrnn_bidirectional(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                         config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_npsrnn_bidirectional_feed(pl_images, pl_keep_prob, 5, 5, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
            # logits, saver = classifier.build_hn_npsrnn_bidirectional(pl_images, pl_keep_prob, config.N_LEFT_CONTEXT, config.N_RIGHT_CONTEXT, config.HEIGHT,
            #                                                       config.OUTPUT_DIM, hiddenlist)
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
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, 0.6)
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

            tmp_model = './tmp_model'
            if new_model_file is not None:
                model_dir = './model/' + new_model_file
                if os.path.isdir(model_dir):
                    print 'Model exists..'
                    print 'Save to tmp model..'
                    save_model_file = tmp_model
                else:
                    print 'Create new model dir..'
                    os.makedirs(model_dir)
                    save_model_file = model_dir + '/model'
            else:
                if os.path.isfile('%s.meta' % model_file):
                    saver.restore(sess, model_file)
                    print("Restore %s success" % model_file)
                    save_model_file = model_file
                    model_dir = os.path.dirname(model_file)
                else:
                    print 'Cannot find restore model file..'
                    print 'Save to tmp model..'
                    save_model_file = tmp_model
                    model_dir = './'

            os.system('cp ./train.py %s' % model_dir + '/')

            logger = ut.init_logger(save_model_file+'.log', stream_level=logging.DEBUG, file_level=logging.DEBUG)
            # raw_input("")

            # --------------------------------------------------------------------#
            # train - Block 4
            no_best_validation = 0
            step = 0
            check_losses = []
            best_validation_accuracy = 0.
            # for epoch in range(1000000):
            epoch = 0
            while True:
                if no_best_validation is 10:
                    learning_rate = learning_rate * 0.9
                    if learning_rate < 0.00001:
                        learning_rate = 0.00001
                    # print 'learning_rate: %f' % learning_rate
                    logger.debug('learning_rate: %f' % learning_rate)
                    no_best_validation = 0
                batch_images = train_image_manager.next_batch(batch_size)
                batch_labels = train_label_manager.next_batch(batch_size)
                # batch_images = batch_images.reshape((batch_size, 40, 40, 1))
                # a = mnist.train.next_batch(batch_size)
                # batch_images = a[0]
                # batch_labels = a[1]
                start_time = time.time()
                feed_dict = {
                    # pl_images: batch_images.reshape((batch_size, config.INPUT_DIM)),
                    pl_images: batch_images,
                    pl_labels: batch_labels,
                    pl_keep_prob: 0.5,
                    pl_learning_rate: learning_rate,
                }

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                check_losses.append(loss_value)

                duration = time.time() - start_time

                # monitor loss per step
                if step % 50 == 0:
                    logger.debug('Epoch %d - Step %d: loss = %.2f (%.3f sec)' % \
                          (train_image_manager.epoch, step, np.mean(check_losses), duration))
                    check_losses = []  # init
                step += 1

                # every n epoch, check performance
                # if epoch % 10 == 0 :
                is_test = False
                if epoch != train_image_manager.epoch:
                    epoch = train_image_manager.epoch

                    learning_rate = learning_rate * 0.9
                    optimizer = tf.train.GradientDescentOptimizer(learning_rate=pl_learning_rate)
                    train_op = optimizer.minimize(loss)

                    if epoch > 100:
                        return
                # if False:
                    b_num_examples, b_true_counts = classifier.check_accuracy(sess, eval_op, pl_images, pl_labels, pl_keep_prob,
                                                                              valid_image_manager.images, valid_label_manager.images,
                                                                   batch_size, device_id)
                    # b_num_examples, b_true_counts = check_accuracy(sess, eval_op, pl_images, pl_labels, pl_keep_prob,
                    #                                            mnist.validation.images, mnist.validation.labels,
                    #                                            batch_size, device_id)
                    all_num_examples = b_num_examples
                    total_true_counts = np.sum(b_true_counts)
                    prec = float(total_true_counts) / float(all_num_examples)
                    if prec > best_validation_accuracy:
                        no_best_validation = 0
                        saver.save(sess, save_model_file, epoch)
                        best_validation_accuracy = prec
                        is_test = True
                        logger.debug('\t[epoch: %d] Num examples: %d  Num correct: %d  Validataion Accuracy @ 1: %0.04f (BEST)' % (
                        epoch, all_num_examples, total_true_counts, prec))
                    #
                    #     b_num_examples, b_true_counts = classifier.check_accuracy(sess, eval_op, pl_images, pl_labels,
                    #                                                               pl_keep_prob,
                    #                                                               test_data_manager.images,
                    #                                                               test_data_manager.labels,
                    #                                                               batch_size, device_id)
                    #     all_num_examples = b_num_examples
                    #     total_true_counts = np.sum(b_true_counts)
                    #     prec = float(total_true_counts) / float(all_num_examples)
                    #     logger.debug(
                    #         '\t[epoch: %d] Num examples: %d  Num correct: %d  Test Accuracy @ 1: %0.04f' % (
                    #             epoch, all_num_examples, total_true_counts, prec))
                    else:
                        no_best_validation = no_best_validation + 1
                        logger.debug('\t[epoch: %d] Num examples: %d  Num correct: %d  Validataion Accuracy @ 1: %0.04f' % (
                        epoch, all_num_examples, total_true_counts, prec))


                # is_test = False
                # if is_test is True:
                #     b_num_examples, b_true_counts = check_accuracy(sess, eval_op, pl_images, pl_labels, pl_keep_prob,
                #                                                    te_smp, te_lab, batch_size, device_id)
                #     all_num_examples = b_num_examples
                #     total_true_counts = np.sum(b_true_counts)
                #     prec = float(total_true_counts) / float(all_num_examples)
                #     print '\tNum examples: %d  Num correct: %d  Test Accuracy @ 1: %0.04f' % (
                #     all_num_examples, total_true_counts, prec)


if __name__ == '__main__':
    train()
