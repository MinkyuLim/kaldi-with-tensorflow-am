# -*- coding: utf-8 -*-

import numpy as np
from collections import deque
from os import listdir, path
import threading

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def _get_file_lists(image_list_file_path, label_list_file_path):
    train_image_file_list = []
    train_label_file_list = []
    valid_image_file_list = []
    valid_label_file_list = []
    test_image_file_list = []
    test_label_file_list = []
    f_image = open(image_list_file_path, 'r')
    f_label = open(label_list_file_path, 'r')
    cnt = 0
    for ll in f_image.readlines():
        if cnt % 10 < 8:
            train_image_file_list.append(ll.strip())
        elif cnt % 10 == 8:
            valid_image_file_list.append(ll.strip())
        else:
            test_image_file_list.append(ll.strip())
        cnt += 1
    cnt = 0
    for ll in f_label.readlines():
        if cnt % 10 < 8:
            train_label_file_list.append(ll.strip())
        elif cnt % 10 == 8:
            valid_label_file_list.append(ll.strip())
        else:
            test_label_file_list.append(ll.strip())
        cnt += 1
    f_image.close()
    f_label.close()

    return train_image_file_list, train_label_file_list,\
           valid_image_file_list, valid_label_file_list,\
           test_image_file_list, test_label_file_list


def get_file_list(data_dir, label_dir, filename_list):
    image_file_list = []
    label_file_list = []

    for filename in filename_list:
        image_file_list.append(path.join(data_dir, filename))
        label_file_list.append(path.join(label_dir, filename))

        assert path.isfile(path.join(data_dir, filename))
        assert path.isfile(path.join(label_dir, filename))

    return image_file_list, label_file_list


def _get_file_lists_from_dir(train_data_dir, train_label_dir,
                             valid_data_dir, valid_label_dir,
                             test_data_dir, test_label_dir):

    train_files = [f for f in listdir(train_data_dir) if path.isfile(path.join(train_data_dir, f)) and '.npy' in f]
    valid_files = [f for f in listdir(valid_data_dir) if path.isfile(path.join(valid_data_dir, f)) and '.npy' in f]
    test_files = [f for f in listdir(test_data_dir) if path.isfile(path.join(test_data_dir, f)) and '.npy' in f]
    np.random.shuffle(train_files)
    np.random.shuffle(valid_files)
    np.random.shuffle(test_files)

    train_image_file_list, train_label_file_list = get_file_list(train_data_dir, train_label_dir, train_files)
    valid_image_file_list, valid_label_file_list = get_file_list(valid_data_dir, valid_label_dir, valid_files)
    test_image_file_list, test_label_file_list = get_file_list(test_data_dir, test_label_dir, test_files)

    return train_image_file_list, train_label_file_list, \
           valid_image_file_list, valid_label_file_list, \
           test_image_file_list, test_label_file_list


def _read_image_file(file_path):
    image = np.load(file_path)
    return image


def _read_label_file(file_path):
    label = np.load(file_path)
    # label = np.load(file_path).transpose()
    return label.reshape(label.shape[0])


def _file_load_thread_worker(idm):
    idm._read_next_image_file_thread()
    return


class InputDataManager:
    image_file_list = None
    q_image_file = None
    image_size = []
    images = None
    current_index = 0
    epoch = 0
    n_file_open = 0
    read_separate = True
    conv3d = False
    class_num = 2
    is_label = False

    thr = None
    is_thread = False
    images_sub = None


    def __init__(self, image_file_list, read_separate=True, is_thread=False, is_label=False, class_num=2):
        self.class_num = class_num
        self.image_file_list = image_file_list
        self.q_image_file = deque(self.image_file_list)
        # self.image_size = image_size

#        self._shuffle_images_and_labels()
        self.epoch = 0
        self.n_file_open = 0
        self.read_separate = read_separate

        self.is_label = is_label

        if read_separate is True:
            self._read_next_image_file()
            if is_thread is True:
                self.is_thread = is_thread
                self.thr = threading.Thread(name='reader', target=_file_load_thread_worker, args=(self,))
                self.thr.setDaemon(True)
                self.thr.start()
        else:
            self.read_all()
        return

    def _read_next_image_file(self):
        image_file_path = self.q_image_file.popleft()
        print (image_file_path)
        if self.is_label is False:
            self.images = _read_image_file(image_file_path)
        else:
            self.images = _read_label_file(image_file_path)
        self.q_image_file.append(image_file_path)
        # print 'read next image : %s' % image_file_path
        return image_file_path


    def _read_next_image_file_thread(self):
        image_file_path = self.q_image_file.popleft()
        print (image_file_path)
        if self.is_label is False:
            self.images_sub = _read_image_file(image_file_path)
        else:
            self.images_sub = _read_label_file(image_file_path)
        self.q_image_file.append(image_file_path)
        # print 'read next image : %s' % image_file_path
        return image_file_path


    def next_batch(self, batch_size):
        """ batch size만큼 데이터를 읽어오는 기능"""
        if self.current_index + batch_size > self.images.shape[0]:
            if self.read_separate is True:
                if self.is_thread is False:
                    print ('reading next file')
                    self._read_next_image_file()
                else:
                    print ('reading next file with thread')
                    self.thr.join()
                    self.thr = threading.Thread(name='reader', target=_file_load_thread_worker, args=(self,))
                    self.thr.setDaemon(True)
                    self.images = self.images_sub
                    self.thr.start()

                self.n_file_open += 1
                if self.n_file_open % len(self.image_file_list) == 0:
                    self.epoch += 1
            else:
                self.epoch += 1
            self.current_index = 0

        images = self.images[self.current_index:self.current_index + batch_size]
        self.current_index = self.current_index + batch_size

        return images

    def read_all(self):
        """ 전체 data 및 label 목록을 읽어 하나의 데이터로 구성한다. 주로 한번에 메모리에 올라가는 작은 데이터에 사용 """
        # 파일 목록에서 첫번째 image와 label 파일을 읽어 images와 labels를 생성한다
        if self.is_label is False:
            self.images = _read_image_file(self.image_file_list[0])
                # self.labels = self.labels.reshape(self.labels.shape[0])
            # 이후 나머지 파일들을 하나씩 읽어 위에서 생성한 images와 labels의 뒤에 추가한다
            for i in range(1, len(self.image_file_list)):
                self.images = np.append(self.images, _read_image_file(self.image_file_list[i]), axis=0)
        else:
            self.images = _read_label_file(self.image_file_list[0])
            # self.labels = self.labels.reshape(self.labels.shape[0])
            # 이후 나머지 파일들을 하나씩 읽어 위에서 생성한 images와 labels의 뒤에 추가한다
            for i in range(1, len(self.image_file_list)):
                self.images = np.append(self.images, _read_label_file(self.image_file_list[i]), axis=0)
        self.read_separate = False
        return

    # def convert_to_one_hot_label(self):
    #     tmp_labels = self.labels.astype(np.int32)
    #     self.labels = np.eye(self.class_num)[tmp_labels]

if __name__ == "__main__":
    image_file_list = ['/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r44100_c1_b16/image_dev_fold1_train.npy']
    label_file_list = ['/home/splab/PersonalMedia/tf/DCASE2016/src/aed/data/r44100_c1_b16/label_dev_fold1_train.npy']
    input_data_manager = InputDataManager(image_file_list, None, read_separate=False, class_num=15)
    batch_image = input_data_manager.next_batch(100)
