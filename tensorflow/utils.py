import os
import logging

stream_handler = None
file_handler = None

def gen_key_from_file(file_path):
    basename = os.path.basename(file_path)
    key = basename.split('.')[0]
    return key


def gen_file_path(in_dir, key, in_ext):
    return in_dir.rstrip('/') + '/' + key + '.' + in_ext.lstrip('.')


def init_logger(file_name, stream_level, file_level):
    global stream_handler
    global file_handler
    logger = logging.getLogger("logger")
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(file_name)
    logger.setLevel(logging.DEBUG)
    stream_handler.setLevel(stream_level)
    file_handler.setLevel(file_level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def close_logger(logger):
    global stream_handler
    global file_handler
    logger.removeHandler(file_handler)
    logger.removeHandler(stream_handler)
    file_handler.close()
    stream_handler.close()


def get_file_size(file_path):
    try:
        size = os.path.getsize(file_path)
    except:
        size = 0
    return size