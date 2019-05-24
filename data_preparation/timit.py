import numpy as np
import pandas as pd
import kaldi_io

import data_preparation as dp

rootdir = '/home/lmkhi/work/kaldi/egs/timit/s5'

datadir = '%s/data/train' % rootdir
alidir = '%s/exp/tri3_ali_train' % rootdir
featnj = 1
alinj = 1
split_ratio = [8, 2]
split_name = ['train', 'valid']

# datadir = '%s/data/test' % rootdir
# alidir = '%s/exp/tri3_ali_test' % rootdir
# featnj = 10
# alinj = 10
# split_ratio = [1]
# split_name = ['test']

corpus = 'timit'
left_context = 50
right_context = 50

featark = 'ark,s,cs:apply-cmvn  --utt2spk=ark:%s/splitNJ/JOB/utt2spk \
    scp:%s/splitNJ/JOB/cmvn.scp \
    scp:%s/splitNJ/JOB/feats.scp \
    ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
    transform-feats %s/exp/tri3/final.mat ark:- ark:- | \
    transform-feats --utt2spk=ark:%s/splitNJ/JOB/utt2spk \
    ark:%s/trans.JOB ark:- ark:- |'%(datadir, datadir, datadir, rootdir, datadir, alidir)

aliark = 'ark:ali-to-pdf %s/final.alimdl \'ark:gunzip -c %s/ali.JOB.gz|\' ark,t:-|'%(alidir,alidir)


if __name__ == '__main__':
    dp.kaldi_alignment_to_numpy_dataset(
        featark=featark, featnj=featnj,
        aliark=aliark, alinj=alinj,
        corpus=corpus, split_ratio=split_ratio, split_name=split_name,
        left_context=left_context, right_context=right_context)
