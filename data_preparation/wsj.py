import numpy as np
import pandas as pd
import kaldi_io

import data_preparation as dp

corpus = 'wsj_pytorchkaldi'
left_context = 7
right_context = 7

rootdir = '/home/lmkhi/work/project/kaldi_tf/kaldi/egs/librispeech/s5'

datadir = '%s/fmllr/train_clean_100' % rootdir
alidir = '%s/exp/tri4b_ali_clean_100' % rootdir
featnj = 10
alinj = 30
split_name = []
for i in range(10):
    split_name.append('train.%d'%i)
split_ratio = [1]*10

# datadir = '%s/data/test_dev93' % rootdir
# alidir = '%s/exp/tri4b_ali_dev93' % rootdir
# featnj = 1
# alinj = 1
# split_name = ['valid']
# split_ratio = [1]

# datadir = '%s/data/test_eval92' % rootdir
# alidir = '%s/exp/tri4b_ali_eval92' % rootdir
# featnj = 1
# alinj = 1
# split_name = ['test']
# split_ratio = [1]


# featark = 'ark,s,cs:apply-cmvn  --utt2spk=ark:%s/splitNJ/JOB/utt2spk \
# scp:%s/splitNJ/JOB/cmvn.scp \'scp:%s/utils/filter_scp.pl --exclude %s/err_utt %s/splitNJ/JOB/feats.scp |\' ark:- | \
# splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats %s/exp/tri4b/final.mat ark:- ark:- | \
# transform-feats --utt2spk=ark:%s/splitNJ/JOB/utt2spk ark:%s/trans.JOB ark:- ark:- |'\
#           % (datadir, datadir, rootdir, alidir, datadir, rootdir, datadir, alidir)

#pytorch-kaldi
featark = 'ark,s,cs:apply-cmvn  --utt2spk=ark:%s/utt2spk \
ark:%s/data/cmvn_speaker.ark \'scp:%s/utils/filter_scp.pl --exclude %s/err_utt %s/feats.scp |\' ark:- |'\
          % (datadir, datadir, rootdir, alidir, datadir)

aliark = 'ark:ali-to-pdf %s/final.alimdl \'ark:gunzip -c %s/ali.JOB.gz|\' ark,t:-|'\
         % (alidir, alidir)


if __name__ == '__main__':
    dp.kaldi_alignment_to_numpy_dataset(
        featark=featark, featnj=featnj,
        aliark=aliark, alinj=alinj,
        corpus=corpus, split_ratio=split_ratio, split_name=split_name,
        left_context=left_context, right_context=right_context)
