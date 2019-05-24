import numpy as np
import pandas as pd
import kaldi_io

import data_preparation as dp

rootdir = '/home/lmkhi/work/project/kaldi_tf/kaldi/egs/librispeech/s5'

datadir = '%s/data/train_960' % rootdir
# datadir = '%s/data/test_clean' % rootdir
# alidir = '%s/exp/tri6b' % rootdir
alidir = '%s/exp/nnet7a_960_gpu_align' % rootdir
# alidir = '%s/exp/tri6b_test_clean_ali' % rootdir

# alidir2 = '%s/exp/nnet7a_960_gpu_ali' % rootdir
featnj = 40
alinj = 10
split_name = []
n_split = 200
for i in range(n_split):
    split_name.append('train.%d'%i)
split_ratio = [1]*n_split

# datadir = '%s/data/test_clean' % rootdir
# alidir = '%s/exp_LIBRISPEECH_ORIGINAL/tri5b_ali_test_clean' % rootdir
# featnj = 10
# alinj = 10
# split_name = ['test_clean']
# split_ratio = [1]

# datadir = '%s/data/test_other' % rootdir
# alidir = '%s/exp_LIBRISPEECH_ORIGINAL/tri5b_ali_test_other' % rootdir
# featnj = 10
# alinj = 10
# split_name = ['test_other']
# split_ratio = [1]


# featark = 'ark,s,cs:apply-cmvn  --utt2spk=ark:%s/splitNJ/JOB/utt2spk \
# scp:%s/splitNJ/JOB/cmvn.scp \'scp:%s/utils/filter_scp.pl --exclude %s/err_utt %s/splitNJ/JOB/feats.scp |\' ark:- | \
# splice-feats --left-context=3 --right-context=3 ark:- ark:- | transform-feats %s/exp/tri5b/final.mat ark:- ark:- | \
# transform-feats --utt2spk=ark:%s/splitNJ/JOB/utt2spk ark:%s/trans.JOB ark:- ark:- |'\
#           % (datadir, datadir, rootdir, alidir, datadir, rootdir, datadir, alidir)

# featark = 'ark,s,cs:%s/utils/filter_scp.pl --exclude %s/err_utt %s/splitNJ/JOB/feats.scp | \
# apply-cmvn  --utt2spk=ark:%s/splitNJ/JOB/utt2spk scp:%s/splitNJ/JOB/cmvn.scp scp:- ark:- | \



# transform-feats --utt2spk=ark:%s/splitNJ/JOB/utt2spk ark:%s/trans.JOB ark:- ark:- |'\
#     % (rootdir, alidir, datadir, datadir, datadir, rootdir, datadir, alidir)

# featark = 'ark,s,cs:%s/utils/filter_scp.pl --exclude %s/err_utt %s/splitNJ/JOB/feats.scp | \
# apply-cmvn  --utt2spk=ark:%s/splitNJ/JOB/utt2spk scp:%s/splitNJ/JOB/cmvn.scp scp:- ark:- | \
# splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
# transform-feats %s/exp/nnet7a_960_gpu/final.mat ark:- ark:- | \
# transform-feats --utt2spk=ark:%s/splitNJ/JOB/utt2spk ark:%s/exp/nnet7a_960_gpu/decode_tgsmall_test_clean_0.1_prior/trans.ark ark:- ark:- |'\
#     % (rootdir, alidir, datadir, datadir, datadir, rootdir, datadir, rootdir)

featark = 'ark,s,cs:%s/utils/filter_scp.pl --exclude %s/exp/tri6b/err_utt %s/data/train_960/splitNJ/JOB/feats.scp |\
apply-cmvn  --utt2spk=ark:%s/data/train_960/splitNJ/JOB/utt2spk \
scp:%s/data/train_960/splitNJ/JOB/cmvn.scp scp:- ark:- | \
splice-feats --left-context=3 --right-context=3 ark:- ark:- | \
transform-feats %s/exp/nnet7a_960_gpu/final.mat ark:- ark:- | \
transform-feats --utt2spk=ark:%s/data/train_960/splitNJ/JOB/utt2spk ark:%s/exp/tri6b/trans.JOB ark:- ark:- |'\
    % (rootdir, rootdir, rootdir, rootdir, rootdir, rootdir, rootdir, rootdir)

# featark = 'ark,s,cs:%s/utils/filter_scp.pl --exclude %s/err_utt %s/data/raw_fbank_train_960_fbank.JOB.scp | \
# apply-cmvn  --utt2spk=ark:%s/utt2spk scp:%s/cmvn.scp scp:- ark:- |'\
#     % (rootdir, alidir, datadir, datadir, datadir)

aliark = 'ark:ali-to-pdf %s/final.mdl \'ark:gunzip -c %s/ali.JOB.gz|\' ark,t:-|'\
         % (alidir, alidir)


corpus = 'librispeech.nnet7a_960_gpu_align.l15.r15'
left_context = 15
right_context = 15






if __name__ == '__main__':
    dp.kaldi_alignment_to_numpy_dataset(
        featark=featark, featnj=featnj,
        aliark=aliark, alinj=alinj,
        corpus=corpus, split_ratio=split_ratio, split_name=split_name,
        left_context=left_context, right_context=right_context)
