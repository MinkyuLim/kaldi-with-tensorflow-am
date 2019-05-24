import numpy as np
import pandas as pd
import os
import kaldi_io


def df_data_split(dataset, split_ratio, shuffle=True):
    print 'Spliting DataFrame...'
    # if len(split_ratio) is not 1 and len(split_ratio) is not 2 and len(split_ratio) is not 3:
    #     return None

    if shuffle is True:
        ds = dataset.sample(frac=1)
    else:
        ds = dataset

    split_list = []

    num = 0
    den = np.asarray(split_ratio).sum()

    cur_end = 0
    for i in range(len(split_ratio)):
        num += split_ratio[i]
        cur_start = cur_end
        cur_end = ds.index.size * num / den

        split_list.append(ds.iloc[cur_start:cur_end])

    print 'Done..'
    return split_list
    #
    # if len(split_ratio) == 1:
    #     return [dataset]
    #
    # if len(split_ratio) is 2:
    #     trend = ds.index.size * split_ratio[0] / (split_ratio[0] + split_ratio[1])
    #     tr = ds.iloc[:trend]
    #     va = ds.iloc[trend:]
    #     print 'Done..'
    #     return [tr, va]
    # else:
    #     trend = ds.size * split_ratio[0] / (split_ratio[0] + split_ratio[1] + split_ratio[2])
    #     vaend = ds.size * (split_ratio[0] + split_ratio[1]) / (split_ratio[0] + split_ratio[1] + split_ratio[2])
    #     tr = ds.iloc[:trend]
    #     va = ds.iloc[trend:vaend]
    #     te = ds.iloc[vaend:]
    #     print 'Done..'
    #     return [tr, va, te]


def df_to_npy(df, left_context=5, right_context=5):
    print 'Converting DataFrame to Numpy...'
    cntlab = 0
    cntfeat = 0
    for idx, row in df.iterrows():
        cntlab += row['labs'].shape[0]
        cntfeat += row['feats'].shape[0]

    if cntfeat == cntlab:
        cnt = cntfeat
    else:
        print 'Size not matched'
        return None, None

    ndim = df.iloc[0]['feats'].shape[1]
    nfrm = left_context+right_context+1
    total_feat_npy = np.zeros((cnt, nfrm, ndim))
    total_lab_npy = np.zeros((cnt,), dtype=np.int32)

    index = 0
    for idx, row in df.iterrows():
        exnpy = np.zeros((row['feats'].shape[0]+left_context+right_context, ndim))
        if right_context == 0:
            exnpy[left_context:] = row['feats']
        else:
            exnpy[left_context:-right_context] = row['feats']
        curnpy = np.zeros((row['feats'].shape[0], nfrm, ndim))
        for i in range(row['feats'].shape[0]):
            curnpy[i] = exnpy[i:i+nfrm]
        total_feat_npy[index:index+curnpy.shape[0]] = curnpy
        total_lab_npy[index:index+curnpy.shape[0]] = row['labs'].reshape((-1,))
        index += curnpy.shape[0]

    print total_feat_npy.shape
    print total_lab_npy.shape
    print 'Done...'

    return total_feat_npy, total_lab_npy


def npy_shuffle(feats_npy, labs_npy):
    print 'Shuffling...'
    if feats_npy.shape[0] != labs_npy.shape[0]:
        print 'Size not matched'
        return None, None

    (nSample, nFrm, nDim) = feats_npy.shape

    tmp_shuffle = np.hstack([feats_npy.reshape((nSample, nFrm*nDim)), labs_npy.reshape((nSample, 1))])
    np.random.shuffle(tmp_shuffle)
    tmp_shuffle_tr = tmp_shuffle.transpose()
    shuffle_feat_data = tmp_shuffle_tr[:-1].transpose()
    shuffle_lab_data = tmp_shuffle_tr[-1:].transpose().astype(int)

    print shuffle_feat_data.shape
    print shuffle_lab_data.shape
    print 'Done..'
    return shuffle_feat_data, shuffle_lab_data


def kaldi_alignment_to_numpy_dataset(
        featark=None, featnj=1,
        aliark=None, alinj=1,
        corpus=None, split_ratio=[8, 2], split_name=['train', 'valid'],
        left_context=5, right_context=5
):
    keys = []
    feats = []
    for i in range(1, featnj + 1):
        curfeatark = featark.replace('JOB', str(i)).replace('NJ', str(featnj))
        for key, feat in kaldi_io.read_mat_ark(curfeatark):
            keys.append(key)
            feats.append(feat)
        cur_dataset_feats = pd.DataFrame(data=feats, index=keys, columns=['feats'])
        keys = []
        feats = []
        if i == 1:
            dataset_feats = cur_dataset_feats
        else:
            dataset_feats = dataset_feats.append(cur_dataset_feats)
            print dataset_feats.shape

    dataset_feats = dataset_feats.sort_index()

    keys = []
    labs = []
    for i in range(1, alinj + 1):
        curaliark = aliark.replace('JOB', str(i))
        for key, lab in kaldi_io.read_ali_ark(curaliark):
            keys.append(key)
            labs.append(lab.reshape((lab.shape[0], 1)))
        cur_dataset_labs = pd.DataFrame(data=labs, index=keys, columns=['labs'])
        keys = []
        labs = []
        if i == 1:
            dataset_labs = cur_dataset_labs
        else:
            dataset_labs = dataset_labs.append(cur_dataset_labs)
            print dataset_labs.shape

    dataset_labs = dataset_labs.sort_index()

    print dataset_feats.shape
    print dataset_labs.shape
    dataset = pd.concat([dataset_feats, dataset_labs], axis=1)

    df_list = df_data_split(dataset=dataset, split_ratio=split_ratio, shuffle=False)

    if os.path.isdir('./%s'%corpus):
        print 'Corpus dir exists..'
    else:
        print 'Create new corpus dir..'
        os.makedirs('./%s'%corpus)

    for i in range(len(df_list)):
        cur_feat_npy, cur_lab_npy = df_to_npy(df=df_list[i], left_context=left_context, right_context=right_context)
        shu_feat_npy, shu_lab_npy = npy_shuffle(feats_npy=cur_feat_npy, labs_npy=cur_lab_npy)

        print 'Saving npys...'
        f_feat = './%s/%s_feat_l%d_r%d' % (corpus, split_name[i], left_context, right_context)
        f_lab = './%s/%s_lab_l%d_r%d' % (corpus, split_name[i], left_context, right_context)
        np.save(file=f_feat, arr=shu_feat_npy)
        np.save(file=f_lab, arr=shu_lab_npy)