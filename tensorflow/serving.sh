#!/usr/bin/env bash

docker run -p 8501:8501 \
  --mount type=bind,source=/home/lmkhi/work/project/pycharm_transfer/hn-pns-rnn/tensorflow/model/libri_build_model2_n5000_layer6_l30_r30,target=/models/hn_nps_rnn_feed \
  -e MODEL_NAME=hn_nps_rnn_feed -t tensorflow/serving &