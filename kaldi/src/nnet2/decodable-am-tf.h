// nnet2/decodable-am-tf.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)

// Original Code
// nnet2/decodatble-am.h
// Modified by Minkyu Lim

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_NNET2_DECODABLE_AM_NNET_H_
#define KALDI_NNET2_DECODABLE_AM_NNET_H_

#include <vector>
#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "nnet2/am-nnet.h"
#include "nnet2/tf-compute.h"


namespace kaldi {
namespace nnet2 {

/// DecodableAmNnet is a decodable object that decodes
/// with a neural net acoustic model of type AmNnet.

class DecodableAmTf: public DecodableInterface {
 public:
  DecodableAmTf(const TransitionModel &trans_model,
                  const AmNnet &am_nnet,
                  const CuMatrixBase<BaseFloat> &feats,
                  bool pad_input = true, // if !pad_input, the NumIndices()
                                         // will be < feats.NumRows().
                  BaseFloat prob_scale = 1.0,
                  const int32 left_context = 7,
                  const int32 right_context = 7,
                  const std::string &tfserving_ip_adress = "http://localhost:8501/v1/models/hn_nps_rnn_feed:predict"):
      trans_model_(trans_model) {
    // Note: we could make this more memory-efficient by doing the
    // computation in smaller chunks than the whole utterance, and not
    // storing the whole thing.  We'll leave this for later.
    int32 num_rows = feats.NumRows() -
       (pad_input ? 0 : left_context +
                         right_context);
    if (num_rows <= 0) {
      KALDI_WARN << "Input with " << feats.NumRows()  << " rows will produce "
                 << "empty output.";
      return;
    }
    CuMatrix<BaseFloat> log_probs(num_rows, trans_model.NumPdfs());
    // the following function is declared in tf-compute.h
    TfComputation(am_nnet.GetNnet(), feats, pad_input, &log_probs, left_context, right_context, tfserving_ip_adress);
    log_probs.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    log_probs.ApplyLog();
    CuVector<BaseFloat> priors(am_nnet.Priors());
    KALDI_ASSERT(priors.Dim() == trans_model.NumPdfs() &&
                 "Priors in neural network not set up.");
    priors.ApplyLog();
    // subtract log-prior (divide by prior)
    log_probs.AddVecToRows(-1.0, priors);
    // apply probability scale.
    log_probs.Scale(prob_scale);
    // Transfer the log-probs to the CPU for faster access by the
    // decoding process.
    log_probs_.Swap(&log_probs);
  }

  // Note, frames are numbered from zero.  But transition_id is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 transition_id) {
    return log_probs_(frame,
                      trans_model_.TransitionIdToPdf(transition_id));
  }

  virtual int32 NumFramesReady() const { return log_probs_.NumRows(); }

  // Indices are one-based!  This is for compatibility with OpenFst.
  virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame) const {
    KALDI_ASSERT(frame < NumFramesReady());
    return (frame == NumFramesReady() - 1);
  }

 protected:
  const TransitionModel &trans_model_;
  Matrix<BaseFloat> log_probs_; // actually not really probabilities, since we divide
  // by the prior -> they won't sum to one.

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmTf);
};





} // namespace nnet2
} // namespace kaldi

#endif  // KALDI_NNET2_DECODABLE_AM_NNET_H_
