// nnet2/tf-compute.cc

// Copyright 2012   Johns Hopkins University (author: Daniel Povey)
// Copyright 2015   David Snyder

// Original Code
// nnet2/nnet-compute.cc
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

#include<string>
#include<cstring>
#include<curl/curl.h>

#include "nnet2/tf-compute.h"
#include "hmm/posterior.h"

#include "cnpy.h"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"


namespace kaldi {
namespace nnet2 {


/*
  This class does the forward and possibly backward computation for (typically)
  a whole utterance of contiguous features.  You'll instantiate one of
  these classes each time you want to do this computation.
*/
class TfComputer {
 public:
  /* Initializer.  If pad == true, pad input with nnet.LeftContext() frames on
     the left and nnet.RightContext() frames on the right (duplicate the first
     and last frames.) */
  TfComputer(const Nnet &nnet,
               const CuMatrixBase<BaseFloat> &input_feats,
               bool pad,
               const int32 left_context,
               const int32 right_context,
               const std::string &tfserving_ip_adress);

  /// The forward-through-the-layers part of the computation.
  void Propagate();


  std::string Serialize();

  CuMatrixBase<BaseFloat> &GetOutput() { return forward_data_.back(); }

 private:
  int32 left_context_;
  int32 right_context_;
  const Nnet &nnet_;
  std::vector<CuMatrix<BaseFloat> > forward_data_;
  std::string tfserving_ip_adress_;
};

TfComputer::TfComputer(const Nnet &nnet,
                       const CuMatrixBase<BaseFloat> &input_feats,
                       bool pad,
                       const int32 lcontext,
                       const int32 rcontext,
                       const std::string &tfserving_ip_adress):
    nnet_(nnet) {
  int32 dim = input_feats.NumCols();
  if (dim != nnet.InputDim()) {
    KALDI_ERR << "Feature dimension is " << dim << " but network expects "
              << nnet.InputDim();
  }
  forward_data_.resize(2);

  left_context_ = lcontext;
  right_context_ = rcontext;
  tfserving_ip_adress_ = tfserving_ip_adress;

  int32 num_rows = left_context_ + input_feats.NumRows() + right_context_;

  CuMatrix<BaseFloat> &input(forward_data_[0]);
  input.Resize(num_rows, dim);
  input.Range(left_context_, input_feats.NumRows(),
              0, dim).CopyFromMat(input_feats);
  for (int32 i = 0; i < left_context_; i++)
    input.Row(i).SetZero();
  int32 last_row = input_feats.NumRows() - 1;
  for (int32 i = 0; i < right_context_; i++)
    input.Row(num_rows - i - 1).SetZero();
}

using namespace rapidjson;

struct string {
  char *ptr;
  size_t len;
};

void init_string(struct string *s) {
  s->len = 0;
  s->ptr = (char*)malloc(s->len+1);
  if (s->ptr == NULL) {
    fprintf(stderr, "malloc() failed\n");
    exit(EXIT_FAILURE);
  }
  s->ptr[0] = '\0';
}

size_t writefunc(void *ptr, size_t size, size_t nmemb, struct string *s)
{
  size_t new_len = s->len + size*nmemb;
  s->ptr = (char*)realloc(s->ptr, new_len+1);
  if (s->ptr == NULL) {
    fprintf(stderr, "realloc() failed\n");
    exit(EXIT_FAILURE);
  }
  memcpy(s->ptr+s->len, ptr, size*nmemb);
  s->ptr[new_len] = '\0';
  s->len = new_len;

  return size*nmemb;
}


std::string TfComputer::Serialize() {
  std::string str = "{\"instances\": [";
   for(int i = left_context_ ; i < forward_data_[0].NumRows()-right_context_ ; i++){
    str.append("[");
     for(int j = 0-left_context_ ; j <= right_context_ ; j++){
      for(int k = 0 ; k < forward_data_[0].NumCols() ; k++){
        str.append(std::to_string(forward_data_[0].Row(i+j).Data()[k]));
         if(j<right_context_ || k<forward_data_[0].NumCols()-1)
          str.append(",");
      }
    }
    str.append("]");
     if(i<forward_data_[0].NumRows()-right_context_-1) str.append(",");
  }
  str.append("]}");

  return str;
}

/// This is the forward part of the computation.
void TfComputer::Propagate() {

  std::string str = Serialize();
  // std::cout << str;

	CURL *curl;
	CURLcode res;

	curl_global_init(CURL_GLOBAL_ALL);

	curl = curl_easy_init();
	if(curl) {
//		curl_easy_setopt(curl, CURLOPT_URL, "http://localhost:8501/v1/models/hn_nps_rnn_feed:predict");
		char addr[100];
		std::size_t addr_len = tfserving_ip_adress_.copy(addr, tfserving_ip_adress_.length(), 0);
		addr[addr_len] = '\0';
		curl_easy_setopt(curl, CURLOPT_URL, addr);


		char * tab2 = new char [str.length()+1];
		strcpy(tab2, str.c_str());

		struct string s;
		init_string(&s);

		curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10);

		curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writefunc);
		curl_easy_setopt(curl, CURLOPT_WRITEDATA, &s);

		curl_easy_setopt(curl, CURLOPT_POSTFIELDS, tab2);

		curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
		// curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);

		res = curl_easy_perform(curl);

		// std::cout << "RESULT!!!!!!!!!!!!!!!\n" << s.ptr << s.ptr;
		curl_easy_cleanup(curl);


		Document d;
		d.Parse(s.ptr);
    delete tab2;
    free(s.ptr);

		Value& sv = d["predictions"];
		// double a = sv[0][0].GetDouble();
		// std::cout << "Size: " << sv[0].Size() << "\n";


    CuMatrix<BaseFloat> &output(forward_data_[1]);
    output.Resize(sv.Size(), sv[0].Size());
    output.SetZero();

    for (int i = 0 ; i < sv.Size() ; i++){
      for (int j = 0 ; j < sv[0].Size() ; j++){
        CuSubMatrix<BaseFloat> mat(output, i, 1, j, 1);
        mat.Add(sv[i][j].GetDouble());
      }
    }
	}
	curl_global_cleanup();
	return ;
}

void TfComputation(const Nnet &nnet,
                   const CuMatrixBase<BaseFloat> &input,  // features
                   bool pad_input,
                   CuMatrixBase<BaseFloat> *output,
                   const int32 left_context,
                   const int32 right_context,
                   const std::string &tfserving_ip_adress) {
  TfComputer tf_computer(nnet, input, pad_input, left_context, right_context, tfserving_ip_adress);
  tf_computer.Propagate();
  output->CopyFromMat(tf_computer.GetOutput());
  return;
}

} // namespace nnet2
} // namespace kaldi
