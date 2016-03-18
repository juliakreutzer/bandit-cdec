#ifndef BANDIT_MT_H_
#define BANDIT_MT_H_

#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <sstream>
#include <time.h> 

//sys
#include <sys/stat.h>
#include <sys/types.h>

//boost libraries
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

//cdec libraries
#include "config.h"
#include "inside_outside.h"
#include "hg_sampler.h"
#include "scorer.h"
#include "viterbi.h"
#include "verbose.h"
#include "hg.h"
#include "prob.h"
#include "kbest.h"
#include "fdict.h"
#include "weights.h"
#include "sparse_vector.h"
#include "sampler.h"
#include "decoder.h"
#include "ff_register.h"
#include "timing_stats.h"
#include "filelib.h"
#include "sentence_metadata.h"
#include "ns.h"
#include "ns_docscorer.h"
#include "feature_vector.h"

using namespace std;
namespace po = boost::program_options;


struct HypothesisInfo{
    SparseVector<double> features;
    double mt_metric_score;
};

struct BanditObserver : public DecoderObserver {
   // const int kbest_size;
  //  const DocumentScorer& ds;
  //  const EvaluationMetric& metric;
  //  //boost::shared_ptr<vector<HypothesisInfo>> kbest_hypos;

   // boost::shared_ptr<HypothesisInfo> sample;
   // SparseVector<prob_t> expected_features_log;
   // SparseVector<double> expected_features;
   // boost::shared_ptr<HypothesisInfo> cur_ref;
   // vector<weight_t>& feature_weights;


    BanditObserver(const int k,
                   const DocumentScorer& d,
                   const EvaluationMetric& m,
                   SparseVector<double> exp_feats,
                   vector<double>& feat_weights,
                   bool viterbi,
                   string objective
                   ) : sample_size(k), ds(d), metric(m), expected_features(exp_feats), feature_weights(feat_weights), viterbi(viterbi), objective(objective) {}
    const int sample_size;
    const DocumentScorer& ds;
    const EvaluationMetric& metric;
    //boost::shared_ptr<vector<HypothesisInfo>> kbest_hypos;
    boost::shared_ptr<HypothesisInfo> best;
    boost::shared_ptr<HypothesisInfo> best2;
    boost::shared_ptr<HypothesisInfo> sample;
    vector<HypothesisInfo> samples;
    vector<HypothesisInfo> samples2;
    boost::shared_ptr<HypothesisInfo> sample2;
    SparseVector<prob_t> expected_features_log;
    SparseVector<double> expected_features;
    SparseVector<prob_t> expected_features_log2;
    SparseVector<double> expected_features2;
    boost::shared_ptr<HypothesisInfo> cur_ref;
    vector<double>& feature_weights;
    bool viterbi;
    string objective;
    prob_t inside;
    prob_t inside2;
    double numberOfPaths;

    const HypothesisInfo& GetCurrentReference() const {
        return *cur_ref;
    }

    //const vector<HypothesisInfo>& GetKBest() const {
    //    return *kbest_hypos;
    //}

    const HypothesisInfo& GetBest() const {
        return *best;
    }

    const HypothesisInfo& GetSample() const {
        return *sample;
    }
    
    const HypothesisInfo& GetSample2() const {
        return *sample2;
    }

    const vector<HypothesisInfo> GetSamples() const {
        return samples;
    }

    const vector<HypothesisInfo> GetSamples2() const {
        return samples2;
    }

    const double GetNumberOfPaths() const {
        return numberOfPaths;
    }

    const prob_t GetZ() const {
        return inside;
    }

    const prob_t GetZ2() const {
        return inside2;
    }

    const SparseVector<double> GetFeatureExpectation(){
        return expected_features;
    }

    const SparseVector<prob_t> GetFeatureExpectationLog(){
        return expected_features_log;
    }

    const SparseVector<double> GetFeatureExpectation2(){
        return expected_features2;
    }

    const SparseVector<prob_t> GetFeatureExpectationLog2(){
        return expected_features_log2;
    }

    
    virtual void UpdateOracles(int sent_id, Hypergraph& hg);

    virtual void NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg);

    virtual boost::shared_ptr<HypothesisInfo> MakeHypothesisInfo(const SparseVector<double>& feats, const double metric);

    virtual HypothesisInfo MakeHypothesisInfo2(const SparseVector<double>& feats, const double metric);

    virtual prob_t Entropy(vector<prob_t> node_probs); 
};

#endif
