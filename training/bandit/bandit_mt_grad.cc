#include "bandit_mt.h"

using namespace std;
namespace po = boost::program_options;

bool invert_score;
boost::shared_ptr<MT19937> rng; //random seed ptr


bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description ini("Configuration options");
  ini.add_options()
        ("weights,w",po::value<string>(),"[REQD] Input feature weights file")
       	("features,f",po::value<string>(),"Features for feature selection")
	 ("input,i",po::value<string>(),"[REQD] Input source file for training")
        ("gradient_dump,d", po::value<string>(), "[REQD] path where to store weights, default: this dir")
	("passes,p", po::value<int>()->default_value(100), "Number of passes through the training data")
        ("reference,r",po::value<vector<string> >(), "[REQD] Reference translation(s) (tokenized text file)")
        ("mt_metric,m",po::value<string>()->default_value("ibm_bleu"), "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")
        ("objective",po::value<string>()->default_value("bayes"), "Objective for bandit optimization (bayes, duel, crossentropy")
        ("sample_size,k",po::value<int>()->default_value(1),"Number of samples for each sentence")
        ("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
        ("sample_from", po::value<string>()->default_value("hg"), "Sample from full hypergraph or k-best lists (hg, kbest)")
        ("decoder_config",po::value<string>(),"Decoder configuration file")
        ("banditverbose,v", "verbose output, for debugging")
        ("viterbi, vi", "report viterbi translations") ;
  po::options_description cl("Command line options");
  cl.add_options()
        ("config,c", po::value<string>(), "Bandit configuration file")
        ("help,h", "Print this help message and exit");
  cl.add(ini);

  po::store(parse_command_line(argc, argv, cl), *conf);
  if (conf->count("config")) {
    ifstream ini_f((*conf)["config"].as<string>().c_str());
    cerr << "reading configuration from bandit config file" << endl;
    po::store(po::parse_config_file(ini_f, ini), *conf);
  }
  po::notify(*conf);
  if (!conf->count("decoder_config")) {
    cerr << "decoder config missing" << endl;
    cerr << cl << endl;
    return false;
  }

  if (conf->count("help")) {
    cerr << "Bandit help:" << endl;
    cerr << cl << endl;
    return false;
  }

  if (!conf->count("gradient_dump")) {
    cerr << "gradient dump is missing" << endl;
    cerr << cl << endl;
    return false;
  }


  if (!conf->count("weights")) {
    cerr << "initial weights are missing!" << endl;
    cerr << cl << endl;
    return false;
  }

  if (!conf->count("features")) {
    cerr << "features are missing!" << endl;
    cerr << cl << endl;
    return false;
  }


 
  if (!conf->count("input")) {
    cerr << "input is missing!" << endl;
    cerr << cl << endl;
    return false;
  } 

  if (!conf->count("reference")) {
    cerr << "reference is missing!" << endl;
    cerr << cl << endl;
    return false;
  }
 
 if (!conf->count("sample_from")) {
    cerr << "sample from is missing!" << endl;
    cerr << cl << endl;
    return false;
  }
  else{
    return true;
  }
}

void ReadTrainingCorpus(const string& fname, vector<string>* c) {
  ReadFile rf(fname);
  istream& in = *rf.stream();
  string line;
  while(in) {
    getline(in, line);
    if (!in) break;
    c->push_back(line);
  }
}

void BanditObserver::UpdateOracles(int sent_id, Hypergraph& hg){  

    //sample k hypotheses
    SufficientStats sstats;
    vector<HypergraphSampler::Hypothesis> hg_samples;
    vector<HypergraphSampler::Hypothesis> hg_samples2;
    hg_samples.clear();
    HypergraphSampler::sample_hypotheses(hg, sample_size, &*rng, &hg_samples);
    samples.clear();

    for (int i=0; i<sample_size; i++){
        ds[sent_id] -> Evaluate(hg_samples[i].words, &sstats);
        float sentscore = metric.ComputeScore(sstats);
        HypothesisInfo sampled = MakeHypothesisInfo2(hg_samples[i].fmap, sentscore);
        samples.push_back(sampled);
    }  
   // sample = samples[0];

    //compute expected features
    const prob_t z = InsideOutside<prob_t, EdgeProb, SparseVector<prob_t>, EdgeFeaturesAndProbWeightFunction>(hg, &expected_features_log);    
    //cerr << "z " << z << endl;
    expected_features_log /= z;

    //move expected features from log to normal space
    for (FastSparseVector<prob_t>::iterator it = expected_features_log.begin(); it != expected_features_log.end(); ++it) {
        const pair<unsigned, prob_t>& gi = *it;
        if (std::isfinite(gi.second.as_float())){
            expected_features.set_value(gi.first, gi.second.as_float());
        }
        else{
            cerr << "as float -> inf/nan :" << gi.second.as_float() << endl; //TODO: what to do then?
        }
    }
   
    inside = Inside<prob_t, EdgeProb>(hg);
    numberOfPaths = hg.NumberOfPaths();
 
    //if pairwise or duel: sample and compute feature expectations twice!
    if (objective == "pairwise" or objective == "xpairwise"){
        //sample also from p(y|x,-w)
        vector<double> new_weights = feature_weights;

        for (int i = 0; i < new_weights.size(); i++){
            new_weights[i] *= -1;
        }
        //cerr << "new weights " << Weights::GetString(new_weights) << endl;
    
        hg.Reweight(new_weights);

        //sample another k hypotheses

        SufficientStats sstats2;
        hg_samples2.clear();
        HypergraphSampler::sample_hypotheses(hg, sample_size, &*rng, &hg_samples2);
        samples2.clear();

        for (int i=0; i<sample_size; i++){
            ds[sent_id] -> Evaluate(hg_samples2[i].words, &sstats2);
            float sentscore2 = metric.ComputeScore(sstats2);
            HypothesisInfo sampled2 = MakeHypothesisInfo2(hg_samples2[i].fmap, sentscore2);
            samples2.push_back(sampled2);
        }

        //compute expected features
        const prob_t z2 = InsideOutside<prob_t, EdgeProb, SparseVector<prob_t>, EdgeFeaturesAndProbWeightFunction>(hg, &expected_features_log2);
      //  cerr << "z2 " << z2 << endl;
        expected_features_log2 /= z2;

        //move expected features from log to normal space
        for (FastSparseVector<prob_t>::iterator it2 = expected_features_log2.begin(); it2 != expected_features_log2.end(); ++it2) {
            const pair<unsigned, prob_t>& gi2 = *it2;
            if (std::isfinite(gi2.second.as_float())){
                expected_features2.set_value(gi2.first, gi2.second.as_float());
            }
            else{
                cerr << "as float -> inf/nan :" << gi2.second.as_float() << endl; //TODO: what to do then?
            }
        }
        inside2 = Inside<prob_t, EdgeProb>(hg);


    }

    else if (objective == "duel"){
    //TODO
    }

    //cerr << expected_features << endl;

    if (viterbi){
        vector<WordID> cur_prediction;
        ViterbiESentence(hg, &cur_prediction);
        SufficientStats sstats_best;
        ds[sent_id]->Evaluate(cur_prediction, &sstats_best);
        float sentscore_best = metric.ComputeScore(sstats_best);
        best = MakeHypothesisInfo(ViterbiFeatures(hg), sentscore_best);
        cerr << "\tsample: " << TD::GetString(hg_samples[0].words) << endl;
        if (sample_size > 1){
            cerr << "\tsample: " << TD::GetString(hg_samples[1].words) << endl;
        }
        if (objective == "pairwise"){
            cerr << "\tsample2: " << TD::GetString(hg_samples2[0].words) << endl;
        }
        //cerr << "\tsampled BLEU: " << sentscore << endl;
        cerr << "\tviterbi: " << TD::GetString(cur_prediction) << endl;
        cerr << "\tviterbi BLEU:" << sentscore_best << endl;
    }
    
    //for inspecting distribution
    //vector<prob_t> node_probs;
    //Inside<prob_t, EdgeProb>(hg, &node_probs, EdgeProb()); 
    //prob_t entropy = Entropy(node_probs);
    //cerr << "\tnode distribution entropy: " << entropy << endl;
}

prob_t BanditObserver::Entropy(vector<prob_t> node_probs){
    //H = - sum_i(p_i*log_2(p_i))
    prob_t h = 0;
    for (int i = 0; i < node_probs.size(); i++){
    //    cerr << node_probs[i] << endl;
        prob_t l = node_probs[i].e();
        prob_t ll = node_probs[i];
        prob_t lll = l*ll;
        h += lll;
    }
    //cerr << "h " << h << endl;
    return -h;
}

boost::shared_ptr<HypothesisInfo> BanditObserver::MakeHypothesisInfo(const SparseVector<double>& feats, const double metric) {
    boost::shared_ptr<HypothesisInfo> h(new HypothesisInfo);
    h->features = feats;
    h->mt_metric_score = metric;
    return h;
}

HypothesisInfo BanditObserver::MakeHypothesisInfo2(const SparseVector<double>& feats, const double metric) {
    HypothesisInfo h;
    h.features = feats;
    h.mt_metric_score = metric;
    return h;
}

void BanditObserver::NotifyTranslationForest(const SentenceMetadata& smeta, Hypergraph* hg){
    BanditObserver::UpdateOracles(smeta.GetSentenceID(), *hg);
}


double MaxFromSparseVector(SparseVector<double> sparse_vector){
    double max = -INFINITY;
    for (SparseVector<double>::iterator it = sparse_vector.begin(); it != sparse_vector.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        if (gi.second > max){
            max = gi.second;
        }
    }
    return max;
}

SparseVector<double> SparseSqrt(SparseVector<double> sparse_vector){
    SparseVector<double> sqrt_vector;
    for (SparseVector<double>::iterator it = sparse_vector.begin(); it != sparse_vector.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        sqrt_vector.set_value(gi.first, sqrt(gi.second));
    }
    return sqrt_vector;
}

SparseVector<double> ElementwiseProduct(SparseVector<double> v_1, SparseVector<double> v_2){
    SparseVector<double> product;
    for (SparseVector<double>::iterator it = v_1.begin(); it != v_1.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        product.set_value(gi.first, gi.second*v_2[gi.first]);
    }
    return product;
}

SparseVector<double> ElementwiseDivision(SparseVector<double> v_1, SparseVector<double> v_2){
    SparseVector<double> quotient;
    for (SparseVector<double>::iterator it = v_1.begin(); it != v_1.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        quotient.set_value(gi.first, gi.second/v_2[gi.first]);
    }
    return quotient;
}

SparseVector<double> SetToValue(SparseVector<double> v, double value){
    for (SparseVector<double>::iterator it = v.begin(); it != v.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        v.set_value(gi.first, value);
    }
    return v;
}

SparseVector<double> RMS(SparseVector<double> accumulated, double epsilon){
    for (SparseVector<double>::iterator it = accumulated.begin(); it != accumulated.end(); ++it) {
        const pair<unsigned, double>& gi = *it;
        accumulated.set_value(gi.first, gi.second+epsilon);
    }
    accumulated = SparseSqrt(accumulated);
    return accumulated;
} 

SparseVector<double> RMS(SparseVector<double> accumulated, SparseVector<double> epsilon_vector){
    accumulated += epsilon_vector;
    accumulated = SparseSqrt(accumulated);
    return accumulated;
}

//features must be a vector of ones
SparseVector<double> SelectFeatures(SparseVector<double> weights, SparseVector<double> features){
	//cerr << "#feat " << weights.size() << endl;
	return ElementwiseProduct(features, weights);
}

bool compareGreater(int i,int j) { return (i>j); }

SparseVector<double> clipWeights(SparseVector<double> weights, int k){
   //clip weights to k largest weights
   if (k!=-1){ //else do nothing
   	cerr << weights.size() << " features in weights" << endl;
	vector<double> dense_weights;
	weights.init_vector(&dense_weights);
	std::sort(dense_weights.begin(), dense_weights.end(), compareGreater);         //sort descending
	dense_weights.resize(k); //clip at k   
     	Weights::InitSparseVector(dense_weights, &weights); 
   } 
   return weights;
}

//gradient: feedback * (y_tilde - avg_x) (bayes)
vector<SparseVector<double>> ComputeGradients(SparseVector<double> sparse_weights, SparseVector<double> avg_x, prob_t z, vector<HypothesisInfo> y_tildes, double &feedback, string objective, double clip, SparseVector<double> features){
    vector<SparseVector<double>> gradients;
    feedback = 0.0;
    //cerr << "Sample BLEUs:" << endl;
    if (objective == "bayes"){
	double z_norm = log(z);
        for (int i=0; i<y_tildes.size(); i++){
      //      cerr << y_tildes[i].mt_metric_score << endl;
            feedback += 1-y_tildes[i].mt_metric_score;
            double score = y_tildes[i].features.dot(sparse_weights); //unnormalized log prob
            score = score-z_norm; // compute probability from score: exp(score)/Z_w: inside score of goal node
            score = exp(score);
	    SparseVector<double> gradient = y_tildes[i].features;
            gradient -= avg_x;
            gradient *= 1-y_tildes[i].mt_metric_score; //feedback
            gradient *= score; //just for true gradient
	    gradient = SelectFeatures(gradient, features);
            gradients.push_back(gradient);
        }
    }
    else if (objective == "crossentropy"){
        double z_norm = log(z);
        for (int i=0; i<y_tildes.size(); i++){
        //    cerr << y_tildes[i].mt_metric_score << endl;
            feedback += 1-y_tildes[i].mt_metric_score;
            double score = y_tildes[i].features.dot(sparse_weights); //unnormalized log prob
            score = score-z_norm; // compute probability from score: exp(score)/Z_w: inside score of goal node
            score = exp(score);
            SparseVector<double> gradient = y_tildes[i].features;
            gradient *= -1;
            gradient += avg_x;
            double gain = y_tildes[i].mt_metric_score; //feedback
            gradient *= gain; //division by score is left out for true gradient
	    gradient = SelectFeatures(gradient, features);
            gradients.push_back(gradient);
        }
    }
    feedback /= y_tildes.size();
    return gradients;
}


//compute gradients for objectives where more than one feature vector is involved
vector<SparseVector<double>> ComputeGradientsPairwise(SparseVector<double> sparse_weights, SparseVector<double> sparse_weights2, SparseVector<double> avg_1,SparseVector<double> avg_2, prob_t z1, prob_t z2, vector<HypothesisInfo> y_tildes1, vector<HypothesisInfo> y_tildes2, double &feedback, string objective,  SparseVector<double> features){
    vector<SparseVector<double>> gradients;
    feedback = 0.0;
    if (objective == "pairwise"){
	double z_norm1 = log(z1);
        double z_norm2 = log(z2);

        for (int i=0; i<y_tildes1.size(); i++){ //compare samples one by one
            double bleu_1 = y_tildes1[i].mt_metric_score;
            double bleu_2 = y_tildes2[i].mt_metric_score;
            
            if (bleu_2 > bleu_1){
                feedback += 1;
		double score1 = y_tildes1[i].features.dot(sparse_weights); //unnormalized log prob
                double score2 = y_tildes2[i].features.dot(sparse_weights); //unnormalized log prob
                score1 = score1-z_norm1; // compute probability from score: exp(score)/Z_w: inside score of goal node
                score1 = exp(score1);
                score2 = score2-z_norm2;
                score2 = exp(score2);
                double score = score1*score2;
                SparseVector<double> x_diff = y_tildes1[i].features - y_tildes2[i].features;
                SparseVector<double> avg_diff = avg_1 - avg_2;
                x_diff -= avg_diff; //sampled minus expected
		x_diff *= score; //only for true gradient
		x_diff = SelectFeatures(x_diff, features);
		cerr << "pw #feat " << x_diff.size() << endl;
                gradients.push_back(x_diff);
            }                
//            else{
//                gradients.push_back(SetToValue(avg_1,0)); //no change
 //           }
            
            //cerr << "bleu_1 " << bleu_1 << endl;
            //cerr << "bleu_2 " << bleu_2 << endl;
            
        }
    }
    
    else if (objective == "xpairwise"){
        double z_norm1 = log(z1);
        double z_norm2 = log(z2);
        for (int i=0; i<y_tildes1.size(); i++){ //compare samples one by one
            double bleu_1 = y_tildes1[i].mt_metric_score;
            double bleu_2 = y_tildes2[i].mt_metric_score;
            
            if (bleu_1 > bleu_2){ //if sampled pair is correctly ordered (gain=1), else gain=0
                feedback += 1;
                double score1 = y_tildes1[i].features.dot(sparse_weights); //unnormalized log prob
                double score2 = y_tildes2[i].features.dot(sparse_weights); //unnormalized log prob
                score1 = score1-z_norm1; // compute probability from score: exp(score)/Z_w: inside score of goal node
                score1 = exp(score1);
                score2 = score2-z_norm2;
                score2 = exp(score2);
                double score = score1*score2;
                double gain = 1; //bleu_1 - bleu_2;
              //  cerr << "scores: " << score1 << " " << score2 << " " << score << endl;
              //  cerr << "gain: " << gain << endl;
                SparseVector<double> x_diff = y_tildes1[i].features - y_tildes2[i].features;
                SparseVector<double> avg_diff = avg_1 - avg_2;
                avg_diff -= x_diff; //expected minus sampled
                if (score==0){
			score += pow(10,-20); //prevent division by 0	
		}
		avg_diff *= gain; //no division by score for true gradient
		avg_diff = SelectFeatures(avg_diff, features);
                gradients.push_back(avg_diff);
            }
   //         else{
    //            gradients.push_back(SetToValue(avg_1,0)); //no change
    //        }

           // cerr << "bleu_1 " << bleu_1 << endl;
           // cerr << "bleu_2 " << bleu_2 << endl;

        }
    }

    else if (objective == "duel"){
        //TODO
    }
        
    feedback /= y_tildes1.size(); //not the actual feedback for the update, but average feedback for sampled pairs
    return gradients;
}

int main(int argc, char** argv) {

    register_feature_functions();

    po::variables_map conf;
    if (!InitCommandLine(argc, argv, &conf)) return 1;

    if (conf.count("random_seed")){
        rng.reset(new MT19937(conf["random_seed"].as<uint32_t>()));
    
    }
    else{
        rng.reset(new MT19937);
    }    

    bool silent = false;
    if (!conf.count("banditverbose")){
        SetSilent(true);
        silent = true;  // turn off verbose decoder output
    }
    else{
        SetSilent(true); //TODO add cli param for cdec verbosity
    }

    bool viterbi = false;
    if (conf.count("viterbi")){
        viterbi = true;
    }

    const string dump_dir = conf["gradient_dump"].as<string>();
    cerr << "Writing full gradient to " << dump_dir << endl;


    vector<string> corpus;
    ReadTrainingCorpus(conf["input"].as<string>(), &corpus);
    
    const string metric_name = conf["mt_metric"].as<string>(); 
    ScoreType type = ScoreTypeFromString(metric_name);
    if (type == TER) {
        invert_score = true; //TODO: might switch around since we're training on 1-BLEU
    } else {
        invert_score = false; //TODO invert_score is not used yet
    }
    EvaluationMetric* metric = EvaluationMetric::Instance(metric_name);

    const string objective = conf["objective"].as<string>();
    cerr << "Learning with objective " << objective << endl;

    const string sample_from = conf["sample_from"].as<string>();
    cerr << "Sampling from " << sample_from << endl;

    const int sample_size = conf["sample_size"].as<int>();
    cerr << "Sampling " << sample_size << " translations for each sentence" << endl;

    bool full_hg = false;    
    if (sample_from.compare("hg")){
        full_hg = true;
    }

    double clip = 0.01;
    if (conf.count("clip")){
	clip = conf["clip"].as<double>();
	cerr << "clipping sample probability (crossentropy) to " << clip << endl;
    }

    DocumentScorer ds(metric, conf["reference"].as<vector<string>>());
    cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;
    if (ds.size() != corpus.size()) {
        cerr << "Warning: mismatched number of references (" << ds.size() << ") and sources (" << corpus.size() << ")\n";
        //return 1; //don't return: shifted seg ids possible
    }

    ReadFile ini_rf(conf["decoder_config"].as<string>());
    Decoder decoder(ini_rf.stream());
    cerr << "Read config from file " << conf["decoder_config"].as<string>() << endl;
    boost::filesystem3::path q(conf["decoder_config"].as<string>());
    string config_string = q.filename().string();

    //initialize weights
    vector<double>& decoder_weights = decoder.CurrentWeightVector();
    SparseVector<double> sparse_weights; 
    string weight_string;
    Weights::InitFromFile(conf["weights"].as<string>(), &decoder_weights);
    cerr << "Loaded weights from " << conf["weights"].as<string>() << endl;
    boost::filesystem3::path p(conf["weights"].as<string>());
    weight_string = p.filename().string();
    Weights::InitSparseVector(decoder_weights, &sparse_weights);
  
    //read in features to be selected
    vector<double> features_dense; 
    SparseVector<double> features;
    Weights::InitFromFile(conf["features"].as<string>(), &features_dense);
    cerr << "Loaded weights from " << conf["features"].as<string>() << endl;
    Weights::InitSparseVector(features_dense, &features);
    features = SetToValue(features, 1.0);

    //initialize bandit parameters
    SparseVector<double> exp_feats;
    
    assert(corpus.size() > 0);

    if (objective=="duel"){ //has to be true for comparing viterbi scores
        viterbi = true;
    }

    BanditObserver observer(conf["sample_size"].as<int>(),
                            ds, //document scorer
                            *metric,
                            exp_feats,
                            decoder_weights,
                            viterbi,
                            objective
                            );
    int cur_sent = 0;
    int line_count = 0;
    int cur_pass = 1; 
    int max_passes = conf["passes"].as<int>();
    SparseVector<double> truegrad;   
    
    while(line_count < max_passes*corpus.size()) {
	cerr << "sent no " << line_count << " / " << corpus.size() << endl;

        decoder.SetId(cur_sent);
        sparse_weights.init_vector(&decoder_weights);        

        if (!silent){
            cerr << "cur sent " << corpus[cur_sent] << endl;
        }
       
        decoder.Decode(corpus[cur_sent], &observer);

        double feedback;
        SparseVector<double> x;
        SparseVector<double> gradient;
        SparseVector<double> avg_x;
    
        //sample and receive feedback
        if (objective == "bayes" or objective == "crossentropy"){ 
            const vector<HypothesisInfo> y_tildes = observer.GetSamples();//sample y_tilde = sample hypothesis
            prob_t z = observer.GetZ();
            double numberOfPaths = observer.GetNumberOfPaths();
            avg_x = observer.GetFeatureExpectation();//compute avg_x = feature expectation via inside-outside
            vector<SparseVector<double>> gradients = ComputeGradients(sparse_weights, avg_x, z, y_tildes, feedback, objective, clip, features);
            for (int i=0; i<gradients.size(); i++){ 
                gradient += gradients[i]; //sum gradients from all samples
            }        
        }

        else if (objective == "pairwise" or objective == "xpairwise"){
            const vector<HypothesisInfo> y_tildes1 = observer.GetSamples();//sample y_tilde = sample hypothesis
            SparseVector<double> avg_1 = observer.GetFeatureExpectation();//compute avg_x = feature expectation via inside-outside
            prob_t z1 = observer.GetZ();
            double numberOfPaths = observer.GetNumberOfPaths();

            const vector<HypothesisInfo> y_tildes2 = observer.GetSamples2();//sample y_tilde = sample hypothesis
            SparseVector<double> avg_2 = observer.GetFeatureExpectation2();//compute avg_x = feature expectation via inside-outside
            prob_t z2 = observer.GetZ2();    
            SparseVector<double> sparse_weights2 = sparse_weights;
            sparse_weights2 *= -1;
        
            vector<SparseVector<double>> gradients = ComputeGradientsPairwise(sparse_weights, sparse_weights2, avg_1, avg_2, z1, z2, y_tildes1, y_tildes2, feedback, objective, features);
            for (int i=0; i<gradients.size(); i++){
                gradient += gradients[i]; //sum gradients from all samples
            }

        }

        else{
            cerr << "no valid objective given (bayes, crossentropy, pairwise, xpairwise)" << endl;
        }

    //clip gradients to 10k largest features
	//truegrad += clipWeights(gradient, 10000);    
	truegrad += gradient;

    cerr << "#feat in grad: " << truegrad.size() << endl;

    if (!silent) {
            cerr << "loss: " << feedback << endl;
            cerr << "x: ";
            print(cerr, x, "=", ",");
            cerr << endl;

            cerr << "avg_x: ";
            print(cerr, avg_x, "=", ",");
            cerr << endl;
         }
  

        ++line_count;
        ++cur_sent;
        
        if (corpus.size() == cur_sent) { //finished a pass
            cerr << "DONE" << endl;
    	}
	
    }
    //truegrad /= corpus.size();  

    ostringstream os;
    os << dump_dir << ".txt";
    vector<double> grad_vector;
    truegrad.init_vector(grad_vector);
    Weights::WriteToFile(os.str(), grad_vector, true);
    cerr << "Wrote gradient sum to " << os.str() << endl;
     
    //double gradnorm = truegrad.pnorm(2);
    //cerr << "Norm of true gradient: " << gradnorm << endl;
    
}
