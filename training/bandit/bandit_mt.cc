#include "bandit_mt.h"

using namespace std;
namespace po = boost::program_options;

bool invert_score;
boost::shared_ptr<MT19937> rng; //random seed ptr


bool InitCommandLine(int argc, char** argv, po::variables_map* conf) {
  po::options_description ini("Configuration options");
  ini.add_options()
        ("weights,w",po::value<string>(),"[REQD] Input feature weights file")
        ("input,i",po::value<string>(),"[REQD] Input source file for training")
        ("passes,p", po::value<int>()->default_value(100), "Number of passes through the training data")
        ("reference,r",po::value<vector<string> >(), "[REQD] Reference translation(s) (tokenized text file)")
        ("mt_metric,m",po::value<string>()->default_value("ibm_bleu"), "Scoring metric (ibm_bleu, nist_bleu, koehn_bleu, ter, combi)")
        ("objective",po::value<string>()->default_value("bayes"), "Objective for bandit optimization (bayes, duel, crossentropy")
 	("feedback_type", po::value<string>()->default_value("binary"), "Feedback type for pairwise objective (binary, continuous, always)")
        ("dueldelta,dd",po::value<int>()->default_value(3),"Delta parameter for dueling bandits")
        ("learning_rate,g", po::value<double>()->default_value(3), "Learning rate = 10**(-g)")
        ("sample_size,k",po::value<int>()->default_value(1),"Number of samples for each sentence")
        ("adadelta,a", "use adaptive learning rate")
        ("normalize","normalize updates by 1/<number of paths>")
        ("regularizer", po::value<double>(),"l2-regularization parameter, lambda = 10**(-r)")
        ("report_interval,n", po::value<int>()->default_value(100), "Number of lines between reporting")
        ("weights_dump,d", po::value<string>(), "[REQD] path where to store weights, default: weights")
        ("random_seed,S", po::value<uint32_t>(), "Random seed (if not specified, /dev/random will be used)")
        ("sample_from", po::value<string>()->default_value("hg"), "Sample from full hypergraph or k-best lists (hg, kbest)")
        ("decoder_config",po::value<string>(),"Decoder configuration file")
        ("banditverbose,v", "verbose output, for debugging")
        ("viterbi, vi", "report viterbi translations")
	("clip, x", po::value<double>()->default_value(0.01), "clip sample prob to this value")
        ("decreasing, dec",po::value<string>(), "decreasing learning rate: exponential or cooling");
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


  if (!conf->count("weights")) {
    cerr << "initial weights are missing!" << endl;
    cerr << cl << endl;
    return false;
  }
 
  if (!conf->count("input")) {
    cerr << "input is missing!" << endl;
    cerr << cl << endl;
    return false;
  } 

  if (!conf->count("weights_dump")) {
    cerr << "weights dump" << endl;
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


float cooling(int t, float n0){
    float c = n0; //default, TODO tune?
    float k = 1.0; //TODO tune?
    return c/(k*t);
}

float exponential(int t, float n0){
    float k = 1.0; //TODO tune?
    return n0*exp(-k*t);
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
vector<SparseVector<double>> ComputeGradients(SparseVector<double> sparse_weights, SparseVector<double> avg_x, prob_t z, vector<HypothesisInfo> y_tildes, double &feedback, string objective, double clip){
    vector<SparseVector<double>> gradients;
    feedback = 0.0;
    cerr << "Sample BLEUs:" << endl;
    if (objective == "bayes"){
        for (int i=0; i<y_tildes.size(); i++){
            cerr << y_tildes[i].mt_metric_score << endl;
            feedback += 1-y_tildes[i].mt_metric_score;
            SparseVector<double> gradient = y_tildes[i].features;
            gradient -= avg_x;
            gradient *= 1-y_tildes[i].mt_metric_score; //feedback
            gradients.push_back(gradient);
        }
    }
    else if (objective == "crossentropy"){
        double z_norm = log(z);
        for (int i=0; i<y_tildes.size(); i++){
            cerr << y_tildes[i].mt_metric_score << endl;
            feedback += 1-y_tildes[i].mt_metric_score;
            double score = y_tildes[i].features.dot(sparse_weights); //unnormalized log prob
            score = score-z_norm; // compute probability from score: exp(score)/Z_w: inside score of goal node
            score = exp(score);
            SparseVector<double> gradient = y_tildes[i].features;
            gradient *= -1;
            gradient += avg_x;
            double gain = y_tildes[i].mt_metric_score; //feedback
	    if (score < clip){
//		cerr << "to clip " << score << endl; 
	    	score = clip;
		cerr << "clipped " << score << endl;
	    }
            cerr << "gain/p(y): " << gain << " / " << score << " = " << gain/score << endl;
            gradient *= (gain/score);
            gradients.push_back(gradient);
        }
    }
    feedback /= y_tildes.size();
    return gradients;
}


//compute gradients for objectives where more than one feature vector is involved
vector<SparseVector<double>> ComputeGradientsPairwise(SparseVector<double> sparse_weights, SparseVector<double> sparse_weights2, SparseVector<double> avg_1,SparseVector<double> avg_2, prob_t z1, prob_t z2, vector<HypothesisInfo> y_tildes1, vector<HypothesisInfo> y_tildes2, double &feedback, string objective, string feedback_type){
    vector<SparseVector<double>> gradients;
    feedback = 0.0;
    double pair_feedback = 0.0;
    if (objective == "pairwise"){
        for (int i=0; i<y_tildes1.size(); i++){ //compare samples one by one
            double bleu_1 = y_tildes1[i].mt_metric_score;
            double bleu_2 = y_tildes2[i].mt_metric_score;
            cerr << "bleu_1 " << bleu_1 << endl;
            cerr << "bleu_2 " << bleu_2 << endl;
	    if (feedback_type != "always"){
            	if (bleu_2 > bleu_1){

			if (feedback_type == "binary"){
				pair_feedback = 1;
			}
			else{
				pair_feedback = (bleu_2 - bleu_1);  //continuous feedback: difference in bleu scores if misranked
			}
			feedback += pair_feedback;
			cerr << "feedback: " << pair_feedback << endl;
 
			SparseVector<double> x_diff = y_tildes1[i].features - y_tildes2[i].features;
        	        SparseVector<double> avg_diff = avg_1 - avg_2;
                	x_diff -= avg_diff; //sampled minus expected
                	x_diff *= pair_feedback;
                	gradients.push_back(x_diff);
            	}                
            	else{
                	gradients.push_back(SetToValue(avg_1,0)); //no change
            	}
            }
	    else { //always update by difference of bleus
		pair_feedback = (bleu_2 - bleu_1); //is positive when misranked, negative when correctly ranked
		feedback += pair_feedback;
                cerr << "feedback: " << pair_feedback << endl;
		SparseVector<double> x_diff = y_tildes1[i].features - y_tildes2[i].features;
                SparseVector<double> avg_diff = avg_1 - avg_2;
                x_diff -= avg_diff; //sampled minus expected
                x_diff *= pair_feedback;
                gradients.push_back(x_diff);
	    }

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
		avg_diff *= gain/score; //factor: gain/probability where gain=1
                gradients.push_back(avg_diff);
            }
            else{
                gradients.push_back(SetToValue(avg_1,0)); //no change
            }

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

    const string weight_dir = conf["weights_dump"].as<string>(); 
    cerr << "Writing weights to " << weight_dir << endl;

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

    double lambda = 0.0;
    if (conf.count("regularizer")){
        lambda = pow(10,-conf["regularizer"].as<double>());
    }
    cerr << "lambda = " << lambda << endl;


    bool normalize = false;
    if (conf.count("normalize")){
        normalize = true;
        cerr << "normalizing weight updates" << endl;
    }

    double clip = 0.01;
    if (conf.count("clip")){
	clip = conf["clip"].as<double>();
	cerr << "clipping sample probability (crossentropy) to " << clip << endl;
    }

    const string feedback_type = conf["feedback_type"].as<string>();
    cerr << "Feedback type for pairwise: " << feedback_type << endl;

    DocumentScorer ds(metric, conf["reference"].as<vector<string>>());
    cerr << "Loaded " << ds.size() << " references for scoring with " << metric_name << endl;
    if (ds.size() != corpus.size()) {
        cerr << "Warning: mismatched number of references (" << ds.size() << ") and sources (" << corpus.size() << ")\n";
        //return 1; //don't return: shifted seg ids possible
    }


    ReadFile ini_rf(conf["decoder_config"].as<string>());
    Decoder decoder(ini_rf.stream());
    cerr << "Read config from file " << conf["decoder_config"].as<string>() << endl;
    boost::filesystem::path q(conf["decoder_config"].as<string>());
    string config_string = q.filename().string();

    //initialize weights
    vector<double>& decoder_weights = decoder.CurrentWeightVector();
    SparseVector<double> sparse_weights; 
    string weight_string;
    Weights::InitFromFile(conf["weights"].as<string>(), &decoder_weights);
    cerr << "Loaded weights from " << conf["weights"].as<string>() << endl;
    boost::filesystem::path p(conf["weights"].as<string>());
    weight_string = p.filename().string();
    Weights::InitSparseVector(decoder_weights, &sparse_weights);
  
    if (normalize){
        sparse_weights /= MaxFromSparseVector(sparse_weights);
 	cerr << "Normalized weights: ";
    	print(cerr, sparse_weights, "=", ", ");
    	cerr << endl;
    }
 
    //initialize bandit parameters
    const double learning_rate = pow(10,-conf["learning_rate"].as<double>()); 
   
    const int report_interval = conf["report_interval"].as<int>();
    
    string schedule = "constant";
    if (conf.count("decreasing")){
        cerr << "Using decreasing learning rate: " << conf["decreasing"].as<string>() << endl;
        schedule = conf["decreasing"].as<string>();
    }
    if (conf.count("adadelta")){
        cerr << "Using adaptive learning rate (adadelta)" << endl;
        schedule = "adadelta";
    }
    else{
        cerr << "Using constant learning rate: " << learning_rate << endl;
    }

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
    const double duel_delta = pow(10,-conf["dueldelta"].as<int>());
    cerr << "duel delta " << duel_delta << endl;
    string msg = "Bandit tuned weights";
    double bleu_map_g = 0;
    double bleu_arm_g = 0;
    double bleu_map = 0;
    double bleu_arm = 0;
    clock_t t0 = clock();
    clock_t t_prev = t0;
    clock_t avg_t = 0; //avg time per iteration

    //adadelta parameters
    double epsilon = pow(10, -6); //TODO tune?
    double decay = 0.95;
     
    cerr << "adadelta params: epsilon=" << epsilon << ", decay=" << decay << endl;

    //initialize adadelta params with zero
    SparseVector<double> exp_g_square_previous;
    SparseVector<double> exp_update_square_previous;
    SparseVector<double> exp_g_square_current;
    SparseVector<double> exp_update_square_current;
    SparseVector<double> step_size_vector;
    
    //trick: initialize a sparse vector of epsilons to control features
    SparseVector<double> epsilon_vector = SetToValue(sparse_weights, epsilon);
    
    while(line_count < max_passes*corpus.size()) {
        SparseVector<double> update;

        decoder.SetId(cur_sent);
        sparse_weights.init_vector(&decoder_weights);        

        if (!silent){
            cerr << "cur sent " << corpus[cur_sent] << endl;
        }
       
        clock_t decode0 = clock(); 
        decoder.Decode(corpus[cur_sent], &observer);
        clock_t decode1 = clock();
        cerr << ((float)decode1-decode0)/CLOCKS_PER_SEC << "s for decoding" << endl;

        double step_size;
        double feedback;
        SparseVector<double> x;
        SparseVector<double> gradient;
        SparseVector<double> avg_x;
    
        if (schedule == "constant"){
            step_size = learning_rate;
        }
        else if (schedule == "cooling"){
            step_size = cooling(cur_pass, learning_rate);
        }
        else if (schedule == "exponential"){
            step_size = exponential(cur_pass, learning_rate);
        }        
       
        //sample and receive feedback
        if (objective == "bayes" or objective == "crossentropy"){ 
            const vector<HypothesisInfo> y_tildes = observer.GetSamples();//sample y_tilde = sample hypothesis
            prob_t z = observer.GetZ();
            double numberOfPaths = observer.GetNumberOfPaths();
            avg_x = observer.GetFeatureExpectation();//compute avg_x = feature expectation via inside-outside
            vector<SparseVector<double>> gradients = ComputeGradients(sparse_weights, avg_x, z, y_tildes, feedback, objective, clip);
            for (int i=0; i<gradients.size(); i++){ 
                if (normalize){
                    gradients[i] *= 1/numberOfPaths;
                }
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
        
            vector<SparseVector<double>> gradients = ComputeGradientsPairwise(sparse_weights, sparse_weights2, avg_1, avg_2, z1, z2, y_tildes1, y_tildes2, feedback, objective, feedback_type);
            for (int i=0; i<gradients.size(); i++){
                if (normalize){
                    gradients[i] *= 1/numberOfPaths;
                }
                gradient += gradients[i]; //sum gradients from all samples
            }

        }


        else if (objective == "duel") {
            random_device rd;
            mt19937 gen(rd());
            normal_distribution<> d(0,1);
            SparseVector<double> sampled_weights;
            for (SparseVector<double>::iterator it = sparse_weights.begin(); it != sparse_weights.end(); ++it) {
                const pair<unsigned, double>& gi = *it;
                sampled_weights.set_value(gi.first,d(gen));
            }
            sampled_weights /= sampled_weights.pnorm(2);

            double feedback_map = observer.GetBest().mt_metric_score;
            SparseVector<double> new_weights = sparse_weights;
            double numberOfPaths = observer.GetNumberOfPaths();
            sampled_weights *= duel_delta;
            new_weights += sampled_weights;
            new_weights.init_vector(&decoder_weights);
            decoder.Decode(corpus[cur_sent], &observer); //re-decode, TODO: better sample weights in updateOracles process (cheaper)
            feedback = observer.GetBest().mt_metric_score;
            sparse_weights.init_vector(&decoder_weights); //reset weights

            if(!silent){
                cerr << "eps ";
                print(cerr, sampled_weights, "=", ",");
                cerr << endl;
                cerr << "feeback(w) " << feedback_map << endl;
                cerr << "w' ";
                print(cerr, new_weights, "=", ",");
                cerr << endl;
                cerr << "feedback(w') " << feedback << endl;
            }

            if (feedback > feedback_map){
                gradient = -sampled_weights;
                if (normalize){
                    gradient *= 1/numberOfPaths;
                }
                gradient /= duel_delta; //because we multiplied it above
            }

         }


        else{
            cerr << "no valid objective given (bayes, crossentropy, pairwise, duel)" << endl;
        }

        //regularize
        if (lambda != 0.0){
            SparseVector<double> scaled_weights = sparse_weights;
            scaled_weights *= (lambda/max_passes*corpus.size());
            gradient += scaled_weights;
        }

        //update weights
        if (schedule!="adadelta"){ //for constant and decaying learning rate step_siz is set above
            if (!gradient.empty()){
                update = gradient;
                update *= step_size;
            }
            else{
                //no update 
            }
        }      
        else { //adadelta
            if(!gradient.empty()){
                exp_g_square_current = exp_g_square_previous;
                exp_g_square_current *= decay;
                SparseVector<double> gradient_square = ElementwiseProduct(gradient, gradient); 
                gradient_square *= (1-decay);
                exp_g_square_current += gradient_square;
             
                step_size_vector = ElementwiseDivision(RMS(exp_update_square_previous, epsilon_vector), RMS(exp_g_square_current, epsilon_vector));
		update = ElementwiseProduct(gradient, step_size_vector);

                exp_update_square_current = exp_update_square_previous;
                exp_update_square_current *= decay;
                SparseVector<double> update_square = ElementwiseProduct(update, update);
                update_square *= (1-decay);
                exp_update_square_current += update_square;

            }
            else {
                exp_g_square_current = exp_g_square_previous;
                exp_update_square_current = exp_update_square_previous;
                //step_size_vector stays from last iteration, update is zero (not in global scope) 
            }
            //prepare for next iteration
            exp_update_square_previous = exp_update_square_current;
            exp_g_square_previous = exp_g_square_current;
        }

        //update
        sparse_weights -= update;


        if (viterbi){
            bleu_map = observer.GetBest().mt_metric_score;
        }
        bleu_arm = 1-feedback;
        decoder_weights.clear();
        sparse_weights.init_vector(&decoder_weights); //copy sparse weights to decoder weights

        if (!silent) {
            cerr << "loss: " << feedback << endl;
            cerr << "x: ";
            print(cerr, x, "=", ",");
            cerr << endl;

            cerr << "avg_x: ";
            print(cerr, avg_x, "=", ",");
            cerr << endl;


            cerr << "update: ";
            print(cerr, update, "=", ",");
            cerr << endl;

            cerr << "new weights: ";
            print(cerr, sparse_weights, "=", ",");
            cerr << endl;
         }
  

        ++line_count;
        ++cur_sent;
        
        if (viterbi){
            bleu_map_g += bleu_map;
        }
        bleu_arm_g += bleu_arm;
        clock_t now = clock();
        clock_t t_delta = now-t_prev;
        avg_t += t_delta;
        t_prev = now;    
	//sparse_weights = clipWeights(sparse_weights, clip); 
	//cerr << "new number of features " << sparse_weights.size() << endl;   

        if (line_count % report_interval ==0 ) {
            cerr << "iter " << line_count;

            if (viterbi){
                cerr << " c^bleu " << bleu_map_g/line_count;
            }

            cerr << " c~bleu " << bleu_arm_g/line_count;
            //cerr << " loss " << 1-bleu_arm;
            
            if (schedule!="adadelta"){
                cerr << " g " << step_size;
            }

            cerr << " l1(w) "  << sparse_weights.pnorm(1);
            cerr << " l2(w) " << sparse_weights.pnorm(2);

            cerr << " avg(secs/it) " << ((float)avg_t)/CLOCKS_PER_SEC/line_count;            

 //           if (schedule=="adadelta"){
  //             cerr << " g ";
    //            print(cerr, step_size_vector, "=", ", ");
      //      }

            cerr << endl;       

            //cerr << "w ";
            //print(cerr, sparse_weights, "=", ",");
            //cerr << endl; 

            //don't write weights
            //ostringstream os;
            //os << weight_dir << "_" << line_count << ".gz"; 
            //sparse_weights.init_vector(&decoder_weights);
            //Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
        }
        if (corpus.size() == cur_sent) { //finished a pass
            // cerr << "<REPORT AFTER EACH PASS>" << endl;
            cur_sent = 0;
            //TODO might insert random permutation here
            cur_pass += 1;
            //cerr << "PASS " << cur_pass << endl;
        }

    }       

    ostringstream os;
    os << weight_dir << ".txt";    
    sparse_weights.init_vector(&decoder_weights);
    Weights::WriteToFile(os.str(), decoder_weights, true, &msg);
    cerr << "Wrote final weights to " << os.str() << endl;

    //TODO write best translations for dev set in output file or evaluate directly? 
}
