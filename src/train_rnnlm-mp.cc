#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/mp.h"
#include "rnnlm.h"
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>


#include <iostream>
#include <fstream>
#include <vector>
/*
TODO:
- The shadow params in the trainers need to be shared.
*/

using namespace std;
using namespace dynet;
using namespace dynet::expr;
using namespace dynet::mp;
using namespace boost::interprocess;
namespace po = boost::program_options;

void InitCommandLine(int argc, char **argv, po::variables_map *conf) {
    po::options_description opts("Configuration options");
    opts.add_options()
            // Data option
            ("train_file", po::value<string>(), "Train file")
            ("dev_file", po::value<string>(), "Dev file")
            ("workers", po::value<unsigned>(), "workers")
            ("iterations", po::value<unsigned>(), "iterations")
            ("word_embedding_file", po::value<string>(), "word embedding file")
            ("help", "Help");
    po::options_description dcmdline_options;
    dcmdline_options.add(opts);
    po::store(parse_command_line(argc, argv, dcmdline_options), *conf);
    if (conf->empty()) {
        cerr << "Error: empty conf" << endl;
        cerr << dcmdline_options << endl;
        exit(1);
    }
    if (conf->count("help")) {
        cerr << dcmdline_options << endl;
        exit(1);
    }
    vector<string> required_options{"train_file", "dev_file", "workers", "iterations", "word_embedding_file"};

    for (auto opt_str:required_options) {
        if (conf->count(opt_str) == 0) {
            cerr << "Error: missed option" << endl;
            cerr << "Please specify --" << opt_str << endl;
            exit(1);
        }
    }
}

typedef vector<int> Datum;

vector<Datum> ReadData(string filename) {
    vector<Datum> data;
    ifstream fs(filename);
    if (!fs.is_open()) {
        cerr << "ERROR: Unable to open " << filename << endl;
        exit(1);
    }
    string line;
    while (getline(fs, line)) {
        data.push_back(read_sentence(line, d));
    }
    return data;
}

void initialize_word_dict(string word_embedding_file_name) {
    ifstream file_in(word_embedding_file_name);
    assert(file_in);
    unsigned INPUT_DIM, word_count;
    file_in >> word_count >> INPUT_DIM;
    for (int i = 0; i < word_count; i++) {
        string word;
        float embedding;
        file_in >> word;
        d.convert(word);
        for (int j = 0; j < INPUT_DIM; j++) {
            file_in >> embedding;
        }
    }
    kSOS = d.convert("<s>");
    kEOS = d.convert("</s>");
    d.freeze();
    d.set_unk("<unk>");
    file_in.close();
    cout << "Initialized " << d.size() << " words" << endl;
}

template<class T, class D>
class Learner : public ILearner<D, dynet::real> {
public:
    explicit Learner(RNNLanguageModel<T> &rnnlm, unsigned data_size) : rnnlm(rnnlm) {}

    ~Learner() {}

    dynet::real LearnFromDatum(const D &datum, bool learn) {
        ComputationGraph cg;
        Expression loss_expr = rnnlm.BuildLMGraph(datum, cg);
        dynet::real loss = as_scalar(cg.forward(loss_expr));
        if (learn) {
            cg.backward(loss_expr);
        }
        return loss;
    }

    void SaveModel() {}

private:
    RNNLanguageModel<T> &rnnlm;
};

int main(int argc, char **argv) {
    dynet::initialize(argc, argv, true);
    po::variables_map conf;
    InitCommandLine(argc, argv, &conf);

    unsigned num_children = conf["workers"].as<unsigned>();
    assert (num_children <= 64);
    initialize_word_dict(conf["word_embedding_file"].as<std::string>());
    vector<Datum> data = ReadData(conf["train_file"].as<std::string>());
    vector<Datum> dev_data = ReadData(conf["dev_file"].as<std::string>());
    unsigned num_iterations = conf["iterations"].as<unsigned>();
    unsigned dev_frequency = 50000;
    unsigned report_frequency = 5000;


    Model model;
    SimpleSGDTrainer sgd(model, 0.2);
    //AdagradTrainer sgd(model, 0.0);
    //AdamTrainer sgd(model, 0.0);

    RNNLanguageModel<LSTMBuilder> rnnlm(model);
    rnnlm.initial_look_up_table_from_file(argv[2]);

    Learner<LSTMBuilder, Datum> learner(rnnlm, data.size());
    run_multi_process<Datum>(num_children, &learner, &sgd, data, dev_data, num_iterations, dev_frequency,
                             report_frequency);
}
