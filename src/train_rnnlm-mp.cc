#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include "dynet/mp.h"
#include "rnnlm.h"
#include <boost/algorithm/string.hpp>

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

    if (argc < 6) {
        cerr << "Usage: " << argv[0] << " cores word_embedding.data corpus.data dev.data iterations" << endl;
        return 1;
    }
    unsigned num_children = atoi(argv[1]);
    assert (num_children <= 64);
    initialize_word_dict(argv[2]);
    vector<Datum> data = ReadData(argv[3]);
    vector<Datum> dev_data = ReadData(argv[4]);
    unsigned num_iterations = atoi(argv[5]);
    unsigned dev_frequency = 5000;
    unsigned report_frequency = 10;


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
