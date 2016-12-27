#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/dict.h"
#include "dynet/lstm.h"

#include <vector>

using namespace std;
using namespace dynet;
using namespace dynet::expr;

unsigned LAYERS = 2;
unsigned INPUT_DIM;  //256

dynet::Dict d;
int kSOS;
int kEOS;

template<class Builder>
struct RNNLanguageModel {
    LookupParameter p_c;
    Parameter p_R;
    Parameter p_bias;
    Builder builder;

    explicit RNNLanguageModel(Model &model) : builder(LAYERS, INPUT_DIM, INPUT_DIM, model) {
        p_c = model.add_lookup_parameters(d.size(), {INPUT_DIM});
        p_R = model.add_parameters({d.size(), INPUT_DIM});
        p_bias = model.add_parameters({d.size()});
    }

    void initial_look_up_table_from_file(std::string file_name) {
        std::cout << "Initializing lookup table from " << file_name << " ..." << std::endl;
        std::ifstream em_in(file_name);
        assert(em_in);
        unsigned em_count, em_size;
        em_in >> em_count >> em_size;
        cout<<em_size<<" "<<INPUT_DIM<<endl;
        assert(em_size == INPUT_DIM);
        std::vector<float> e(em_size);
        std::string w;
        unsigned initialized_word_count = 0;
        for (unsigned i = 0; i < em_count; i++) {
            em_in >> w;
            for (unsigned j = 0; j < em_size; j++) {
                em_in >> e[j];
            }
            unsigned index = d.convert(w);
            initialized_word_count++;
            assert(index < d.size() && index >= 0);
            p_c.initialize(index, e);
        }
        std::cout << "Initialize " << initialized_word_count << " words" << std::endl;
        std::cout << d.size() - initialized_word_count << " words not initialized" << std::endl;
    }

    // return Expression of total loss
    Expression BuildLMGraph(const vector<int> &sent, ComputationGraph &cg) {
        const unsigned slen = sent.size() - 1;
        builder.new_graph(cg);  // reset RNN builder for new graph
        builder.start_new_sequence();
        Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
        Expression i_bias = parameter(cg, p_bias);  // word bias
        vector<Expression> errs;
        for (unsigned t = 0; t < slen; ++t) {
            Expression i_x_t = lookup(cg, p_c, sent[t]);
            // y_t = RNN(x_t)
            Expression i_y_t = builder.add_input(i_x_t);
            Expression i_r_t = i_bias + i_R * i_y_t;

            // LogSoftmax followed by PickElement can be written in one step
            // using PickNegLogSoftmax
            Expression i_err = pickneglogsoftmax(i_r_t, sent[t + 1]);
            errs.push_back(i_err);
        }
        Expression i_nerr = sum(errs);
        return i_nerr;
    }

    // return Expression for total loss
    void RandomSample(int max_len = 150) {
        cerr << endl;
        ComputationGraph cg;
        builder.new_graph(cg);  // reset RNN builder for new graph
        builder.start_new_sequence();

        Expression i_R = parameter(cg, p_R);
        Expression i_bias = parameter(cg, p_bias);
        vector<Expression> errs;
        int len = 0;
        int cur = kSOS;
        while (len < max_len && cur != kEOS) {
            ++len;
            Expression i_x_t = lookup(cg, p_c, cur);
            // y_t = RNN(x_t)
            Expression i_y_t = builder.add_input(i_x_t);
            Expression i_r_t = i_bias + i_R * i_y_t;

            Expression ydist = softmax(i_r_t);

            unsigned w = 0;
            while (w == 0 || (int) w == kSOS) {
                auto dist = as_vector(cg.incremental_forward(ydist));
                double p = rand01();
                for (; w < dist.size(); ++w) {
                    p -= dist[w];
                    if (p < 0.0) { break; }
                }
                if (w == dist.size()) w = kEOS;
            }
            cerr << (len == 1 ? "" : " ") << d.convert(w);
            cur = w;
        }
        cerr << endl;
    }
};
