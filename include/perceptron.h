#pragma once

#include <vector>

class Perceptron {
private:
    std::vector<float> weights;
    float bias;
    float learning_rate;

public:
    Perceptron(int input_size, float learning_rate = 0.01);

    int predict(const std::vector<float>& inputs);
    void train(const std::vector<std::vector<float>>& training_data, const std::vector<int>& labels, int epochs);
    std::vector <float> get_parameters();
    
    void print_parameters();
};
