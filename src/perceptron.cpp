#include "perceptron.h"
#include "functions.h"
#include <iostream>

Perceptron::Perceptron(int input_size, float learning_rate=0.01f)
    : weights(input_size, 0.0f), bias(0.0f), learning_rate(learning_rate) {}

int Perceptron::predict(const std::vector<float>& inputs) {
    float weighted_sum = bias;
    for (size_t i = 0; i < inputs.size(); ++i) {
        weighted_sum += weights[i] * inputs[i];
    }
    return heaviside(weighted_sum);
}

void Perceptron::train(const std::vector<std::vector<float>>& training_data, const std::vector<int>& labels, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < training_data.size(); ++i) {
            int prediction = predict(training_data[i]);
            int error = labels[i] - prediction;

            for (size_t j = 0; j < weights.size(); ++j) {
                weights[j] += learning_rate * error * training_data[i][j];
            }
            bias += learning_rate * error;
        }
    }
}

std::vector <float> Perceptron::get_parameters() {
    std::vector <float> parameters = {bias};
    for (float w : weights){
        parameters.insert(parameters.end(), w);
    }
    return parameters;
}

void Perceptron::print_parameters() {
    std::cout << "Weights: ";
    for (float w : weights) {
        std::cout << w << " ";
    }
    std::cout << "\nBias: " << bias << std::endl;
}

