#include "include/perceptron.h"
#include <vector>
#include <iostream>
int main() {
    std::vector<std::vector<float>> training_data = {
        {0.5, 1.5},
        {1.0, 1.0},
        {1.5, 0.5},
        {-0.5, -1.0},
        {-1.0, -1.5},
        {-1.5, -0.5}
    };
    std::vector<int> labels = {1, 1, 1, 0, 0, 0};

    Perceptron perceptron(2, 0.1);

    perceptron.train(training_data, labels, 20);

    perceptron.print_parameters();
    auto w = perceptron.get_parameters();

    for (auto x : w){
        std::cout << x << std::endl;
    }

    return 0;
}
