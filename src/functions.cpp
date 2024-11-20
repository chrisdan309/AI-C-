#include "functions.h"
#include <cmath>
#include <math.h>

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double sigmoid_derivative(double x){
    double s = sigmoid(x);
    return s * (1.0 - s);
}

double tanh_activation(double x) {
    return std::tanh(x);
}

double tanh_derivative(double x) {
    double t = tanh_activation(x);
    return 1.0 - t * t;
}

double relu(double x) {
    return x > 0 ? x : 0;
}

double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

double leaky_relu(double x, double alpha) {
    return x > 0 ? x : alpha * x;
}

double leaky_relu_derivative(double x, double alpha) {
    return x > 0 ? 1 : alpha;
}

int heaviside(double x) {
    return x >= 0 ? 1 : 0;
}