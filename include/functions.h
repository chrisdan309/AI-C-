#pragma once

double sigmoid(double x);
double sigmoid_derivative(double x);

double tanh_activation(double x);
double tanh_derivative(double x);

double relu(double x);
double relu_derivative(double x);

double leaky_relu(double x, double alpha = 0.01);
double leaky_relu_derivative(double x, double alpha = 0.01);

int heaviside(double x);