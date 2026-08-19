#pragma once
#include <vector>
#include <functional>

#include "tensor.hpp"
#include "cudaTensor.cuh"
#include "cudaLaunch.cuh"
#include "Node.cuh"
#include "AutoGrad.cuh"


template<typename T>
struct Network {
    std::vector<std::function<NodePtr<T>(NodePtr<T>)>> layers;
    //so like the goal here is to create something that can hold our entire CNN
    //We want to create a struct that has a forward function that will run our input across everything
    //So we want to hold a bunch of functions so that we can repeadely call func(func(input))
    template<typename Layer>
    Network& add(Layer& l) {
        //a Layer has a given function given to it and it needs an input, it also wants input + output
        layers.push_back([&l](NodePtr<T> x) { return l.forward(x); });
        return *this;   //
    }

    NodePtr<T> forward(NodePtr<T> x) {
        for (auto& layer : layers) x = layer(x);
        return x;
    }
};
