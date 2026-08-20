#pragma once


#include <cmath>
#include <cstddef>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Linear decay: decay with a linear slope of (lrMax-lrMin)/(0-tE)-> at x = 0 lrMax at x = totaEpochs lrMin
//so given some x we have lrMax + -m(x) = y where m is (lrMax-lrMin)/tE * epoch  -> lrMax - (lrMax-lrMin) *epoch/tE
template<typename T>
T linear_LR(std::size_t epoch, std::size_t totalEpochs, T lrMax, T lrMin) {
    double frac = totalEpochs ? (double)epoch / (double)totalEpochs : 1.0;
    if (frac > 1.0) frac = 1.0;
    return lrMax - (lrMax - lrMin) * (T)frac;
}

//js use the formula </3
template<typename T>
T cosine_annealing_LR(std::size_t epoch, std::size_t totalEpochs, T lrMax, T lrMin) {
    double t = totalEpochs ? (double)epoch / (double)totalEpochs : 1.0;
    return lrMin + T(0.5) * (lrMax - lrMin) * (T)(1.0 + std::cos(M_PI * t));
}

//no decay
template<typename T>
T constant_LR(std::size_t yes, std::size_t no, T lrMax, T maybeSo) { return lrMax; }
