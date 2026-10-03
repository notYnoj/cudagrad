// Timing harness around the unmodified WordHuntAI CNN (cnn.hpp) + cudagrad.
// Same model / batch / optimizer as train.cpp; replicates CNN::train's inner loop
// so the training part of each epoch can be timed separately from evaluation.
//
// usage: train_bench <epochs> <data_dir> [max_batches_per_epoch (0 = all)]
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>

#include "cnn.hpp"
#include "loader.hpp"

using clk = std::chrono::steady_clock;
static double secs(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count(); }

int main(int argc, char** argv) {
    const int epochs = argc > 1 ? std::atoi(argv[1]) : 3;
    const std::string data = argc > 2 ? std::string(argv[2]) + "/" : "../data/";
    const size_t maxBatches = argc > 3 ? (size_t)std::atoll(argv[3]) : 0;
    const size_t BATCH = 128;

    auto trainData = loadEMNIST<float>(data + "emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte",
                                       data + "emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte", -1);
    auto testData  = loadEMNIST<float>(data + "emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte",
                                       data + "emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte", -1);

    CNN<float> cnn(0.01f, 0.000001f, epochs, cosine_annealing_LR<float>);
    cnn.add(std::make_unique<Conv<float>>(cnn.opt, 1, 8, 3));
    cnn.add(std::make_unique<MaxPool<float>>(2));
    cnn.add(std::make_unique<Conv<float>>(cnn.opt, 8, 16, 3));
    cnn.add(std::make_unique<MaxPool<float>>(2));
    cnn.add(std::make_unique<Flatten<float>>());
    cnn.add(std::make_unique<Linear<float>>(cnn.opt, 16 * 5 * 5, 128, true));
    cnn.add(std::make_unique<Linear<float>>(cnn.opt, 128, 26, false));
    cnn.opt.totalEpochs = epochs;

    std::vector<size_t> idx(trainData.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(0);
    CudaTensor<float> lossCorrect(std::vector<long long>{2}, false);
    int* dLabels = nullptr;
    CUDA_CHECK(cudaMalloc(&dLabels, sizeof(int) * BATCH));

    double sumTrain = 0; int counted = 0;
    for (int e = 0; e < epochs; ++e) {
        std::shuffle(idx.begin(), idx.end(), rng);
        lossCorrect.zero();
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = clk::now();
        size_t nb = 0, seen = 0;
        for (size_t s = 0; s < idx.size(); s += BATCH) {
            if (maxBatches && nb >= maxBatches) break;
            const size_t end = std::min(s + BATCH, idx.size());
            std::vector<size_t> b(idx.begin() + s, idx.begin() + end);
            std::vector<int> labels; labels.reserve(b.size());
            for (size_t j : b) labels.push_back(trainData[j].second);
            CUDA_CHECK(cudaMemcpy(dLabels, labels.data(), sizeof(int) * b.size(), cudaMemcpyHostToDevice));
            cnn.opt.zero_grad();
            auto logits = cnn.forward(leaf(CNN<float>::toBatch(trainData, b)));
            auto loss = softmax_cross_entropy(logits, labels);
            loss->backward();
            cnn.opt.step();
            launch_accumulate_loss_correct(lossCorrect.data, loss->value.data, logits->value.data, dLabels, (int)b.size(), 26);
            ++nb; seen += b.size();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t1 = clk::now();
        cnn.opt.stepEpoch();
        auto lc = lossCorrect.to_host().getData();
        double tTrain = secs(t0, t1);
        std::cout << "epoch " << e + 1 << "/" << epochs << " | batches " << nb
                  << " | loss " << lc[0] / seen << " | acc " << 100.0 * lc[1] / seen << "%"
                  << " | train " << tTrain << " s";
        if (!maxBatches) {
            auto t2 = clk::now();
            auto [vl, va] = cnn.evaluate(testData);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << " | val acc " << va * 100 << "% | eval " << secs(t2, clk::now()) << " s";
        }
        std::cout << "\n";
        if (e > 0 || epochs == 1) { sumTrain += tTrain; ++counted; }
    }
    std::cout << "mean train epoch (excluding epoch 1): " << sumTrain / counted << " s\n";
    cudaFree(dLabels);
}
