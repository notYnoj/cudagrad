#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <numeric>
#include <algorithm>
#include <utility>

#include "autogradEngine.cuh"
#include "schedulers.hpp"

template<typename T>
struct CNN {
    using Dataset = std::vector<std::pair<std::vector<Tensor<T>>, int>>;
    //optimizer will own params and call stepEpoch
    SGD<T> opt;
    //entire net
    std::vector<std::unique_ptr<Module<T>>> layers;

    explicit CNN(T lrMax = T(0.01), T lrMin = T(1e-6), size_t totalEpochs = 20,
                 std::function<T(std::size_t, std::size_t, T, T)> sched = cosine_annealing_LR<T>): opt(lrMax, lrMin, totalEpochs, std::move(sched)) {}

    //Adds a layer
    CNN& add(std::unique_ptr<Module<T>> layer) {
        layers.push_back(std::move(layer));
        return *this;
    }

    //creates a single tensor with all channels from a vector of channels 
    static Tensor<T> toInput(const std::vector<Tensor<T>>& channels) {
        const long long C = (long long)channels.size();
        const long long H = channels[0].getShape()[0];
        const long long W = channels[0].getShape()[1];
        std::vector<T> data;
        data.reserve((size_t)C * H * W);
        for (const auto& ch : channels)
            for (const T& v : ch.getData()) data.push_back(v);
        return Tensor<T>(std::vector<long long>{1, C, H, W}, data);
    }
    //the goal here is to create a tensor of size [Example, Channels, H, W] from a dataset of the given idxes
    static Tensor<T> toBatch(const Dataset& data, const std::vector<size_t>& batchIdx){
        long long Examples = (long long) batchIdx.size();
        long long Channels = (long long) data[batchIdx[0]].first.size();
        long long H = (long long) data[batchIdx[0]].first[0].getShape()[0];
        long long W = (long long) data[batchIdx[0]].first[0].getShape()[1];
        std::vector<T> buf;
        buf.reserve((size_t)Examples * Channels * H * W);
        for(size_t idx: batchIdx){
            for(const auto& channel : data[idx].first){
                for(const T& v : channel.getData()){
                    buf.push_back(v);
                }
            }
        }
        
        return Tensor<T>(std::vector<long long>{Examples, Channels, H, W}, buf);
    }
    NodePtr<T> forward(NodePtr<T> x) {
        for (auto& l : layers) x = l->forward(x);
        return x;
    }
    //Creates a function that runs a single N channel image
    NodePtr<T> runSingle(const std::vector<Tensor<T>>& input){
        return forward(leaf(toInput(input)));
    }

    //predicts for a singel N channel image
    int predict(const std::vector<Tensor<T>>& image) {
        NodePtr<T> ret = runSingle(image);
        Tensor<T> logits = ret->value.to_host();
        const auto& d = logits.getData();
        return (int)(std::max_element(d.begin(), d.end()) - d.begin());
    }

    //evaluates a dataset
    std::pair<T, T> evaluate(Dataset& data, size_t BZ = 128) {
        const size_t BATCH_SIZE = (BZ == 0) ? 1 : BZ;
        CudaTensor<T> loss_and_correct_DEVICE(std::vector<long long>{ 2 }, false);
        loss_and_correct_DEVICE.zero();
        int* dLabels = nullptr;
        CUDA_CHECK(cudaMalloc(&dLabels, sizeof(int) * BATCH_SIZE));

        for(size_t start = 0; start < data.size(); start+=BATCH_SIZE){
            size_t end = std::min(data.size(), start + BATCH_SIZE);
            size_t M = end-start;
            std::vector<size_t> currentIdexes(M);
            std::iota(currentIdexes.begin(), currentIdexes.end(), start);
            std::vector<int> labels;
            for(int idx = start; idx < end; idx++){
                labels.push_back(data[idx].second);
            }
            CUDA_CHECK( cudaMemcpy(dLabels, labels.data(), sizeof(int) * M, cudaMemcpyHostToDevice) );
            Tensor<T> curBatch = toBatch(data, currentIdexes);
            NodePtr<T> x = leaf(curBatch);
            NodePtr<T> logits = forward(x);
            NodePtr<T> loss = softmax_cross_entropy(logits, labels);
            launch_accumulate_loss_correct(loss_and_correct_DEVICE.data, loss->value.data, logits->value.data, dLabels, M, 26);
        }
        Tensor<T> loss_and_correct_HOST = loss_and_correct_DEVICE.to_host();
        const auto& d = loss_and_correct_HOST.getData();
        cudaFree(dLabels);
        return {d[0]/(T)data.size(), d[1]/(T)data.size()};
    }

    //saves when valacc is better than before
    void train(Dataset& trainData, size_t epochs, const std::string& path, Dataset* valData = nullptr, size_t BZ = 128) {
        try {
            load(path);
            std::cout << "resumed from " << path << "weights.bin\n";
        } catch (const std::exception& e) {
            std::cout << "no existing weights (" << e.what() << "), training from scratch\n";
        }
        opt.totalEpochs = epochs;
        //if its 0 we have inf loop
        const size_t BATCH_SIZE = (BZ == 0) ? 1 : BZ;

        std::vector<size_t> idx(trainData.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(std::random_device{}());
        T bestValAcc = T(-1);


        CudaTensor<T> loss_and_correct_DEVICE(std::vector<long long>{ 2 }, false);
        //device labels per batch size
        int* dLabels = nullptr;
        CUDA_CHECK(cudaMalloc(&dLabels, sizeof(int) * BATCH_SIZE));
        for (size_t e = 0; e < epochs; ++e) {
            std::shuffle(idx.begin(), idx.end(), rng);
            loss_and_correct_DEVICE.zero();
            for(size_t start = 0; start < idx.size(); start+=BATCH_SIZE){
                const size_t end = std::min(start + BATCH_SIZE, idx.size());
                std::vector<size_t> batchIdxs(idx.begin() + start, idx.begin() + end);
                const int Examples = (int)batchIdxs.size();
                std::vector<int> labels;
                labels.reserve(Examples);
                for (size_t j : batchIdxs) labels.push_back(trainData[j].second);
                CUDA_CHECK(cudaMemcpy(dLabels, labels.data(),
                                  sizeof(int) * Examples, cudaMemcpyHostToDevice));

                opt.zero_grad();
                NodePtr<T> x = leaf(toBatch(trainData, batchIdxs));
                NodePtr<T> logits = forward(x);
                auto loss = softmax_cross_entropy(logits, labels);
                loss->backward();
                opt.step();
                launch_accumulate_loss_correct(loss_and_correct_DEVICE.data, loss->value.data, logits->value.data, dLabels, Examples, 26);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            opt.stepEpoch();
            Tensor<T> loss_and_correct_HOST = loss_and_correct_DEVICE.to_host();
            const auto& d = loss_and_correct_HOST.getData();
            T trLoss =  (T)d[0] / (T)trainData.size();
            T trAcc  = (T)d[1] / (T)trainData.size();
            std::cout << "epoch " << (e + 1) << "/" << epochs
                      << " | loss " << trLoss << " | acc " << (trAcc * 100) << "%"
                      << " | lr " << opt.lr;
            

            if (valData) {
                auto [vLoss, vAcc] = evaluate(*valData);
                std::cout << " | val loss " << vLoss << " | val acc " << (vAcc * 100) << "%";
                if (vAcc > bestValAcc) {
                    bestValAcc = vAcc;
                    save(path);
                    std::cout << "  (saved)";
                }
            }
            std::cout << "\n";
        }
        cudaFree(dLabels);
    }

    void perLetterAccuracy(Dataset& data, int numLetter = 26) {
        std::vector<int> total(numLetter, 0), correct(numLetter, 0);
        for (auto& [image, label] : data) {
            int pred = predict(image);
            ++total[label];
            if (pred == label) ++correct[label];
        }
        for (int c = 0; c < numLetter; ++c) {
            char letter = (char)('A' + c);
            float acc = total[c] ? (100.0f * (float)correct[c] / (float)total[c]) : 0.0f;
            std::cout << letter << ": " << acc << "%  (" << correct[c] << "/" << total[c] << ")\n";
        }
    }

    void save(const std::string& path) const { save_params<T>(opt.params, path + "weights.bin"); }
    void load(const std::string& path)       { load_params<T>(opt.params, path + "weights.bin"); }
};
