#pragma once
#include "cnn.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>


// Really annoying but idx headers store 4-byte ints big-endian -> x86 is little-endian, so we gotta swap alot and make sure it is in propery binary format.
inline uint32_t swapBytes(uint32_t v) {
    return ((v & 0xFF000000u) >> 24) | ((v & 0x00FF0000u) >> 8)
         | ((v & 0x0000FF00u) << 8)  | ((v & 0x000000FFu) << 24);
}
inline uint32_t readBigEndian(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    return swapBytes(v);
}


template <typename T>
typename CNN<T>::Dataset loadEMNIST(const std::string& imagePath,
                                    const std::string& labelPath,
                                    int maxSamples = -1) {
    std::ifstream imgFile(imagePath, std::ios::binary);
    std::ifstream lblFile(labelPath, std::ios::binary);
    if (!imgFile) throw std::runtime_error("cant open image file: " + imagePath);
    if (!lblFile) throw std::runtime_error("cant open label file: " + labelPath);

    uint32_t imgMagic = readBigEndian(imgFile);
    uint32_t numImages = readBigEndian(imgFile);
    uint32_t rows = readBigEndian(imgFile);
    uint32_t cols = readBigEndian(imgFile);

    uint32_t lblMagic = readBigEndian(lblFile);
    uint32_t numLabels = readBigEndian(lblFile);
    //debug out range?
    std::cout << "Label magic: " << lblMagic << " numLabels: " << numLabels << "\n";
    uint8_t raw = 0;
    for (int dbg = 0; dbg < 5; dbg++) {
        lblFile.read(reinterpret_cast<char*>(&raw), 1);
        std::cout << "raw label " << dbg << " " << static_cast<int>(raw) << "\n";
    }

    lblFile.seekg(8, std::ios::beg);  // seek back to after header

    //why tf does emnist use magic numbers to ensure ts is opened properly
    if (imgMagic != 2051) throw std::runtime_error("bad image magic (got " + std::to_string(imgMagic) + "): " + imagePath);
    if (lblMagic != 2049) throw std::runtime_error("bad label magic (got " + std::to_string(lblMagic) + "): " + labelPath);
    if (numImages != numLabels) throw std::runtime_error("image/label count mismatch");

    uint32_t total = (maxSamples > 0) ? std::min<uint32_t>(maxSamples, numImages) : numImages;

    typename CNN<T>::Dataset dataset;
    dataset.reserve(total);
    std::vector<uint8_t> buf(rows * cols);

    for (uint32_t n = 0; n < total; n++) {
        imgFile.read(reinterpret_cast<char*>(buf.data()), rows * cols);
        if (imgFile.gcount() != (std::streamsize)(rows * cols))
            throw std::runtime_error("end of file while reading image ts is bad " + std::to_string(n));

        // why is emnist so fucking annoying: images are stored TRANSPOSED. need x y to y x .
        std::vector<T> data(rows * cols);
        for (uint32_t r = 0; r < rows; r++)
            for (uint32_t c = 0; c < cols; c++)
                data[r * cols + c] = static_cast<T>(buf[c * rows + r]) / static_cast<T>(255);

        uint8_t label = 0;
        lblFile.read(reinterpret_cast<char*>(&label), 1);
        int classIdx = static_cast<int>(label) - 1;   // FFS letters are 1-26 need 0-25
        if (classIdx < 0 || classIdx > 25) {
            throw std::runtime_error("label out of range it is instead: " + std::to_string((int)label));
        }
        Tensor<T> img(std::vector<long long>{static_cast<long long>(rows), static_cast<long long>(cols)}, data);
        dataset.push_back({ {img}, classIdx });

        if ((n + 1) % 10000 == 0) std::cout << "  " << (n + 1) << " loaded\n";
    }
    std::cout << "nice everything is loaded.\n";
    return dataset;
}