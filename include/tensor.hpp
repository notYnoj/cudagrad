#pragma once
#include <vector>
#include <map>
#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <random>
#include <concepts>
#include <functional>
#include <algorithm>

template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

enum class Init {
    Zero,
    One,
    Random,
    He
};

template<typename T>
class Tensor;

template <typename T>
class TensorProxy {
private:
    Tensor<T>& tensor;
    size_t idx;
    int depth;
public:
    TensorProxy(Tensor<T>& tensor, size_t idx, int depth) : tensor(tensor), idx(idx), depth(depth) {}
    TensorProxy operator[](int i) {
        if (depth >= (int)tensor.shape.size()) {
            throw std::out_of_range("Too many indices for tensor dimensions");
        }
        if (i < 0 || i >= tensor.shape[depth]) {
            throw std::out_of_range("Index out of bounds");
        }
        return TensorProxy(tensor, idx + i * tensor.strides[depth], depth + 1);
    }
    operator T& () {
        if (depth != (int)tensor.shape.size()) {
            throw std::out_of_range("Too few indices for tensor dimensions");
        }
        return tensor.data[idx];
    }
    operator T() const {
        if (depth != static_cast<int>(tensor.shape.size())) {
            throw std::out_of_range("Too few indices for tensor dimensions");
        }
        return tensor.data[idx];
    }
};

template <typename T>
struct VectorFlatten {
    using ValueType = T;
    static void flatten(const T& input, std::vector<T>& data, std::vector<int>& shape) {
        data.push_back(input);
    }
};

template <typename T>
struct VectorFlatten<std::vector<T>> {
    using ValueType = typename VectorFlatten<T>::ValueType;
    static void flatten(const std::vector<T>& input, std::vector<ValueType>& data, std::vector<int>& shape) {
        shape.push_back((int)input.size());
        for (size_t i = 0; i < input.size(); i++) {
            if (i == 0) {
                VectorFlatten<T>::flatten(input[i], data, shape);
            }
            else {
                std::vector<int> ignored;
                VectorFlatten<T>::flatten(input[i], data, ignored); //discard shape so we only get 1 dimension
            }
        }
    }
};

template <typename T>
class Tensor {
public:
    std::vector<size_t> strides;
    std::vector<T> data;
    std::vector<long long> shape;
    long long sz = -1;

    Tensor() : strides(), data(), shape(), sz(0) {}

    template <typename U>
    Tensor(std::vector<U>& input) {
        VectorFlatten<U>::flatten(input, data, shape);
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size() - 2); i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        szs();
    }
    Tensor(const std::vector<long long>& shape, std::vector<T> data) : data(data), shape(shape) {
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = (int)shape.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        szs();
    }
    Tensor(std::vector<long long> shape, Init init = Init::Zero, int fanIn = 0) : shape(shape) {
        szs();
        data.resize(sz);
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = (int)shape.size() - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        switch (init) {
        case Init::Zero: {
            std::fill(data.begin(), data.end(), T{});
            break;
        }
        case Init::One: {
            std::fill(data.begin(), data.end(), T{ 1 });
            break;
        }
        case Init::Random: {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<T> dist(0.0, 1.0);
            for (T& val : data) {
                val = dist(gen);
            }
            break;
        }
        case Init::He: {
            std::random_device rd;
            std::mt19937 gen(rd());
            float stdD = std::sqrt(2.0f / fanIn);
            std::normal_distribution<T> dist(0.0, stdD);  // mean=0, std=sqrt(2/fanIn)
            for (T& val : data) {
                val = dist(gen);
            }
            break;
        }
        }
    }

    void szs() {
        int temp = 1;
        for (const long long& i : shape) temp *= i;
        sz = temp;
    }

    int getSize() const {
        return sz;
    }
    int ndim() {
        return shape.size();
    }
    Tensor<T> reshape(std::vector<long long>& newShape) {
        shape = std::move(newShape);
        strides.clear();
        strides.resize(shape.size());
        strides[shape.size() - 1] = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; i--) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }
        szs();
        data.resize(sz);
        return *this;
    }

    TensorProxy<T> operator[](const int idx) {
        if (shape.empty()) {
            throw std::out_of_range("You Can't Index into an Empty Tensor");
        }
        if (idx < 0 || idx >= shape[0]) {
            throw std::out_of_range("Index out of bounds");
        }
        return TensorProxy<T>(*this, idx * strides[0], 1);
    }

    Tensor operator+(const Tensor& x) const {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        std::vector<T> temp(sz);
        for (size_t i{}; i < sz; i++) {
            temp[i] = x.data[i] + data[i];
        }
        Tensor ret(shape, temp);
        return ret;
    }
    Tensor& operator+=(const Tensor& x) {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        for (size_t i{}; i < sz; i++) {
            data[i] += x.data[i];
        }
        return *this;
    }
    Tensor operator-(const Tensor& x) const {
        return (*this + (x * -1));
    }
    Tensor& operator-=(const Tensor& x) {
        return (*this += (x * -1));
    }
    Tensor operator*(const Tensor& x) const {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        std::vector<T> temp(sz);
        for (size_t i{}; i < sz; i++) {
            temp[i] = x.data[i] * data[i];
        }
        Tensor ret(shape, temp);
        return ret;
    }

    Tensor& operator*=(const Tensor& x) {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        for (size_t i{}; i < sz; i++) {
            data[i] *= x.data[i];
        }
        return *this;
    }
    Tensor operator/(const Tensor& x) const {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        std::vector<T> temp(sz);
        for (size_t i{}; i < sz; i++) {
            temp[i] = data[i] / x.data[i];
        }
        Tensor ret(shape, temp);
        return ret;
    }
    Tensor& operator/=(const Tensor& x) {
        if (x.shape != shape) {
            throw std::invalid_argument("Different Shape");
        }
        for (size_t i{}; i < sz; i++) {
            data[i] /= x.data[i];
        }
        return *this;
    }

    template <Numeric U>
    Tensor operator* (const U& scalar) const {
        std::vector<T> temp(sz);
        for (size_t i{}; i < sz; i++) {
            temp[i] = scalar * data[i];
        }
        Tensor ret(shape, temp);
        return ret;
    }
    template <Numeric U>
    Tensor operator/ (const U& scalar) const {
        return (*this * (static_cast<U>(1) / scalar));
    }
    template<Numeric U>
    Tensor operator+(const U& scalar) const {
        std::vector<T> temp(sz);
        for (size_t i{}; i < sz; i++) {
            temp[i] = scalar + data[i];
        }
        Tensor ret(shape, temp);
        return ret;
    }
    template<Numeric U>
    Tensor operator-(const U& scalar) const {
        return (*this + (static_cast<U>(-1) * scalar));
    }
    template <Numeric U>
    Tensor& operator*= (const U& scalar) {
        for (size_t i{}; i < sz; i++) {
            data[i] *= scalar;
        }
        return *this;
    }
    template <Numeric U>
    Tensor& operator/= (const U& scalar) {
        for (size_t i{}; i < sz; i++) {
            data[i] /= scalar;
        }
        return *this;
    }
    template<Numeric U>
    Tensor& operator+=(const U& scalar) {
        for (size_t i{}; i < sz; i++) {
            data[i] += scalar;
        }
        return *this;

    }  // adding biases
    template<Numeric U>
    Tensor& operator-=(const U& scalar) {
        for (size_t i{}; i < sz; i++) {
            data[i] -= scalar;
        }
        return *this;
    }

    void printRecursive(size_t dim, size_t offset) const {
        if (dim == shape.size() - 1) {
            std::cout << "[";
            for (size_t i{}; i < (size_t)shape[dim]; i++) {
                std::cout << data[offset + i];
                if (i < (size_t)shape[dim] - 1) std::cout << ", ";
            }
            std::cout << "]";
        }
        else {
            std::cout << "[";
            for (size_t i = 0; i < (size_t)shape[dim]; i++) {
                printRecursive(dim + 1, offset + i * strides[dim]);  // use precomputed strides
                if (i < (size_t)shape[dim] - 1) std::cout << ",\n" << std::string(dim + 1, ' ');
            }
            std::cout << "]";
        }
    }
    void print() const {
        std::cout << "=========SHAPE=========\n";
        for (const long long& i : shape) {
            std::cout << i << " x ";
        }
        std::cout << "\n";
        std::cout << "=========DATA=========\n";
        printRecursive(0, 0);
        std::cout << "\n";
    }

    T get(const std::vector<size_t>& indices) const {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Different size");
        }
        size_t idx = 0;
        for (size_t i{}; i < indices.size(); i++) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Indices out of range");
            }
            idx += (indices[i] * strides[i]);
        }
        return data[idx];
    }

    void place(const std::vector<size_t>& indices, const T& val) {
        if (indices.size() != shape.size()) {
            throw std::out_of_range("Different size");
        }
        size_t idx = 0;
        for (size_t i{}; i < indices.size(); i++) {
            if (indices[i] >= shape[i]) {
                throw std::out_of_range("Indices out of range");
            }
            idx += (indices[i] * strides[i]);
        }
        data[idx] = val;
    }

    Tensor slice(const std::vector<size_t>& start, const std::vector<size_t>& sizes) const {
        if (start.size() != shape.size() || sizes.size() != shape.size()) {
            throw std::out_of_range("Different size");
        }
        std::vector<long long> newShape(sizes.begin(), sizes.end());
        Tensor ret(newShape, Init::Zero);

        const size_t total = ret.sz;
        for (size_t i{}; i < total; i++) {
            std::vector<size_t> outIdx(shape.size());
            size_t remain = i;
            size_t counter = 0;
            for (size_t& idx : outIdx) {
                idx = (remain / ret.strides[counter]);
                remain %= ret.strides[counter++];
            }
            std::vector<size_t> inIdx(shape.size());
            for (size_t j{}; j < shape.size(); j++) {
                inIdx[j] = start[j] + outIdx[j];
            }
            ret.place(outIdx, get(inIdx));
        }
        return ret;
    }
    Tensor transpose() const {
        Tensor ret = *this;
        std::reverse(ret.strides.begin(), ret.strides.end());
        std::reverse(ret.shape.begin(), ret.shape.end());
        return ret;
    }
    Tensor flip() const {
        Tensor ret = *this;
        std::reverse(ret.data.begin(), ret.data.end());
        return ret;
    }
    Tensor apply(std::function<T(T)> func) const {
        Tensor ret = *this;
        for (auto& val : ret.data) {
            val = func(val);
        }
        return ret;
    }
    T sum() const {
        T ret{};
        for (const T& val : data) {
            ret += val;
        }
        return ret;
    }
    //we have a tensor in n dimension, sum returns an n-1 dimension tensor where the dim dimension is removed
    Tensor sum(int dim) const {
        if (dim < 0 || dim >= (int)shape.size()) {
            throw std::out_of_range("Dimension out of range");
        }
        std::vector<long long> newshape;
        for (long long i{}; i < (long long)shape.size(); i++) {
            if (i != dim) {
                newshape.push_back(shape[i]);
            }
        }
        Tensor ret(newshape, Init::Zero);
        std::map<std::vector<size_t>, T> mp;

        for (size_t i{}; i < sz; i++) {
            std::vector<size_t> first;
            size_t idx = i;
            for (size_t d{}; d < shape.size(); d++) {
                if (d != dim) {
                    size_t cur = idx / strides[d];
                    first.push_back(cur);
                    idx = (idx % strides[d]);
                }
                else {
                    idx %= strides[d];
                }
            }
            mp[first] += data[i];
        }
        for (auto& i : mp) {
            ret.place(i.first, i.second);
        }
        return ret;
    }

    T mean() const {
        return sum() / static_cast<T>(sz);
    }

    Tensor clip(T minVal, T maxVal) const {
        Tensor ret = *this;
        for (auto& val : ret.data) {
            val = std::clamp(val, minVal, maxVal);
        }
        return ret;
    }

    T dot(const Tensor<T>& other) const {
        if (other.shape != shape) {
            throw std::out_of_range("To take the dot product size must be the same");
        }
        T total{};
        for (size_t i{}; i < data.size(); i++) {
            total += (data[i] * other.data[i]);
        }
        return total;
    }

    static Tensor zeros(std::vector<int> shape) {
        Tensor ret(shape, Init::Zero);
        return ret;
    };
    static Tensor ones(std::vector<int> shape) {
        Tensor ret(shape, Init::One);
        return ret;
    }
    explicit operator Tensor<double>() const {
        std::vector<double> newData(sz);
        for (size_t i{}; i < sz; i++) {
            newData[i] = static_cast<double>(data[i]);
        }
        return Tensor<double>(shape, newData);
    }
    explicit operator Tensor<float>() const {
        std::vector<float> newData(sz);
        for (size_t i{}; i < sz; i++) {
            newData[i] = static_cast<float>(data[i]);
        }
        return Tensor<float>(shape, newData);
    }

    const std::vector<T>& getData() const { return data; }
    std::vector<T>& getData() { return data; }

    const std::vector<size_t>& getStrides() const { return strides; }
    const std::vector<long long>& getShape() const { return shape; }
};