#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "TinyCNN.hpp"
#include "SimpleDataset.hpp"

std::vector<int64_t> readLabels(const std::string& labelsPath, int kNumSamples);
std::vector<cv::Mat> loadImages(const std::string& imgDir, int kNumSamples);

int main() {

    const int epochs = 30;
    const int batchSize = 64;

    torch::Device device(torch::cuda::is_available() ?
                         torch::kCUDA : torch::kCPU);

    std::cout << "Using device: " << device << std::endl;

    // ------------------------------------------------------------------
    // Load real dataset (kNumSamples images + labels)
    // ------------------------------------------------------------------

    const int kNumSamples = 700;
    const std::string imgDir = "../../cvMap/img/out/binning/";
    const std::string labelsPath = "../../cvMap/img/out/labels.txt";

    // ---- 1) Read labels.txt ----
    std::vector<int64_t> labels = readLabels(labelsPath, kNumSamples);

    // ---- 2) Load images roi_nxn_i.jpg ----
    std::vector<cv::Mat> images = loadImages(imgDir, kNumSamples);

    std::cout << "Loaded "
            << images.size() << " images and "
            << labels.size() << " labels.\n";

    // ---- 3) Train/Test split (600/100) ----
    const int trainCount = 600;
    const int testCount  = kNumSamples - trainCount; // 100

    std::vector<int> indices(kNumSamples);
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle indices to randomize split (reproducible)
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    std::vector<cv::Mat> trainImages, testImages;
    std::vector<int64_t> trainLabels, testLabels;
    trainImages.reserve(trainCount);
    trainLabels.reserve(trainCount);
    testImages.reserve(testCount);
    testLabels.reserve(testCount);

    for (int i = 0; i < kNumSamples; ++i) {
        int idx = indices[i];
        if (i < trainCount) {
            trainImages.push_back(images[idx]);
            trainLabels.push_back(labels[idx]);
        } else {
            testImages.push_back(images[idx]);
            testLabels.push_back(labels[idx]);
        }
    }

    std::cout << "Split: train=" << trainImages.size()
            << ", test=" << testImages.size() << "\n";

    // ---- 4) Datasets + DataLoaders ----
    auto trainDataset = SimpleDataset(trainImages, trainLabels)
        .map(torch::data::transforms::Stack<>());

    auto testDataset = SimpleDataset(testImages, testLabels)
        .map(torch::data::transforms::Stack<>());

    auto trainLoader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
            std::move(trainDataset),
            torch::data::DataLoaderOptions().batch_size(batchSize)
    );

    auto testLoader = torch::data::make_data_loader(
            std::move(testDataset),
            torch::data::DataLoaderOptions().batch_size(batchSize)
    );

    // ---- 5) Create a model "Tiny CNN" ----
    TinyCNN model(10);
    model->to(device);

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(1e-3).weight_decay(1e-4)
    );

    auto criterion = torch::nn::CrossEntropyLoss();

    for (int epoch = 1; epoch <= epochs; ++epoch) {

        model->train();
        double runningLoss = 0.0;
        int64_t correct = 0;
        int64_t total = 0;

        for (auto& batch : *trainLoader) {

            auto x = batch.data.to(device);
            auto y = batch.target.to(device);

            optimizer.zero_grad();

            auto logits = model->forward(x);
            auto loss = criterion(logits, y);

            loss.backward();
            optimizer.step();

            runningLoss += loss.item<double>() * x.size(0);

            auto pred = logits.argmax(1);
            correct += pred.eq(y).sum().item<int64_t>();
            total += y.size(0);
        }

        std::cout << "Epoch " << epoch
                  << " | Loss: " << runningLoss / total
                  << " | Accuracy: "
                  << 100.0 * correct / total
                  << "%\n";
    }

    // ---- 6) Final evaluation on test set ----
    model->eval();
    torch::NoGradGuard no_grad;

    double testLoss = 0.0;
    int64_t testCorrect = 0;
    int64_t testTotal = 0;

    for (auto& batch : *testLoader) {
        auto x = batch.data.to(device);
        auto y = batch.target.to(device);

        auto logits = model->forward(x);
        auto loss = criterion(logits, y);

        testLoss += loss.item<double>() * x.size(0);

        auto pred = logits.argmax(1);
        testCorrect += pred.eq(y).sum().item<int64_t>();
        testTotal += y.size(0);
    }

    std::cout << "TEST | Loss: " << (testLoss / testTotal)
              << " | Accuracy: " << (100.0 * testCorrect / testTotal) << "%\n";

    return 0;
}

std::vector<int64_t> readLabels(const std::string& labelsPath, int kNumSamples) {
    std::vector<int64_t> labels;
    labels.reserve(kNumSamples);

    std::ifstream fin(labelsPath);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open labels file: " + labelsPath);
    }

    char c;
    while (fin.get(c)) {
        if (c >= '0' && c <= '9') {
            labels.push_back(static_cast<int64_t>(c - '0'));
        }
    }

    if (labels.size() != static_cast<size_t>(kNumSamples)) {
        throw std::runtime_error("labels.txt does not contain 'kNumSamples' labels.");
    }

    return labels;
}

std::vector<cv::Mat> loadImages(const std::string& imgDir, int kNumSamples) {
    std::vector<cv::Mat> images;
    images.reserve(kNumSamples);

    for (int i = 0; i < kNumSamples; ++i) {
        std::string path = imgDir + "roi_nxn_" + std::to_string(i) + ".jpg";

        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);

        if (img.empty()) {
            throw std::runtime_error("Failed to load image: " + path);
        }

        if (img.rows != 16 || img.cols != 16) {
            cv::resize(img, img, cv::Size(16, 16), 0, 0, cv::INTER_NEAREST);
        }

        images.push_back(img);
    }

    return images;
}
