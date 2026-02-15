#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <array>
#include <numeric>
#include <random>
#include <fstream>
#include <iostream>
#include <filesystem>

#include "TinyCNN.hpp"
#include "SimpleDataset.hpp"

std::vector<int64_t> readLabels(const std::string& labelsPath, int kNumSamples);
std::vector<cv::Mat> loadImages(const std::string& imgDir, int kNumSamples);
void printLabelDistribution(const std::vector<int64_t>& labels, const std::string& name);

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

    // ---- 3) Train/Test split by blocks: 120 train + 20 test, repeated 5 times ----
    const int trainBlock = 120;
    const int testBlock  = 20;
    const int blockSize  = trainBlock + testBlock; // 140

    if (kNumSamples % blockSize != 0) {
        throw std::runtime_error("kNumSamples must be a multiple of 140 for this split scheme.");
    }

    const int numBlocks = kNumSamples / blockSize; // 5 for 700
    const int trainCount = numBlocks * trainBlock; // 600
    const int testCount  = numBlocks * testBlock;  // 100

    std::vector<cv::Mat> trainImages, testImages;
    std::vector<int64_t> trainLabels, testLabels;
    trainImages.reserve(trainCount);
    trainLabels.reserve(trainCount);
    testImages.reserve(testCount);
    testLabels.reserve(testCount);

    // Fill train/test in the requested order
    for (int b = 0; b < numBlocks; ++b) {
        int start = b * blockSize;

        // 120 train: [start .. start+119]
        for (int i = 0; i < trainBlock; ++i) {
            int idx = start + i;
            trainImages.push_back(images[idx]);
            trainLabels.push_back(labels[idx]);
        }

        // 20 test: [start+120 .. start+139]
        for (int i = 0; i < testBlock; ++i) {
            int idx = start + trainBlock + i;
            testImages.push_back(images[idx]);
            testLabels.push_back(labels[idx]);
        }
    }

    // Shuffle train only (keep image/label alignment)
    {
        std::vector<int> perm(trainCount);
        std::iota(perm.begin(), perm.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<cv::Mat> shuffledTrainImages;
        std::vector<int64_t> shuffledTrainLabels;
        shuffledTrainImages.reserve(trainCount);
        shuffledTrainLabels.reserve(trainCount);

        for (int j : perm) {
            shuffledTrainImages.push_back(trainImages[j]);
            shuffledTrainLabels.push_back(trainLabels[j]);
        }

        trainImages.swap(shuffledTrainImages);
        trainLabels.swap(shuffledTrainLabels);
    }

    std::cout << "Split (blocked): train=" << trainImages.size()
              << ", test=" << testImages.size() << "\n";

    printLabelDistribution(trainLabels, "TRAIN");
    printLabelDistribution(testLabels, "TEST");

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

//--------------------------------------------------------------    
// Helper functions
//--------------------------------------------------------------

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

void printLabelDistribution(const std::vector<int64_t>& labels,
                            const std::string& name) {

    std::array<int, 10> counts{};
    counts.fill(0);

    for (auto l : labels) {
        if (l >= 0 && l < 10)
            counts[l]++;
    }

    std::cout << "Label distribution in " << name << ":\n";
    for (int i = 0; i < 10; ++i) {
        std::cout << " " << i << " : " << counts[i] << " /";
    }
    std::cout << "\n";
}
