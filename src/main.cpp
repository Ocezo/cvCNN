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

    int testCount = 100;          // counts ONLY ORIGINAL samples
    const int scale_factor = 10;  // consistent with cvMap (1..10)
    const int kNumSamples = 7000; // Total number of samples (originals + augmented).
    const std::string imgDir = "../../cvMap/img/out/binning/";
    const std::string labelsPath = "../../cvMap/img/out/labels.txt";

    // ---- 1) Read labels.txt ----
    std::vector<int64_t> labels = readLabels(labelsPath, kNumSamples);

    // ---- 2) Load images roi_nxn_i.jpg ----
    std::vector<cv::Mat> images = loadImages(imgDir, kNumSamples);

    std::cout << "Loaded "
            << images.size() << " images and "
            << labels.size() << " labels.\n";
    
    // ---- 3) Train/Test split: TEST = last originals PER CLASS (uniform), TRAIN = everything else ----
    const int blockSize = 2 * scale_factor;      // e.g. 20 when scale=10
    const int originalsPerBlock = 2;             // always 2

    if (kNumSamples % blockSize != 0) {
        throw std::runtime_error("kNumSamples must be a multiple of (2*scale_factor).");
    }
    const int numBlocks = kNumSamples / blockSize;

    // testCount counts ONLY ORIGINAL samples
    if (testCount % 10 != 0) {
        throw std::runtime_error("For uniform test over 10 classes, testCount must be a multiple of 10.");
    }
    const int perClass = testCount / 10;         // e.g. 10 per class

    // Each block contributes exactly 2 originals: (even label, odd label) = a pair (0,1) or (2,3)...
    // So perClass must be feasible given how many originals you have per class.
    if (perClass % 1 != 0) {
        // nothing to do; kept for clarity
    }

    // We'll mark specific original indices to go to test: isTestOriginal[i] = true if sample i is an original in test.
    std::vector<bool> isTestOriginal(kNumSamples, false);

    // For each class c in 0..9, collect ALL original indices of that class (within=0 or 1 in each block)
    std::vector<int> originalsIdxByClass[10];

    for (int b = 0; b < numBlocks; ++b) {
        int start = b * blockSize;

        // original 0
        int l0 = static_cast<int>(labels[start + 0]);
        if (l0 >= 0 && l0 < 10) originalsIdxByClass[l0].push_back(start + 0);

        // original 1
        int l1 = static_cast<int>(labels[start + 1]);
        if (l1 >= 0 && l1 < 10) originalsIdxByClass[l1].push_back(start + 1);
    }

    // Take the LAST 'perClass' originals of each class into TEST
    for (int c = 0; c < 10; ++c) {
        auto& v = originalsIdxByClass[c];
        if ((int)v.size() < perClass) {
            throw std::runtime_error("Not enough originals for class " + std::to_string(c) +
                                    " (need " + std::to_string(perClass) +
                                    ", have " + std::to_string(v.size()) + ").");
        }
        for (int i = (int)v.size() - perClass; i < (int)v.size(); ++i) {
            isTestOriginal[v[i]] = true;
        }
    }

    // Build TRAIN/TEST:
    // - TEST: only samples i where isTestOriginal[i] == true (these are guaranteed to be originals)
    // - TRAIN: everything else (including augmented samples, and originals not selected for test)
    std::vector<cv::Mat> trainImages, testImages;
    std::vector<int64_t> trainLabels, testLabels;

    trainImages.reserve(kNumSamples - testCount);
    trainLabels.reserve(kNumSamples - testCount);
    testImages.reserve(testCount);
    testLabels.reserve(testCount);

    for (int i = 0; i < kNumSamples; ++i) {
        if (isTestOriginal[i]) {
            // sanity: ensure i is an original (within-block 0 or 1)
            int within = i % blockSize;
            if (within >= originalsPerBlock) {
                throw std::runtime_error("Internal error: selected a non-original for test.");
            }
            testImages.push_back(images[i]);
            testLabels.push_back(labels[i]);
        } else {
            trainImages.push_back(images[i]);
            trainLabels.push_back(labels[i]);
        }
    }

    // Shuffle TRAIN only (keep alignment)
    {
        const int trainCount = (int)trainLabels.size();
        std::vector<int> perm(trainCount);
        std::iota(perm.begin(), perm.end(), 0);

        std::mt19937 rng(42);
        std::shuffle(perm.begin(), perm.end(), rng);

        std::vector<cv::Mat> ti;
        std::vector<int64_t> tl;
        ti.reserve(trainCount);
        tl.reserve(trainCount);

        for (int p : perm) {
            ti.push_back(trainImages[p]);
            tl.push_back(trainLabels[p]);
        }
        trainImages.swap(ti);
        trainLabels.swap(tl);
    }

    std::cout << "Split LAST-ORIGINALS-PER-CLASS (scale=" << scale_factor << "): "
            << "train=" << trainImages.size()
            << ", test(originals-only,uniform)=" << testImages.size()
            << " (perClass=" << perClass
            << ", blockSize=" << blockSize << ")\n";

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
