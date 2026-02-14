#include <torch/torch.h>
#include <opencv2/opencv.hpp>
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

    auto dataset = SimpleDataset(images, labels)
        .map(torch::data::transforms::Stack<>());

    auto dataLoader = torch::data::make_data_loader<
        torch::data::samplers::RandomSampler>(
            std::move(dataset),
            torch::data::DataLoaderOptions().batch_size(batchSize)
    );

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

        for (auto& batch : *dataLoader) {

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
