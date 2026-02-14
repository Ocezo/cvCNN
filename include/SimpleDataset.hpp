#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "cv_to_tensor.hpp"

struct SimpleDataset : torch::data::datasets::Dataset<SimpleDataset> {

    std::vector<cv::Mat> images_;
    std::vector<int64_t> labels_;

    SimpleDataset(std::vector<cv::Mat> images,
                  std::vector<int64_t> labels)
        : images_(std::move(images)), labels_(std::move(labels)) {}

    torch::data::Example<> get(size_t index) override {

        auto x = cvMatToTensor16x16(images_[index]);
        auto y = torch::tensor(labels_[index], torch::kLong);

        return {x, y};
    }

    torch::optional<size_t> size() const override {
        return images_.size();
    }
};
