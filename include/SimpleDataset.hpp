#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include "cv_to_tensor.hpp"

struct SimpleDataset : torch::data::datasets::Dataset<SimpleDataset> {

    std::vector<cv::Mat> images_;
    std::vector<int64_t> labels_;
    int input_res_;

    SimpleDataset(std::vector<cv::Mat> images,
                  std::vector<int64_t> labels,
                  int input_res = 16)
        : images_(std::move(images)),
          labels_(std::move(labels)),
          input_res_(input_res) {}

    torch::data::Example<> get(size_t index) override {

        auto x = cvMatToTensor(images_[index], input_res_);
        auto y = torch::tensor(labels_[index], torch::kLong);

        return {x, y};
    }

    torch::optional<size_t> size() const override {
        return images_.size();
    }
};
