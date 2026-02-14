#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

inline torch::Tensor cvMatToTensor16x16(const cv::Mat& img) {

    CV_Assert(img.rows == 16 && img.cols == 16);
    CV_Assert(img.type() == CV_8UC1);

    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        floatImg.data,
        {1, 16, 16},
        torch::kFloat32
    ).clone(); // important!

    return tensor; // [1,16,16]
}
