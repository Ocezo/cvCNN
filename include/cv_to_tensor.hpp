#pragma once
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

inline torch::Tensor cvMatToTensor(const cv::Mat& img, int res) {

    CV_Assert(res > 0);
    CV_Assert(img.rows == res && img.cols == res);
    CV_Assert(img.type() == CV_8UC1);

    cv::Mat floatImg;
    img.convertTo(floatImg, CV_32F, 1.0 / 255.0);

    auto tensor = torch::from_blob(
        floatImg.data,
        {1, res, res},
        torch::kFloat32
    ).clone(); // important!

    return tensor; // [1,res,res]
}

inline torch::Tensor cvMatToTensor16x16(const cv::Mat& img) {
    return cvMatToTensor(img, 16);
}
