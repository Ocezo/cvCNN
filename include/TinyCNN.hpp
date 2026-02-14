#pragma once
#include <torch/torch.h>

// TinyCNN-GAP architecture for 16x16x1 input
struct TinyCNNImpl : torch::nn::Module {

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
    torch::nn::AdaptiveAvgPool2d gap{nullptr};
    torch::nn::Linear fc{nullptr};

    TinyCNNImpl(int num_classes = 10) {

        // 16x16 -> 16x16
        conv1 = register_module("conv1",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(16));

        // 16x16 -> 8x8
        conv2 = register_module("conv2",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 24, 3).stride(2).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(24));

        // 8x8 -> 8x8
        conv3 = register_module("conv3",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(24, 32, 3).stride(1).padding(1).bias(false)));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(32));

        // 8x8 -> 4x4
        conv4 = register_module("conv4",
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 48, 3).stride(2).padding(1).bias(false)));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(48));

        gap = register_module("gap",
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1})));

        fc = register_module("fc", torch::nn::Linear(48, num_classes));
    }

    torch::Tensor forward(torch::Tensor x) {

        x = torch::relu(bn1(conv1(x)));
        x = torch::relu(bn2(conv2(x)));
        x = torch::relu(bn3(conv3(x)));
        x = torch::relu(bn4(conv4(x)));

        x = gap(x);                        // [B,48,1,1]
        x = x.view({x.size(0), -1});       // [B,48]
        x = fc(x);                         // logits [B,10]

        return x;
    }
};

TORCH_MODULE(TinyCNN);
