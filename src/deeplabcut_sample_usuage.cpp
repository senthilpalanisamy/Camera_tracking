// Author: Senthil Palanisamy
// A file showing how the pybind11 wrapper can be used for calling the python
// script for calling deeplabcut

#include<deeplabFrameInference.hpp>

using cv::imread;

int main (int argc, char *argv[])
{
    // float data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    //Mat mat1 (cv::Size (5, 2), CV_32F, data, Mat::AUTO_STEP);

    auto image = imread("./samples/test.jpeg");
    cout<<image.type()<<"\n";
    auto dlc = DeepLabFrameInference();
    dlc.predictPoseForImage(image);
    dlc.predictPoseForImage(image);

    return 0;
}
