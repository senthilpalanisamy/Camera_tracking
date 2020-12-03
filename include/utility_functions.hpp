#ifndef UTILITY_FUNCTIONS_INCLUDE_GAURD
#define UTILITY_FUNCTIONS_INCLUDE_GAURD

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using std::vector;
//using cv::detail::MatchesInfo;
using std::cout;

extern Mat outputImage;



struct ImageStitchData
{
  Mat dstImage;
  Mat inputImage;
  Mat homography;
};

Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography);
void* WarpandStitchImages(void *arguments);

#endif

