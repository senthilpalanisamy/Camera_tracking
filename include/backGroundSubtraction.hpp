// Author: Senthil Palanisamy
// This file defines the class for background subtraction. This is a convenience
// class for initializing and using OpenCV background subtraction
#ifndef BACKGROUND_SUBTRACTION
#define BACKGROUND_SUBTRACTION

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

using cv::Mat;
using namespace std;
using namespace cv;
using namespace cv::cuda;

enum Method
{
    MOG,
    MOG2,
};

class BackGroundSubtractor
{

  GpuMat d_frame, d_fgmask, d_fgimg, d_bgimg;
  bool isVisualise;
  Mat fgmask, fgimg, bgimg;
  Ptr<BackgroundSubtractor> mog, mog2;
  Method m;

  public:
  BackGroundSubtractor(Method m, const Mat& firstImage, bool isVisualise_);
  Mat processImage(const Mat& nextFrame);
  void visualiseImage(const Mat& nextFrame);
};


#endif
