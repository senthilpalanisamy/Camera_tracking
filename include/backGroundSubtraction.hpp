#ifndef BACKGROUND_SUBTRACTION
#define BACKGROUND_SUBTRACTION

#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/cudabgsegm.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"

using cv::Mat;

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
  Ptr<BackgroundSubtractor> mog;

  public:
  BackGroundSubtraction(Method m, const Mat& firstImage, bool isVisualise_);
  void processImage();
  void visualiseImage()
};


#endif
