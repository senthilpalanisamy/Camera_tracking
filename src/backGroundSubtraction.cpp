#include <backGroundSubtraction.hpp>
#include <iostream>
#include <string>



using namespace std;
using namespace cv;
using namespace cv::cuda;


BackGroundSubtractor::BackGroundSubtractor(Method m, const Mat& firstImage, 
                                           bool isVisualise_)
  {

    GpuMat d_frame(firstImage);
    if(m == MOG)
    {
      mog = cuda::createBackgroundSubtractorMOG();
    }
    else
    {
      mog = cuda::createBackgroundSubtractorMOG2();
    }

    if(m = MOG)
    {
      mog->apply(d_frame, d_fgmask, 0.01);
    }
    else
    {

      mog2->apply(d_frame, d_fgmask);

    }

    d_frame.upload(firstImage);


    isVisualise = isVisualise_;

    if(isVisualise)
    {
      namedWindow("image", WINDOW_NORMAL);
      namedWindow("foreground mask", WINDOW_NORMAL);
      namedWindow("foreground image", WINDOW_NORMAL);
      namedWindow("mean background image", WINDOW_NORMAL);
    }
  }

  void BackgroundSubtractor::processImage()
  {
    if(m == MOG)
    {
        mog->apply(d_frame, d_fgmask, 0.01);
        mog->getBackgroundImage(d_bgimg);
    }
    else
    {
       mog2->apply(d_frame, d_fgmask);
       mog2->getBackgroundImage(d_bgimg);

    }

    d_fgimg.create(d_frame.size(), d_frame.type());
    d_fgimg.setTo(Scalar::all(0));
    d_frame.copyTo(d_fgimg, d_fgmask);

    d_fgmask.download(fgmask);
    d_fgimg.download(fgimg);
    if (!d_bgimg.empty())
        d_bgimg.download(bgimg);

    if(isVisualise)
    {
      visualiseImage()
    }
  }

  void BackGroundSubtractor::visualiseImage()
  {
  imshow("image", frame);
  imshow("foreground mask", fgmask);
  imshow("foreground image", fgimg);
  if (!bgimg.empty())
      imshow("mean background image", bgimg);
  }
