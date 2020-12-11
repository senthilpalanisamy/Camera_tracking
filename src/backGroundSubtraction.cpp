// Author: Senthil Palanisamy
// This file defines the class for background subtraction. This is a convenience
// class for initializing and using OpenCV background subtraction

#include <ctime>
#include <fstream>
#include <sys/stat.h>
#include <algorithm>
#include <math.h>
#include <iostream>

#include <video_recorder.hpp>
#include <backGroundSubtraction.hpp>

#include "simple_capture.hpp"

using cv::Scalar;


BackGroundSubtractor::BackGroundSubtractor(Method m_, const Mat& firstImage, 
                                           bool isVisualise_)
// Constructor
// m_ - an enum which describes the method used for background subtraction. Choices
// allowed are MOG or MOG2
// firstImage - Intial OpenCv mat image for the algorithm
// isVisualise_ - enables or disables visualizing provisions. Useful when debugging
// algorithms
  {

    GpuMat d_frame(firstImage);
    if(m == MOG)
    {
      mog = cuda::createBackgroundSubtractorMOG();
      mog->apply(d_frame, d_fgmask, 0.01);
    }
    else
    {
      mog = cuda::createBackgroundSubtractorMOG2();
      mog->apply(d_frame, d_fgmask);
    }

    isVisualise = isVisualise_;

    if(isVisualise)
    {
      namedWindow("image", WINDOW_NORMAL);
      namedWindow("foreground mask", WINDOW_NORMAL);
      namedWindow("foreground image", WINDOW_NORMAL);
      namedWindow("mean background image", WINDOW_NORMAL);
    }
    m = m_;
  }

  Mat BackGroundSubtractor::processImage(const Mat& nextFrame)
  // Process each image for the background subtraction pipeline
  // nextFrame - each successive image to be fed into the algorithm
  {

    d_frame.upload(nextFrame);
    if(m == MOG)
    {
        mog->apply(d_frame, d_fgmask, 0.01);
    }
    else
    {
       mog->apply(d_frame, d_fgmask);
    }

    mog->getBackgroundImage(d_bgimg);



    d_fgmask.download(fgmask);

    if(isVisualise)
    {
      visualiseImage(nextFrame);
    }
    return fgmask;
  }

  void BackGroundSubtractor::visualiseImage(const Mat& nextFrame)
  // A utility function for visualizing all background subtraction outputs.
  // nextFrame - the frame currently processed.
  {

  d_fgimg.create(d_frame.size(), d_frame.type());
  d_fgimg.setTo(Scalar::all(0));
  d_frame.copyTo(d_fgimg, d_fgmask);

   d_fgimg.download(fgimg);
   if (!d_bgimg.empty())
     d_bgimg.download(bgimg);


  imshow("image", nextFrame);
  imshow("foreground mask", fgmask);
  imshow("foreground image", fgimg);
  if (!bgimg.empty())
      imshow("mean background image", bgimg);
  waitKey(30);
  }
