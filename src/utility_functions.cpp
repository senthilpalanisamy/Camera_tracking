#include <pthread.h>

#include "utility_functions.hpp"


pthread_mutex_t lock;

class ParallelPixelTransfer: public ParallelLoopBody
{
  mutable Mat unwarpedImage;
  mutable Mat stitchedImage;
  public:
    ParallelPixelTransfer(Mat& srcPtr, Mat& dstPtr)
    {
      unwarpedImage = srcPtr;
      stitchedImage = dstPtr;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    {
      int r=0;
      for(r = range.start; r < range.end; r++)
      {
        int i = r / unwarpedImage.cols;
        int j = r % unwarpedImage.cols;

          if(stitchedImage.ptr<uchar>(i)[j] == 0)
          {
            stitchedImage.ptr<uchar>(i)[j] = unwarpedImage.ptr<uchar>(i)[j];
          }
          }
     }

    ParallelPixelTransfer& operator=(const ParallelPixelTransfer&)
    {
      return *this;
    }

};

  void* WarpandStitchImages(void *arguments)
  {
    cout<<"\nthread started\n";

    Mat imageUnwarped;
    ImageStitchData *stitchArgs = (ImageStitchData*) arguments;

    pthread_mutex_lock(&lock);

    Size warpedImageSize = stitchArgs->dstImage.size();
    // warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
    //                  warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);

    warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
                     warpedImageSize, INTER_LINEAR);

    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, stitchArgs->dstImage);
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
    pthread_mutex_unlock(&lock);

    return NULL;
    }



  Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography)
  {

    Mat imageUnwarped;
    size_t i=0, j=0;

    Size warpedImageSize = stitchedImage.size();

    // warpPerspective (ipImage, imageUnwarped, Homography, warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);
    warpPerspective (ipImage, imageUnwarped, Homography, warpedImageSize,INTER_LINEAR);

    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, stitchedImage);

    // cout<<"starting parallel for";
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
    // cout<<"ending parallel for\n";

    #ifdef DEBUG
    namedWindow("StitchedImage",WINDOW_NORMAL);
    resizeWindow("StitchedImage", 600, 600);
    imshow("StitchedImage", stitchedImage);


    namedWindow("warpedImage",WINDOW_NORMAL);
    resizeWindow("warpedImage", 600, 600);
    imshow("warpedImage", imageUnwarped);
    waitKey(0);
    #endif

    return stitchedImage;

  }






