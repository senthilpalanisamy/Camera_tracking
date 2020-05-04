#include <stdio.h>
#include <stdlib.h>
#include "xcliball.h"
#include <iostream>
//#include <cstdlib>


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
constexpr int HORIZONTAL_RES=2048;
constexpr int VERTICAL_RES=2048;
using cv::imshow;
using cv::waitKey;


class frameGrabber
{
  public:
  cv::Mat image0, image1, image2, image3;
  cv::Mat* image;

  uchar *buf, *buf0, *buf1, *buf2, *buf3;
  frameGrabber(const char* configPath)
  {
   pxd_PIXCIopen("", "", configPath);

   buf = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(unsigned char));
  
   buf0 = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(unsigned char));
   
   buf1 = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(unsigned char));
   buf2 = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(unsigned char));
   buf3 = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(unsigned char));
  }

  void transferImagetoPC(size_t frameGrabberNo)
  {
    size_t cameraNo = 1 << frameGrabberNo;
    pxd_doSnap(cameraNo, 1, 0);

    switch(frameGrabberNo)
    {
      case 0: buf = buf0;
              image = &image0;
              break;

      case 1: buf = buf1;
              image = &image1;
              break;

      case 2: buf = buf2;
              image = &image2;
              break;

      case 3: buf = buf3;
              image = &image3;
              break;
      default: cout<<"Invalid number for frame grabber unit. Enter a number between 0-3";
    }

    auto i = pxd_readuchar(cameraNo,         // select PIXCI(R) unit 1
                    1,           // select frame buffer 1
                    0, 0,        // upper left X, Y AOI of buffer
                    -1, -1,      // lower right X, Y AOI of buffer,
                                 // -1 is an abbreviation for the maximum X or Y
                    buf,         // program buffer to be filled
                    pxd_imageXdim() * pxd_imageYdim(),
                                 // size of program buffer in short's
                    "Grey");     // color space to access

    *image = cv::Mat( pxd_imageYdim(),  pxd_imageXdim(), CV_8UC1, buf , 0 );

  }

  void transferAllImagestoPC()
  {

    for(size_t i=0; i < 4; i++)
    {
      transferImagetoPC(i);
    }

  }

  void displayAllImages()
  {
    imshow("image0", image0);
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("image3", image3);
    waitKey(0);
  }

  ~frameGrabber()
  {
   pxd_PIXCIclose();
   free(buf0);
   free(buf1);
   free(buf2);
   free(buf3);
  }
};

int main(void)
{
  frameGrabber imageTransferObj("camera2.fmt");
  imageTransferObj.transferAllImagestoPC();
  imageTransferObj.displayAllImages();

}
