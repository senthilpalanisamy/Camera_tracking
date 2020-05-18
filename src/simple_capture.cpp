#include <stdio.h>
#include <stdlib.h>
#include "xcliball.h"
#include <iostream>
#include "simple_capture.hpp"
//#include <cstdlib>



using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
constexpr int HORIZONTAL_RES=2048;
constexpr int VERTICAL_RES=2048;
using cv::imshow;
using cv::waitKey;


frameGrabber::frameGrabber(const char* configPath)
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

  void frameGrabber::transferImagetoPC(size_t frameGrabberNo)
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

  void frameGrabber::transferAllImagestoPC()
  {

    for(size_t i=0; i < 4; i++)
    {
      transferImagetoPC(i);
    }

  }

  void frameGrabber::displayAllImages()
  {
    imshow("image0", image0);
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("image3", image3);
    waitKey(0);
  }

  frameGrabber::~frameGrabber()
  {
   pxd_PIXCIclose();
   free(buf0);
   free(buf1);
   free(buf2);
   free(buf3);
  }

// int main(void)
// {
//   frameGrabber imageTransferObj("../data/camera2.fmt");
//   imageTransferObj.transferAllImagestoPC();
//   imageTransferObj.displayAllImages();
// 
// }
