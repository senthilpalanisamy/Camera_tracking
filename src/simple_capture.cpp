// Author: Senthil Palanisamy
// This file defines a class for capturing images from framGrabber memory
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <iostream>


#include "xcliball.h"
#include "simple_capture.hpp"
#include "utility_functions.hpp"


using namespace std::chrono;



using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
using cv::imshow;
using cv::waitKey;
using cv::Mat;
using std::to_string;
using std::vector;





frameGrabber::frameGrabber(const char* configPath, bool doLensCorrection_,
                           string lensCorrectionFolderPath_)
  // Constructor for intialising the image capture using frame grabbers
  // confifpath - path to video config file that is to be used
  // doLensCorrection_  - A boolean indicating whether lens correction should be done
  // lensCorrectionFolderPath_ - folderpath containing lens distortion and camera
  //                             intrinsic parameters
  // Lens distortion is not free. It takes upto 10 ms for correcting lens distortions
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
   doLensCorrection = doLensCorrection_;
   lensCorrectionFolderPath = lensCorrectionFolderPath_;
   cout<<"start error status:";
   cout<<pxd_goLive(15, 1);
  }


  void frameGrabber::transferImagetoPC(size_t frameGrabberNo)
  // Transfers an image from the framegrabber memory to PC memory
  // frameGrabberNo - The frame grabber number to be accessed. 
  {
    //cout<<"here";
    // each PIXCI(R) unit stands for one bit in a 4 bit number. We can select any number 
    // of framegrabber by setting bits. In this case, we are only trying to access one 
    // particular camera and hence, we are tying to set only that particular bit
    size_t cameraNo = 1 << frameGrabberNo;
    //pxd_doSnap(cameraNo, 1, 0);

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
    //cout<<"reading buffer\n";

    auto i = pxd_readuchar(cameraNo,         // select PIXCI(R) unit 1
                    1,           // select frame buffer 1
                    0, 0,        // upper left X, Y AOI of buffer
                    -1, -1,      // lower right X, Y AOI of buffer,
                                 // -1 is an abbreviation for the maximum X or Y
                    buf,         // program buffer to be filled
                    pxd_imageXdim() * pxd_imageYdim(),
                                 // size of program buffer in short's
                    "Grey");     // color space to access
    // buf = pxd_capturedBuffer(0x1)
    //cout<<"\nerror status"<<i<<pxd_mesgErrorCode(i)<<"\n";

    *image = cv::Mat( pxd_imageYdim(),  pxd_imageXdim(), CV_8UC1, buf , 0 );

  }

  void frameGrabber::transferAllImagestoPC()
  // Transfer all 4 camera images from framegrabber memory to PC
  {

    for(size_t i=0; i < 4; i++)
    {
      transferImagetoPC(i);
    }

    auto start = high_resolution_clock::now();

    // TODO: Avoid doing file read in every image capture cycle. Read the file 
    // once in the constructor to avoid repeated work slowing down capture
    if(doLensCorrection)
    {
      performLensCorrection(image0, 0, lensCorrectionFolderPath);
      performLensCorrection(image1, 1, lensCorrectionFolderPath);
      performLensCorrection(image2, 2, lensCorrectionFolderPath);
      performLensCorrection(image3, 3, lensCorrectionFolderPath);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\ntime: "<<duration.count() << endl;


  }

  void frameGrabber::displayAllImages()
  // A debugging function for seeing all camera images
  {
    imshow("image0", image0);
    imshow("image1", image1);
    imshow("image2", image2);
    imshow("image3", image3);
    waitKey(0);
  }

  frameGrabber::~frameGrabber()
  // A destructor for freeing up all allocated memory
  {


   pxd_goUnLive(15);
   pxd_PIXCIclose();
   free(buf0);
   free(buf1);
   free(buf2);
   free(buf3);
   free(buf);
  }

