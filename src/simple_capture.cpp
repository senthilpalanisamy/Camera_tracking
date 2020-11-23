#include <stdio.h>
#include <stdlib.h>
#include "xcliball.h"
#include <iostream>
#include "simple_capture.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <jsoncpp/json/json.h>
#include <fstream>



using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
constexpr int HORIZONTAL_RES=2048;
constexpr int VERTICAL_RES=2048;
using cv::imshow;
using cv::waitKey;
using cv::Mat;
using std::to_string;
using std::vector;


frameGrabber::frameGrabber(const char* configPath, bool doLensCorrection_,
                           string lensCorrectionFolderPath_)
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

 void frameGrabber::performLensCorrection(Mat& image, int imageNo)
  {

  cv::Size imageSize(cv::Size(image.cols,image.rows));
  vector<double> distCoeffs(5);
  cv::Mat cameraMatrix(3, 3, CV_64F);

  string json_file_path = lensCorrectionFolderPath + "/" +"camera_"+ to_string(imageNo)+".json";
  std::ifstream cameraParametersFile(json_file_path, std::ifstream::binary);
  Json::Value cameraParameters;


  cameraParametersFile >> cameraParameters;
  auto intrinsic = cameraParameters["intrinsic"];

  for(int i=0; i < 3; ++i)
   {
     for(int j=0; j < 3; ++j )
     {
       cameraMatrix.at<double>(i, j)= cameraParameters["intrinsic"][i][j].asDouble();
     }
   }
  for(int i=0; i < distCoeffs.size(); ++i)
  {
    distCoeffs[i] = cameraParameters["dist"][0][i].asDouble();


  }

  // Refining the camera matrix using parameters obtained by calibration
  auto new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);

  // Method 1 to undistort the image
  cv::Mat dst;
  cv::undistort( image, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
  image = dst;
  }

  void frameGrabber::transferImagetoPC(size_t frameGrabberNo)
  {
    //cout<<"here";
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
  {

    for(size_t i=0; i < 4; i++)
    {
      transferImagetoPC(i);
    }

    if(doLensCorrection)
    {
      performLensCorrection(image0, 0);
      performLensCorrection(image1, 1);
      performLensCorrection(image2, 2);
      performLensCorrection(image3, 3);
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


   pxd_goUnLive(15);
   pxd_PIXCIclose();
   free(buf0);
   free(buf1);
   free(buf2);
   free(buf3);
   free(buf);
  }

