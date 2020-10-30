#include <stdio.h>
#include <stdlib.h>
#include "xcliball.h"
#include <iostream>
//#include <cstdlib>
//
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
 #include <experimental/filesystem>

namespace fs = std::experimental::filesystem::v1;


using namespace cv::xfeatures2d;
using std::cout;
using std::endl;
constexpr int HORIZONTAL_RES=2048;
constexpr int VERTICAL_RES=2048;
using cv::imshow;
using cv::imwrite;
using cv::waitKey;
using std::string;
using std::to_string;

constexpr int sequenceCount=100;


class frameGrabber
{
  cv::Mat image;
  uchar *buf, *buf0, *buf1, *buf2, *buf3;
  public:
	
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
   int odata;
   odata = pxd_getGPIn(0x1,    // select PIXCI(R) unit 1
                        0);     // reserved

   cout<<"\nwaiting for trigger\n";

   while (odata == pxd_getGPIn(0x1, 0));
   cout<<"\nreceived trigger\n";

   for(int i=0; i < 4; i++)
   {

     size_t cameraNo = 1 << i;
     //pxd_goLivePair(cameraNo, 0, 1);
    pxd_goLiveSeq(cameraNo,              // select PIXCI(R) unit 1
                  1,                // select start frame buffer 1
                  pxd_imageZdim(),  // select end as last frame buffer
                  1,                // incrementing by one buffer
                  sequenceCount,    // for this many captures
                  1);               // advancing to next buffer after each 1 frame
     pxd_goLive(cameraNo, 1);  
   }
  }

  void writeImagestoPC()
  {
    for(size_t i=0; i < 4; i++)
    {

      string base_path = string("./results/")+string("oct_30/")+to_string(i)+"/";
      fs::create_directories(base_path);
      for(int j=1; i<=pxd_imageZdim(); ++i)
      {
    
       size_t cameraNo = 1 << j;
       //pxd_doSnap(cameraNo, 1, 0);

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

       image = cv::Mat( pxd_imageYdim(),  pxd_imageXdim(), CV_8UC1, buf , 0 );

       string result_path = base_path+"_"+to_string(j)+".png";
       imwrite(result_path, image);
    }

  }
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
   frameGrabber imageTransferObj("../data/camera2.fmt");
   imageTransferObj.writeImagestoPC();
 }
