#include <stdio.h>
#include <stdlib.h>
#include "xcliball.h"
#include <iostream>
//#include <cstdlib>


#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using std::cout;
using std::endl;

int main(void)
{
int horizontal_resolution = 2048;
int vertical_resolution = 2048;

 //printf("started");
 cout<<"started\n";


pxd_PIXCIopen("", "",  "./camera1.fmt");
// pxd_goLive(0x1, // select PIXCI(R) unit 1
//                    1);  // select frame buffer 1
//
pxd_doSnap(0x1, 1, 0);

     unsigned char *buf = (unsigned char*) malloc(  pxd_imageXdim()    // horizontal resolution
                        * pxd_imageYdim()    // vertical resolution
                        * sizeof(short));
   // auto buf = static_cast<uint8_t *>(pxd_imageXdim() * pxd_imageYdim()  * sizeof(short));


     // pxd_readushort(0x4,         // select PIXCI(R) unit 1
     //                0,           // select frame buffer 1
     //                0, 0,        // upper left X, Y AOI of buffer
     //                -1, -1,      // lower right X, Y AOI of buffer,
     //                             // -1 is an abbreviation for the maximum X or Y
     //                buf,         // program buffer to be filled
     //                pxd_imageXdim() * pxd_imageYdim(),
     //                             // size of program buffer in short's
     //                "Grey");     // color space to access


     pxd_readuchar(0x1,         // select PIXCI(R) unit 1
                    1,           // select frame buffer 1
                    0, 0,        // upper left X, Y AOI of buffer
                    -1, -1,      // lower right X, Y AOI of buffer,
                                 // -1 is an abbreviation for the maximum X or Y
                    buf,         // program buffer to be filled
                    pxd_imageXdim() * pxd_imageYdim(),
                                 // size of program buffer in short's
                    "Grey");     // color space to access

    cv::Mat image = cv::Mat( pxd_imageYdim(),  pxd_imageXdim(), CV_8UC1, (unsigned char*)buf , 0 );
    cout<< image;

    imshow("captured_image", image);
    waitKey(0);


  //   pxd_readushort(0x1,         // select PIXCI(R) unit 1
  //                  1,           // select frame buffer 1
  //                  0, 0,        // upper left X, Y AOI of buffer
  //                  -1, -1,      // lower right X, Y AOI of buffer,
  //                               // -1 is an abbreviation for the maximum X or Y
  //                  buf,         // program buffer to be filled
  //                  pxd_imageXdim() * pxd_imageYdim(),
  //                               // size of program buffer in short's
  //                  "Grey");     // color space to access
  //  image = cv::Mat( pxd_imageYdim(),  pxd_imageXdim(), CV_8UC1, buf , 0 );
  //  cout<< image;

  //  imshow("captured_image", image);
  //  waitKey(0);

    cout<<"finished\n";
    //printf("finished")


free(buf);
pxd_PIXCIclose();


}

