#include "xcliball.h"

#include "opencv2/core.hpp"
#ifdef HAVE_OPENCV_XFEATURES2D
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


pxd_PIXCIopen("", "",  "./camera1.fmt");
pxd_doSnap(0x1, 1, 0);
{
    void *buf = malloc(  pxd_imageXdim()    // horizontal resolution
                       * pxd_imageYdim()    // vertical resolution
                       * sizeof(short));

    pxd_readushort(0x1,         // select PIXCI(R) unit 1
                   1,           // select frame buffer 1
                   0, 0,        // upper left X, Y AOI of buffer
                   -1, -1,      // lower right X, Y AOI of buffer,
                                // -1 is an abbreviation for the maximum X or Y
                   buf,         // program buffer to be filled
                   pxd_imageXdim() * pxd_imageYdim(),
                                // size of program buffer in short's
                   "Grey");     // color space to access
    imshow("SURF Keypoints", buf);

    free(buf);
}
pxd_PIXCIclose();


}

