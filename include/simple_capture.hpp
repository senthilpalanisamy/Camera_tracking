# ifndef SIMPLE_CAPTURE_INCLUDE_GAURD
# define SIMPLE_CAPTURE_INCLUDE_GAURD

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

class frameGrabber
{

  public:
  cv::Mat image0, image1, image2, image3;
  cv::Mat* image;
  uchar *buf, *buf0, *buf1, *buf2, *buf3;
  frameGrabber(const char* configPath);
  void transferImagetoPC(size_t frameGrabberNo);
  void transferAllImagestoPC();
  void displayAllImages();
  ~frameGrabber();
};
#endif

