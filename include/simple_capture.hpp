// Author: Senthil Palanisamy
// This file defines a class for capturing images from framGrabber memory

# ifndef SIMPLE_CAPTURE_INCLUDE_GAURD
# define SIMPLE_CAPTURE_INCLUDE_GAURD

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using std::string;

class frameGrabber
{

  public:
  cv::Mat image0, image1, image2, image3;
  cv::Mat* image;
  bool doLensCorrection;
  string lensCorrectionFolderPath;
  uchar *buf, *buf0, *buf1, *buf2, *buf3;
  frameGrabber(const char* configPath, bool doLensCorrection=false,
              string lensCorrectionFolderPath=".");
  void transferImagetoPC(size_t frameGrabberNo);
  void transferAllImagestoPC();
  void displayAllImages();
  ~frameGrabber();

  private:
  //void performLensCorrection(cv::Mat& image, int cameraNo);
};
#endif

