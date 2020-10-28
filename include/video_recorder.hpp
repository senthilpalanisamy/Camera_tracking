#ifndef VIDEO_RECORDER_INCLUDE_GAURD
#define VIDEO_RECORDER_INCLUDE_GAURD

#include "opencv2/opencv.hpp"
using cv::Mat;
using cv::Size

class videoRecorder
{
  public:
  videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps=30.0,
		const string ouputPath="./results");
  writeFrames(const vector<Mat> newFrames) const;
  ~videoRecorder();
}

#endif
