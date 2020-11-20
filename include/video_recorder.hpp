#ifndef VIDEO_RECORDER_INCLUDE_GAURD
#define VIDEO_RECORDER_INCLUDE_GAURD

#include "opencv2/opencv.hpp"
using cv::Mat;
using cv::Size;
using std::string;
using std::vector;
using cv::VideoWriter;

class videoRecorder
{

  vector<VideoWriter> allWriters;
  public:
  string baseName, fileFormat, outputPath;
  videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps=30.0,
                bool isColor=false,
		const string outputPath="./results");
  void writeFrames(const vector<Mat>& newFrames);
  ~videoRecorder();
};

#endif
