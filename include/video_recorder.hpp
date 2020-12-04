#ifndef VIDEO_RECORDER_INCLUDE_GAURD
#define VIDEO_RECORDER_INCLUDE_GAURD

#include<future>

#include "opencv2/opencv.hpp"

#include "image_stitching.hpp"

using cv::Mat;
using cv::Size;
using std::string;
using std::vector;
using cv::VideoWriter;
using std::async;
using std::future;
using std::queue;
using std::ref;


class videoRecorder
{

  public:
  vector<VideoWriter> allWriters;
  string baseName, fileFormat, outputPath;
  vector<future<void>> m_futures;
  vector<Mat> images;
  bool isMultiProcess, isFirst, isFinished;
  videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps=30.0,
                bool isColor=false,
		const string outputPath="./results", bool isMultiProcess=false);
  void writeFrames(const vector<Mat>& newFrames);
  ~videoRecorder();
};


class stitchedVideoRecorder : public videoRecorder 
{   
  public:
  imageStitcher imgStitcher;
  future<Mat> stitchStatus;
  future<void> writingStatus;
  queue<Mat> stitchedImages;

  stitchedVideoRecorder(const int writerCount, const string baseName,
                        const Size imageSize, double fps=30.0,
                        bool isColor=false,
		        const string outputPath="./results", bool isMultiProcess=false,
			string homographyConfigPath_="./config/camera_homographies/",
			string lensCorrectionFolderPath_="./config/camera_intrinsics_1024x1024");
  void writeFrames(const vector<Mat>& newFrames);
};

#endif
