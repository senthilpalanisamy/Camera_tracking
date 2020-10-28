#include "simple_capture.hpp"
#include "video_recorder.hpp"

using std::to_string;
using cv::VideoWriter;


videoRecorder::videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps,
		const string ouputPath="./results");

{
  vector<VideoWriter> allWriters;
  string fileFormat=".h265";

  for(int i=0; i < writerCount; ++i)
  { 
    string filePath = outputPath + baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, CV_FOURCC('H', '2', '6', '5'),
		            fps, imageSize);
  }
}

videoRecorder::writeFrames(const<Mat> newFrames)
{
  for(int i=0; i<allWriters.size(); i++)
  {
    allWriters.write(newFrames[i]);
  }
}

videoRecorder::~videoRecorder()
{

}


