#include "video_recorder.hpp"

using std::to_string;
using std::cout;


videoRecorder::videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps, bool isColor,
                const string outputPath)
{
  string fileFormat=".mp4";
  int fourcc = VideoWriter::fourcc('M', 'P', '4', 'V');       



  for(int i=0; i < writerCount; ++i)
  {
    string filePath = outputPath + baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, fourcc, fps, imageSize, isColor);
  }
}

void videoRecorder::writeFrames(const vector<Mat>& newFrames)
{
  for(int i=0; i<allWriters.size(); i++)
  {
    cout<<i<<"\t"<<newFrames[i].size().width<<"\t"<<newFrames[i].size().height<<"\t";
    allWriters[i].write(newFrames[i]);
  }
}

videoRecorder::~videoRecorder()
{

 for(auto writer: allWriters)
 {
	 writer.release();
 }

}


