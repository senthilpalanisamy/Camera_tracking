#include "video_recorder.hpp"
#include <sys/types.h> 
#include <sys/stat.h> 
#include <unistd.h> 
#include<stdlib.h>

using std::to_string;
using std::cout;


videoRecorder::videoRecorder(const int writerCount, const string baseName_,
                const Size imageSize, double fps, bool isColor,
                const string outputPath_)
{
  fileFormat=".mp4";
  int fourcc = VideoWriter::fourcc('M', 'P', '4', 'V');       
  baseName = baseName_;
  outputPath = outputPath_;

  int check = mkdir(outputPath.c_str(), 0777);




  for(int i=0; i < writerCount; ++i)
  {
    string filePath = baseName + to_string(i) + fileFormat;
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

 for(int i=0; i < allWriters.size(); ++i)
 {
	 string command = "mv " + baseName + to_string(i) + fileFormat +" " 
		          + outputPath + "/" + baseName + 
			  to_string(i) + fileFormat;
	 system(command.c_str());
 }

}


