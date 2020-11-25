#include "video_recorder.hpp"
#include <sys/types.h> 
#include <sys/stat.h> 
#include <unistd.h> 
#include<stdlib.h>

using std::to_string;
using std::cout;


videoRecorder::videoRecorder(const int writerCount, const string baseName_,
                const Size imageSize, double fps, bool isColor,
                const string outputPath_,  bool isMultiProcess_)
{
  string fileFormat=".mp4";
  int fourcc = VideoWriter::fourcc('M', 'P', '4', 'V');       
  isMultiProcess = isMultiProcess_;



  for(int i=0; i < writerCount; ++i)
  {
    string filePath = outputPath + baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, fourcc, fps, imageSize, isColor);
  }

  if(isMultiProcess)
  {
    m_futures.reserve(writerCount);
    images.reserve(writerCount);
  }

}

void videoRecorder::writeFrames(const vector<Mat>& newFrames)
{

  if(isMultiProcess)
  {
    if(isFirst)
    {
      isFinished = true;
      isFirst = false;
    }
    else
    {
      isFinished = true;
      for(int i=0; i < newFrames.size(); ++i)
      {
        if(m_futures[i].wait_for(std::chrono::seconds(0)) != std::future_status::ready)
        {
          isFinished = false;
          break;
        }

      }

    }

    if(isFinished)
    {

      for(size_t i=0; i < newFrames.size();++i)
      {
        images[i] = newFrames[i];
      }

      for(size_t i=0; i <newFrames.size(); ++i)
      {
        m_futures[i] = async(std::launch::async, &VideoWriter::write, allWriters[i], images[i]);
      }

    }

  }
  else
  {

    for(int i=0; i<allWriters.size(); i++)
    {
      // cout<<i<<"\t"<<newFrames[i].size().width<<"\t"<<newFrames[i].size().height<<"\t";

      allWriters[i].write(newFrames[i]);
    }

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


