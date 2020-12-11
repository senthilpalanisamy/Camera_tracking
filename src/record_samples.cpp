// Author: Senthil Palanisamy
// This file contains a class that is used for writing images into video. There is one
// class for raw images and one class for stitching images and writing video as a stitched
// image

#include "video_recorder.hpp"
#include <sys/types.h> 
#include <sys/stat.h> 
#include <unistd.h> 
#include <stdlib.h>


using std::to_string;
using std::cout;


videoRecorder::videoRecorder(const int writerCount, const string baseName_,
                const Size imageSize, double fps, bool isColor,
                const string outputPath_,  bool isMultiProcess_)
// Constructor for writing raw images as vidoes
// writerCount - No of videos to be written (Usually 4 since we have 4 cameras)
// baseName_ - basename for all video. This name followed by 0, 1, 2, or 3 is that the 
//             video will be named as
// imageSize - imageSize in each frame of the video 
// fps - fps to write the video at
// isColor - indicates if the image is color (Usally false since we only deal with grey 
//            scale images)
// outputPath - Path where the video should be written to
// isMultiProcess_ - should multi-threading / multiprocessing be used to speed up the process
{
  fileFormat=".mp4";
  int fourcc = VideoWriter::fourcc('M', 'P', '4', 'V');       
  isMultiProcess = isMultiProcess_;
  baseName = baseName_;
  outputPath = outputPath_;

  // create the given output path
  string command = "mkdir -p "+ outputPath;
  auto _ = system(command.c_str());


  // intiliase all video writers
  for(int i=0; i < writerCount; ++i)
  {
    string filePath = baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, fourcc, fps, imageSize, isColor);
  }


  // intialise all std::async variables if multiprocessing is used
  if(isMultiProcess)
  {
    m_futures.resize(writerCount);
    images.resize(writerCount);
    isFirst = true;
  }

}

void videoRecorder::writeFrames(const vector<Mat>& newFrames)
{

  // Use 4 threads to writer a video, one for each image feed sepertely to speed up the 
  // process. Since we will need to hold a std::future reference to begin with, a special
  // if case is setup for running the async thread the first time. But beyond that,
  // the process is regular - Check if all 4 threads have finished. If the threads have
  // finished, write the new images
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
      // Check if all threads have finished
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
      // Write new images if previous write has finished

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
    // write all frames in a sequential fashion if multithreading is not used

    for(int i=0; i<allWriters.size(); i++)
    {
      // cout<<i<<"\t"<<newFrames[i].size().width<<"\t"<<newFrames[i].size().height<<"\t";

      allWriters[i].write(newFrames[i]);
    }

  }

}

videoRecorder::~videoRecorder()
 // Move the videos to the desired location specified. OpenCV video writers is a little
 // annoying - its not possible to speicfy a direct path to write the videos to. The
 // vidoes are written by default in the base directory. So all videos in the base 
 // directory are then shifted to the desired directory
{

 // Release all video writers
 for(auto writer: allWriters)
 {
	 writer.release();
 }

 // Move all vidoes
 for(int i=0; i < allWriters.size(); ++i)
 {
	 string command = "mv " + baseName + to_string(i) + fileFormat +" " 
		          + outputPath + "/" + baseName + 
			  to_string(i) + fileFormat;
	 auto _ = system(command.c_str());
 }

}


stitchedVideoRecorder::stitchedVideoRecorder(const int writerCount, const string baseName,
                                            const Size imageSize, double fps,
                                            bool isColor,const string outputPath, bool isMultiProcess,
			                    string homographyConfigPath, string lensCorrectionFolderPath):
	                                    videoRecorder
			                    (writerCount, baseName, imageSize, fps,
                                             isColor, outputPath, isMultiProcess)


// Constructor for writing stitched images as vidoes
// writerCount - No of videos to be written (Usually 4 since we have 4 cameras)
// baseName_ - basename for all video. This name followed by 0, 1, 2, or 3 is that the 
//             video will be named as
// imageSize - imageSize in each frame of the video 
// fps - fps to write the video at
// isColor - indicates if the image is color (Usally false since we only deal with grey 
//            scale images)
// outputPath - Path where the video should be written to
// isMultiProcess_ - should multi-threading / multiprocessing be used to speed up the process
// homographyConfigPath - Path where the pre-calibbrated homographies of camreas are
//                        stored
// lensCorrectionFolderPath - path which contains json files containing camera intrinsics
//                            and lens distortion parameters
  {
  imgStitcher = imageStitcher(homographyConfigPath, true, lensCorrectionFolderPath);
  stitchStatus = std::async([](){Mat x; return x;});
  writingStatus = std::async([](){});
  }

void stitchedVideoRecorder::writeFrames(const vector<Mat>& newFrames)
 // Stitched the given frames and writes the stitched frame
{
  if(isMultiProcess)
   // use multiple threads for accomplishing the stitching and writing process
  {
    // Check if image stitching thread is finished
    if(stitchStatus.wait_for(std::chrono::seconds(0)) == std::future_status::ready) 
    {
       // push the stitched image to a queue and begin next image stitching 
       stitchedImages.push(std::move(stitchStatus.get()));	   
       stitchStatus = std::async(std::launch::async, &imageStitcher::stitchImagesOnline,
                	               imgStitcher, newFrames); 
    }

    // check if video writing thread has completed and that there are some images to 
    // write in the queue
    // Since image writing is much faster than image stitching, we will never run into 
    // a scenario where the queue starts using explosive amounts of memory

    if((writingStatus.wait_for(std::chrono::seconds(0)) == std::future_status::ready) &&
       stitchedImages.size() > 0)
    {
      // Pop the image from the queue
      Mat imageTowrite = stitchedImages.front();
      stitchedImages.pop();

      // write the image to video writer
      writingStatus = async(std::launch::async, &VideoWriter::write, allWriters[0], imageTowrite);
    }
  }
  else
  {
    // Construct the stitched image and write the stitched image in a sequential 
    // fashion. (This will slow down the exectuio heavily)
    Mat stitchedImage = imgStitcher.stitchImagesOnline(newFrames);
    allWriters[0].write(stitchedImage);


  }
}






