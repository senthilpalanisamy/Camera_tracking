#include "simple_capture.hpp"
#include "video_recorder.hpp"

using std::to_string;


videoRecorder::videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps,
		const string outputPath)
{
  string fileFormat=".h265";

  for(int i=0; i < writerCount; ++i)
  { 
    string filePath = outputPath + baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, cv::VideoWriter::fourcc('H', '2', '6', '5'),
		            fps, imageSize);
  }
}

void videoRecorder::writeFrames(const vector<Mat>& newFrames)
{
  for(int i=0; i<allWriters.size(); i++)
  {
    allWriters[i].write(newFrames[i]);
  }
}

videoRecorder::~videoRecorder()
{

}


int main()
{

   frameGrabber imageTransferObj("./config/camera3.fmt");

   imageTransferObj.transferAllImagestoPC();
   auto image0 = imageTransferObj.image0;
   auto recorder = videoRecorder(4, "sample_trial", image0.size(),
		                 30);

   for(int i=0; i<100; i++)
   {


    vector<Mat> images;

    images.push_back(imageTransferObj.image0);
    images.push_back(imageTransferObj.image1);
    images.push_back(imageTransferObj.image2);
    images.push_back(imageTransferObj.image3);
	 
    recorder.writeFrames(images);

 
   }



}
