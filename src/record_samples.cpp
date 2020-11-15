#include "simple_capture.hpp"
#include "video_recorder.hpp"

using std::to_string;
constexpr int WAIT_TIME=1; 
using std::cout;


videoRecorder::videoRecorder(const int writerCount, const string baseName,
                const Size imageSize, double fps,
		const string outputPath)
{
  string fileFormat=".mp4";
  //int fourcc = cv::cv::CV_FOURCC(*"mp4v");
  //int fourcc = cv::VideoWriter_fourcc(*"X264");
  //int fourcc = VideoWriter::fourcc('a', 'v', 'c', '1');  
  int fourcc = VideoWriter::fourcc('M', 'P', '4', 'V');       



  for(int i=0; i < writerCount; ++i)
  { 
    string filePath = outputPath + baseName + to_string(i) + fileFormat;
    allWriters.emplace_back(filePath, fourcc, fps, imageSize, false);
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


int main()
{

   frameGrabber imageTransferObj("./config/room_light.fmt");

   imageTransferObj.transferAllImagestoPC();
   auto image0 = imageTransferObj.image0;
   auto recorder = videoRecorder(4, "sample_trial", image0.size(),
		                 30);

   while(true) 
   {

    imageTransferObj.transferAllImagestoPC();
    vector<Mat> images;

    images.push_back(imageTransferObj.image0);
    images.push_back(imageTransferObj.image1);
    images.push_back(imageTransferObj.image2);
    images.push_back(imageTransferObj.image3);

    for(int i=0; i < 4; ++i)
    {
     imshow("image"+to_string(i), images[i]);
     //waitKey(WAIT_TIME);
    }
	 
    recorder.writeFrames(images);
    // if(cv::waitKey(33) == 
    if ( (char)27 == (char) cv::waitKey(WAIT_TIME) ) 
       break;

 
   }



}
