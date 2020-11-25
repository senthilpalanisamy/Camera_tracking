#include <video_recorder.hpp>
#include "simple_capture.hpp"
#include <iostream>
#include <ctime>
#include <sys/stat.h>



using std::to_string;
using namespace std::chrono;
using std::cout;
using std::cin;
using std::endl;

constexpr int WAIT_TIME=1; 

string return_date_time_header()
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_time_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday) 
	                     + "_" + to_string(5+ltm->tm_hour) + ":" + to_string(30+ltm->tm_min) + ":" +
		             to_string(ltm->tm_sec);
}

string return_date_header()
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday); 
   return date_header;
		             
}

int main()
{
   string folder_name, outputPath;
   bool isFolderPathUnique = true;

   while(isFolderPathUnique)
   {

   cout<<"Please enter a unique folder name for saving results\n";
   cin>>folder_name;
   outputPath = "./samples/" + return_date_header() + "/" + folder_name;
   struct stat buffer;
   isFolderPathUnique = !stat (outputPath.c_str(), &buffer); 
   }
   cout<<outputPath;


   frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");
   imageTransferObj.transferAllImagestoPC();

   auto image0 = imageTransferObj.image0;
   auto recorder = videoRecorder(4, "sample_trial", image0.size(),
		                 30, false, outputPath);

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
