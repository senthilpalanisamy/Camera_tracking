#include <iostream>
#include <ctime>
#include <sys/stat.h>

#include <video_recorder.hpp>

#include "simple_capture.hpp"
#include "utility_functions.hpp"



using std::to_string;
using namespace std::chrono;
using std::cout;
using std::cin;
using std::endl;

constexpr int WAIT_TIME=1; 


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



   // frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", false);

   // frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt", true,
   // 		                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");

   frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt");
   imageTransferObj.transferAllImagestoPC();

   auto image0 = imageTransferObj.image0;

   string homographyConfigPath = "./config/camera_homographies/";
   Size stitchedImageSize;

   string imageSizePath = homographyConfigPath + "image_size.txt";
   std::ifstream infile(imageSizePath);
   infile >> stitchedImageSize.width;
   infile >> stitchedImageSize.height;

   auto recorder = videoRecorder(4, "raw_video", image0.size(),
		                 30, false, outputPath);
   auto stitchedRecorder = stitchedVideoRecorder(1, "stitchedVideo", stitchedImageSize,
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
    stitchedRecorder.writeFrames(images);
    // if(cv::waitKey(33) == 
    if ( (char)27 == (char) cv::waitKey(WAIT_TIME) ) 
       break;

 
   }



}
