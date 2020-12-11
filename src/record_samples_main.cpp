// Author: Senthil Palanisamy
// This file is a convinience script for recording samples from cameras while
// doing experiment

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

   // Gets a folder name and accepts the folder name only if its unique. It repeats 
   // asking for folder name until a unique name is entered

   while(isFolderPathUnique)
   {

   cout<<"Please enter a unique folder name for saving results\n";
   cin>>folder_name;
   outputPath = "./samples/" + return_date_header() + "/" + folder_name;
   struct stat buffer;
   isFolderPathUnique = !stat (outputPath.c_str(), &buffer); 
   }
   cout<<outputPath;



   // While initialising frame grabbers, be sure to specify righ
   // video config fmt files. A wrong file will result in image capture failue
   // frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", false);

   // frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt", true,
   // 		                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");

   frameGrabber imageTransferObj("./config/video_config/new_white_light.fmt");
   imageTransferObj.transferAllImagestoPC();

   auto image0 = imageTransferObj.image0;

   string homographyConfigPath = "./config/camera_homographies/";
   Size stitchedImageSize;

   string imageSizePath = homographyConfigPath + "image_size.txt";
   std::ifstream infile(imageSizePath);
   infile >> stitchedImageSize.width;
   infile >> stitchedImageSize.height;

   // A recorder for writing raw images
   auto recorder = videoRecorder(4, "raw_video", image0.size(),
		                 30, false, outputPath);
   // A recorder for writing stitched images
   auto stitchedRecorder = stitchedVideoRecorder(1, "stitchedVideo", stitchedImageSize,
		                                30, false, outputPath); 

   while(true) 
   {

    // Gets all images from the frame grabber memory
    // If you are doing long operations on images, clone these images into another mat
    // container before processing. This is because these mat object are just shared pointers
    // to the framerabber memory and hence, might be overridden when frame grabbers capture
    // new images.
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
	
    // Writing frames
    recorder.writeFrames(images);
    stitchedRecorder.writeFrames(images);

    // Press escape to exit image capture
    if ( (char)27 == (char) cv::waitKey(WAIT_TIME) ) 
       break;

 
   }



}
