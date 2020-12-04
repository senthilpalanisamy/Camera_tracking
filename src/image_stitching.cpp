#include <limits>
#include <iostream>
#include <fstream>


#include "utility_functions.hpp"
#include "image_stitching.hpp"



# define DEBUG 0

using std::to_string;



//using cv::detail::MatchesInfo;
using std::cout;
using std::endl;

Mat outputImage;

imageStitcher::imageStitcher(string configPath, bool doLensCorrection_,
		string lensCorrectionFolderPath_)
  {

   vector<string> fileNames = {"camera1_to_ground_plane.txt", "camera2_to_ground_plane.txt", 
	                       "camera3_to_ground_plane.txt", "camera4_to_ground_plane.txt"};
   doLensCorrection = doLensCorrection_;
   lensCorrectionFolderPath = lensCorrectionFolderPath_;

   double number;
   for(int h_index=0; h_index < fileNames.size(); ++h_index)
   { 
     string fileName = fileNames[h_index];
     std::ifstream infile(configPath+fileName);
     //allHomographies[h_index].create(3, 3, CV_64F, Scalar(0));
     Mat homography = Mat(3, 3, CV_64F, Scalar(0));
     //double homography[9];
     for(int i=0; i<3; ++i)
     {
       for(int j=0; j<3; ++j)
       {
	 infile >> number;
	 //homography[3*i+j] = number;
	 homography.at<double>(i, j) = number;
       }

     }
     allHomographies.push_back(homography);

    }

     string imageSizePath = configPath + "image_size.txt";
     std::ifstream infile(imageSizePath);
     infile >> finalSize.width;
     infile >> finalSize.height;
  }


  Mat imageStitcher::stitchImagesOnline(vector<Mat> images)
  {

   Mat dstImage(finalSize, CV_8U, Scalar(0));
   outputImage = dstImage;

   size_t i;

   pthread_t tid[images.size()];
   ImageStitchData imagedataVector[4];
   if(doLensCorrection)
   {
	   
     for(int i=0; i < 4; ++i)
     {
      performLensCorrection(images[i], i, lensCorrectionFolderPath);
     }
   }

  

   for(i=0; i < images.size(); i++)
   {
   struct ImageStitchData imageStitchData;
   imageStitchData.dstImage = outputImage;
   imageStitchData.homography = allHomographies[i];
   imageStitchData.inputImage = images[i];
   imagedataVector[i] = imageStitchData;
   int error = pthread_create(&(tid[i]), NULL, WarpandStitchImages, (void *) &imagedataVector[i]);
   if(error != 0)
   {
      throw "thread not created";
   }

  }
   for(i=0; i < images.size(); i++)
    {
     pthread_join(tid[i], NULL);
   }
    return outputImage;
   }




