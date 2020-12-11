// Author: Senthil Palanisamy
// File defining the image stitching class
// This class constructs a stitched image given the input raw images
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

// TODO: Remove this duplicate code and use this code from utilities function

class ParallelPixelTransfer: public ParallelLoopBody
// This is a class for parallelising the pixel transfer mechanism when constructing a 
// stitched image (When constructing a stitched image, pixels from the raw image need to 
// be transferred to the stitched image)
// For more information visit https://docs.opencv.org/master/d7/dff/tutorial_how_to_use_OpenCV_parallel_for_.html
{
  mutable Mat unwarpedImage;
  mutable Mat stitchedImage;
  public:
    ParallelPixelTransfer(Mat& srcPtr, Mat& dstPtr)
    {
      unwarpedImage = srcPtr;
      stitchedImage = dstPtr;
    }

    virtual void operator() (const Range& range) const CV_OVERRIDE
    // The row and column count are combined into a signle continuous number and this
    // "range" number is broken down into row and column index in the loop for parallel
    // pixel transfer
    {
      int r=0;
      for(r = range.start; r < range.end; r++)
      {
        int i = r / unwarpedImage.cols;
        int j = r % unwarpedImage.cols;

          if(stitchedImage.ptr<uchar>(i)[j] == 0)
          {
            stitchedImage.ptr<uchar>(i)[j] = unwarpedImage.ptr<uchar>(i)[j];
          }
          }
     }

    ParallelPixelTransfer& operator=(const ParallelPixelTransfer&)
    {
      return *this;
    }

};




imageStitcher::imageStitcher(string configPath, bool doLensCorrection_,
		string lensCorrectionFolderPath_)
  // Constructor for image stitching
  // configPath - folder path which contains all the pre-calibrated camera homographies
  //              This is the path where calibrate_homographies scripts stores the results
  // doLensCorrection_ - should lens correction be performed inside this image stitching class
  //                     If lens correction is performed before sending the image into this
  //                     class, this should be set to false. If it is not perfomed before
  //                     sending the images, this should be set to True
  // lesCorrectionFolderPath_ - Path where json files describing lens distortion for each
  //                            camera is placed
  // Returns
  // Stiched Image
  {

   vector<string> fileNames = {"camera1_to_ground_plane.txt", "camera2_to_ground_plane.txt", 
	                       "camera3_to_ground_plane.txt", "camera4_to_ground_plane.txt"};
   doLensCorrection = doLensCorrection_;
   lensCorrectionFolderPath = lensCorrectionFolderPath_;

   double number;

   // reading all homographies files
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

      // Reading size of the stitched image
     string imageSizePath = configPath + "image_size.txt";
     std::ifstream infile(imageSizePath);
     infile >> finalSize.width;
     infile >> finalSize.height;
  }


  Mat imageStitcher::stitchImagesOnline(vector<Mat> images)
  // Constructs a sitched image for the given images using pre-calibrated homopgrahies
  {

   Mat dstImage(finalSize, CV_8U, Scalar(0));
   //outputImage = dstImage;
   size_t i;

   if(doLensCorrection)
   {
	   
     for(int i=0; i < 4; ++i)
     {
       // TODO: This is a bad structure setup. For every image to be stitched, json 
       // files are read everytime for getting lens distortion parameters. This is 
       // very slow since hard disk access is very cost and hence, slows down the 
       // speed of image stitching. Read the lens distortion parameters once inside the
       // constructors and pass the read parameters to avoid repeated work
      performLensCorrection(images[i], i, lensCorrectionFolderPath);
     }
   }

  

   for(i=0; i < images.size(); i++)
   {

    Mat imageUnwarped;

    warpPerspective (images[i], imageUnwarped, allHomographies[i], finalSize, INTER_LINEAR);
    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, dstImage);
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
   }
    return dstImage;
   }




