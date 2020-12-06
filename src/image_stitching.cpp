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

class ParallelPixelTransfer: public ParallelLoopBody
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
   //outputImage = dstImage;
   size_t i;

   if(doLensCorrection)
   {
	   
     for(int i=0; i < 4; ++i)
     {
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




