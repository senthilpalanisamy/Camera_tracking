// Author: Senthil Palanisamy
// A file describing some utility functions used across all the other scripts

#include <pthread.h>
#include <fstream>
#include <iostream>

#include <sys/types.h> 
#include <sys/stat.h> 
#include <unistd.h> 
#include<stdlib.h>

#include "opencv2/calib3d/calib3d.hpp"
#include <jsoncpp/json/json.h>

#include "utility_functions.hpp"

using std::to_string;
using std::pow;
using std::ifstream;
using std::cin;
using std::make_tuple;
pthread_mutex_t lock;


cameraCellAssociator::cameraCellAssociator(string fileName)
  // CellAssociator call associates a given mice position with a cell in which it exits
  // fileName- filePath where the cellAssociation config file is stored
{

   //ifstream myfile("./config/camera0.txt");
   ifstream myfile(fileName);
   int pixelx, pixely, cellx, celly;

    while(myfile >> pixelx >> pixely >> cellx >> celly)
   {
     cell_centers.push_back({pixelx, pixely});
     cell_index.push_back({cellx, celly});
   }

}


vector<int> cameraCellAssociator::return_closest_cell(int mice_x, int mice_y)
  // Given a x,y mice location, this function returns the cell which contains the given x,y
  // pixel location
  {
    vector<double> distances(cell_centers.size());

    for(int i=0; i < cell_centers.size(); ++i)
    {
      auto point = cell_centers[i];
      distances[i] = pow(pow(point[0] - mice_x, 2)+
                     pow(point[1] - mice_y, 2), 0.5);
    }
    int index = min_element(distances.begin(), distances.end()) - distances.begin();
    auto cell = cell_index[index];
    return cell;
  }

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




  Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography)
  // This function transforms the given image to the common stitched image view using
  // the given homography and transfers pixels from the trasnformed image to the common
  // stitched image
  {

    Mat imageUnwarped;
    size_t i=0, j=0;

    Size warpedImageSize = stitchedImage.size();

    // warpPerspective (ipImage, imageUnwarped, Homography, warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);
    warpPerspective (ipImage, imageUnwarped, Homography, warpedImageSize,INTER_LINEAR);

    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, stitchedImage);

    // cout<<"starting parallel for";
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
    // cout<<"ending parallel for\n";

    #ifdef DEBUG
    namedWindow("StitchedImage",WINDOW_NORMAL);
    resizeWindow("StitchedImage", 600, 600);
    imshow("StitchedImage", stitchedImage);


    namedWindow("warpedImage",WINDOW_NORMAL);
    resizeWindow("warpedImage", 600, 600);
    imshow("warpedImage", imageUnwarped);
    waitKey(0);
    #endif

    return stitchedImage;

  }

 tuple<vector<double>, Mat> readCameraParameters(string jsonFilePath)
  // A function for reading camera parameres from the given path
  // Returns
  // A tuple containing lens distortion parameters and camera intrinsic parameters
 {

  vector<double> distCoeffs(5);
  cv::Mat cameraMatrix(3, 3, CV_64F);

  std::ifstream cameraParametersFile(jsonFilePath, std::ifstream::binary);
  Json::Value cameraParameters;


  cameraParametersFile >> cameraParameters;
  auto intrinsic = cameraParameters["intrinsic"];

  for(int i=0; i < 3; ++i)
   {
     for(int j=0; j < 3; ++j )
     {
       cameraMatrix.at<double>(i, j)= cameraParameters["intrinsic"][i][j].asDouble();
     }
   }
  for(int i=0; i < distCoeffs.size(); ++i)
  {
    distCoeffs[i] = cameraParameters["dist"][0][i].asDouble();

  }

  return make_tuple(distCoeffs, cameraMatrix);
 }

 void performLensCorrection(Mat& image, int imageNo, string lensCorrectionFolderPath)
  // This function performs lens distortion correction for the given image
  // image - this is an input output parameter, which contains the input image and this
  //         image will get replaced by the transformed image
  // imageNo - cameraNo from which the image was captured. This information is critical
  //           since we only send the folder path containing lens distortion parameters 
  //           of all cameras and hence, this number is used to read appropriate files
  // lensCorrectionFolderPath - path containing json files of camera intrinsics and 
  //                            lens distortion coefficients
  {

  cv::Size imageSize(cv::Size(image.cols,image.rows));

  string json_file_path = lensCorrectionFolderPath + "/" +"camera_"+ to_string(imageNo)+".json";
  auto cameraParameters = readCameraParameters(json_file_path);
  auto cameraMatrix = std::get<1>(cameraParameters);
  auto distCoeffs = std::get<0>(cameraParameters);

  // Refining the camera matrix using parameters obtained by calibration
  auto new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, 0);

  // Method 1 to undistort the image
  cv::Mat dst;
  cv::undistort( image, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
  image = dst;
  }

int getMaxAreaContourId(vector <vector<cv::Point>> contours, Point2f robotPosition) 
 // This function returns the id of the contour that has the maximum area in the image
 // Note: The maximum contour area has to greater than 600 or else -1 is returned.
 // This 600 is chosen in accordance with the minimum mice movement observed so that 
 // this function returns the contour ID containing the mice in the background subtracted 
 // image
{
    double maxArea = 0;
    int nearnessThreshold=200;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {

        auto M = moments(contours[j]);
        int cx = int(M.m10 / M.m00);
        int cy = int(M.m01 / M.m00);

	if(pow(pow(cx-robotPosition.x, 2) + pow(cy - robotPosition.y, 2), 0.5) < nearnessThreshold)
	{
          continue;
	}
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        }
    }

    if(maxArea > 600)
    {
      return maxAreaContourId;
    }
    else
    {
      return -1;
    }
}

string return_date_header()
 // Returns the date header. The format is mm_dd
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday); 
   return date_header;
		             
}

string getFolderPath()
 // A function for getting a unique folder path. This functions keeps insisting for
 // an unique name until it is entered
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
   return outputPath;
}

string return_date_time_header()
 // Returns a date_time stamp
 // The format is MM_DD_HH:MM:SS(month_day_hour:min:seconds)
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_time_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday) 
	                     + "_" + to_string(5+ltm->tm_hour) + ":" + to_string(30+ltm->tm_min) + ":" +
		             to_string(ltm->tm_sec);
}








