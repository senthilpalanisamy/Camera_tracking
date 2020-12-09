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

  void* WarpandStitchImages(void *arguments)
  {
    cout<<"\nthread started\n";

    Mat imageUnwarped;
    ImageStitchData *stitchArgs = (ImageStitchData*) arguments;

    pthread_mutex_lock(&lock);

    Size warpedImageSize = stitchArgs->dstImage.size();
    // warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
    //                  warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);

    warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
                     warpedImageSize, INTER_LINEAR);

    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, stitchArgs->dstImage);
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
    pthread_mutex_unlock(&lock);

    return NULL;
    }



  Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography)
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
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday); 
   return date_header;
		             
}

string getFolderPath()
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
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_time_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday) 
	                     + "_" + to_string(5+ltm->tm_hour) + ":" + to_string(30+ltm->tm_min) + ":" +
		             to_string(ltm->tm_sec);
}








