#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/video.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>

using std::vector;
using std::string;



int main()
{
  cv::Mat image = cv::imread("/home/senthilpalanisamy/work/final_project/Camera_tracking/data/calibration_1024_1024/lens_distortion_selected/camera3.bmp");
  cv::Mat map1, map2;
  cv::Size imageSize(cv::Size(image.cols,image.rows));
  vector<double> distCoeffs(5);
  cv::Mat cameraMatrix(3, 3, CV_64F);
   
  string json_file_path = "/home/senthilpalanisamy/work/final_project/Camera_tracking/config/camera_intrinsics_1024x1024/camera_0.json";
  std::ifstream cameraParametersFile(json_file_path, std::ifstream::binary);
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

  // cameraMatrix = (cv::Mat_<double>(3,3) << intrinsic[0][0], intrinsic[0][1], 
  //                                        intrinsic[0][2], intrinsic[1][0],
  //                                        intrinsic[1][1], intrinsic[1][2], 
  //                                        intrinsic[2][1], intrinsic[2][2],
  //                                        intrinsic[2][2]);
  // 
  // Refining the camera matrix using parameters obtained by calibration
  auto new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0);
  
  // Method 1 to undistort the image
  cv::Mat dst;
  cv::undistort( image, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
  
  // Method 2 to undistort the image
  // cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),imageSize, CV_16SC2, map1, map2);
  // 
  // cv::remap(frame, dst, map1, map2, cv::INTER_LINEAR);
  
  //Displaying the undistorted image

  cv::imshow("original image", image);
  cv::imshow("undistorted image",dst);
  cv::imwrite("undistorted_image.bmp", dst);
  cv::waitKey(0);;




}
