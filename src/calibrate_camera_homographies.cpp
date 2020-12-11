// Author: Senthil Palanisamy
// This file contains code for estimating the camera homographies after a checker board
// or a charuco board has been placed in the field of view of a camera

#include <iostream>
#include <fstream>
#include <limits>
#include <unordered_map>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/aruco/charuco.hpp>
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "simple_capture.hpp"
#include "utility_functions.hpp"

using std::unordered_map;





# define DEBUG 0

using namespace cv;
using std::vector;
using cv::detail::MatchesInfo;
using std::cout;
using std::endl;
using std::to_string;



std::ofstream openFile(string inputFilePath)
 // open a file. Check if a file has been opened correctly or else raise an exception
{

   std::ofstream inFile;

   inFile.open(inputFilePath);
   if (!inFile) 
   {
     cout << "Unable to open file at" + inputFilePath;
     exit(1); // terminate with error
   }
   return inFile;

}




class imageStitcher
{
  public:

  vector<Point2f> pointsMapping;
  Size opSize;
  Size finalSize;
  vector<Mat> allHomographies;
  int min_x, max_x, min_y, max_y;
  Size chessboardDims;
  vector<Mat> cameraMatrixList;
  vector<vector<double>> distCoeffsList;
  unordered_map<int, Point2f> cornerToPoint;

  vector<Point2f> initialiseChessBoardPoints(Size opSize, int boardWidth=9, 
                                             int boardHeight=7)
  // This function initialises manual points for each checker board corner. Each 
  // detected checker board corner needs to associated with a known position in the 
  // stitched image view so that the homography between the raw camera view and the 
  // stitched image view can be calculated
  {

     float squareSize = 30;
     int opImgWidth = opSize.width, opImgHeight = opSize.height;
     float chessCenterx = opImgWidth / 2.0 - boardWidth / 2.0 * squareSize;
     float chessCenterY = opImgHeight / 2.0 - boardHeight / 2.0 * squareSize;
     vector<Point2f> sourcePoints;



     for(int i=boardHeight-1; i >= 0; --i)
     {
       for(int j=boardWidth-1; j >= 0; --j)
       {
         Point2f boardPoint;
         boardPoint.x = chessCenterx + j * squareSize;
         boardPoint.y = chessCenterY + i * squareSize;
         sourcePoints.push_back(boardPoint);

       }
     }
  return sourcePoints;
  }

  imageStitcher(vector<Mat> images, string opPath="./config/camera_homographies",
		string cameraParametersPath="./config/camera_intrinsics_1024x1024")
   // Constructor for estimating camera homographies
   // images - A vector containing all four camera images
   // opPath - path where the estimated homographies should be written
   // cameraParametersPath - Path which contains jsons of camera intrinsic parameters and
   //                        lens distortion parameters
  {

    // Read camera parameters for all cameras
   for(int i=0; i < 4; ++i)
   {

     string json_file_path = cameraParametersPath + "/" +"camera_"+ to_string(i)+".json";
     auto cameraParameters = readCameraParameters(json_file_path);
     cameraMatrixList.push_back(std::get<1>(cameraParameters));
     distCoeffsList.push_back(std::get<0>(cameraParameters));
   }

   size_t i, j;
   // Stitched image size. Keep it so high so that all transformed images are still
   // visible in the stitched image
   size_t opimgWidth = images[0].cols * 6, opimgheight = images[0].rows * 6;
   opSize = Size(opimgWidth, opimgheight);
   chessboardDims.width = 9;
   chessboardDims.height = 6;
   // intialising points for checkerboard
   pointsMapping = initialiseChessBoardPoints(opSize, chessboardDims.width, 
		                             chessboardDims.height);

   // initialisng points for charuco board
   cornerToPoint = generateCornerPositions();


   min_x = std::numeric_limits<int>::max();
   max_x = std::numeric_limits<int>::min();
   min_y = std::numeric_limits<int>::max();
   max_y = std::numeric_limits<int>::min();


   Mat stitchedImage(opSize, CV_8UC1, Scalar(0));

   // create output directory where the homography results should be written
   string command = "mkdir -p "+ opPath;
   auto _ = system(command.c_str());

   // Find homography for all images and contruct a stitched image
   for(i=0; i < images.size(); i++)
   {

   imshow("image", images[i]);
   waitKey(0);

   Mat ipImage = images[i];

   Mat homography = computeHomographyCharucoBoard(ipImage);
   allHomographies.push_back(homography);
   stitchedImage = stitchImageschessBoard(stitchedImage, ipImage, homography);
   }
   imwrite("stitchedImage.png", stitchedImage);


   // Finding the bounding box where the stitched image actually lies
   getbiggestBoundingboxImage(stitchedImage);

   // Adjusting all camera homographies so that all images are transfomed into the 
   // reduced true size image
    for(auto& h:allHomographies)
    {

    // h = h.inv();
    double data[9] = {1, 0, -(double) min_x, 0, 1, -(double) min_y, 0, 0, 1};
    Mat trans = Mat(3, 3, CV_64F, data); 
    h = trans * h;
    }

    // Write the estimated homographies to the results directory
   vector<string> fileNames = {"camera1_to_ground_plane.txt", "camera2_to_ground_plane.txt", 
	                       "camera3_to_ground_plane.txt", "camera4_to_ground_plane.txt"};
   for(int h_index=0; h_index < 4; ++h_index)
   {
     string fileName = fileNames[h_index];
     string inputFilePath = opPath + "/" + fileName;
     auto inFile = openFile(inputFilePath);

      for(int i=0; i < 3; ++i)
      {
	 for(int j=0; j < 3; ++j)
	 {
           inFile<<allHomographies[h_index].at<double>(i, j)<<endl;
	 }
      }
      inFile.close();
   }

   string inputFilePath = opPath + "/" + "image_size.txt";
   auto inFile = openFile(inputFilePath);
   inFile<<finalSize.width<<endl;
   inFile<<finalSize.height;

  }

  void getbiggestBoundingboxImage(const Mat& image)
  // iterate through all pixels in the stitched image and find the min_x, min_y,
  // max_x, max_y where the pixel values are non-zero. These values can be clubbed to
  // form two points: (min_x, min_y), (max_x, max_y), which define the bounding box
  // for the stitched image
  {


    int i, j;
    for(i=0; i < image.size().height; i++)
    {
      for(j=0; j < image.size().width; j++)
      {
        if(image.at<char>(i,j) !=0)
        {
          if(i < min_y)
          {
            min_y = i;
          }
           if(i > max_y)
           {
             max_y = i;
           }

           if(j < min_x)
           {
             min_x = j;
           }

           if(j > max_x)
           {
             max_x = j;
           }
        }

      }

    }
  finalSize.width = max_x - min_x;
  finalSize.height = max_y - min_y;

  }


  Mat computeHomographyChessBoard(Mat& image)
  // Finds homography by detecting a checker board within the given image and uses the 
  // corners to the checker board to associate against corresponding points defined manually
  // image - camera image containing checkerboard image for which homography is to be
  // estimated
  {
    bool found;
    vector<Point2f> detectedPoints;
    int winSize = 11;
    Mat H;
    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    found = findChessboardCorners( image, chessboardDims, detectedPoints, chessBoardFlags);
    if ( found )                // If done with success,
    {
       cornerSubPix( image, detectedPoints, Size(winSize,winSize),
                                   Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001  ));
       drawChessboardCorners( image, chessboardDims, Mat(detectedPoints), found);

       H = findHomography( detectedPoints, pointsMapping, 0);

       #ifdef DEBUG
       namedWindow("image",WINDOW_NORMAL);
       resizeWindow("image", 600, 600);
       imshow("image", image);
       waitKey(0);
       #endif
    }
    else
    {
      throw "Chess board not detected";
    }

   return H;


  }


  unordered_map<int, Point2f> generateCornerPositions()
  // Generates a known corner position for each charuco board corners
  // Returns
  // A map which given a charuco index gives out its true corner position in the stitched 
  // image view
  {
    int id=0;
    int scale = 50;
    int offset = int(opSize.width / 2);
    unordered_map<int, Point2f> cornerToPoint;

   for(int i=0; i < 14; ++i)
   {
     for(int j=0; j < 14; ++j)
     {
	Point2f cornerPoint; 
	cornerPoint.x = i * scale+offset;
	cornerPoint.y = j * scale+offset;
	cornerToPoint[id] = cornerPoint;
        ++id;
     }
  }
   return cornerToPoint;
  }

  Mat computeHomographyCharucoBoard(Mat& image)
  // This function estimates homography for the given image to the ground plan. The input
  // image should contain a partially observable charuco board. It detects charuco
  // points and assocaites each point with a known position to determine the homography
  {

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
    cv::Ptr<cv::aruco::CharucoBoard> board = cv::aruco::CharucoBoard::create(15, 15, 0.06f, 0.04f, dictionary);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_NONE;
    cv::Mat imageCopy;
    imageCopy = image.clone();
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f> > markerCorners;
    cv::aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);
    //or
    //cv::aruco::detectMarkers(image, dictionary, markerCorners, markerIds, params);
    // if at least one marker detected
    //
    std::vector<cv::Point2f> charucoCorners;
    std::vector<int> charucoIds;

    if (markerIds.size() > 0) 
    {
       // cv::aruco::drawDetectedMarkers(imageCopy, markerCorners, markerIds);
       cv::aruco::interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners, charucoIds);
       // if at least one charuco corner detected
       if (charucoIds.size() > 0);
	    cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
     }

     vector<Point2f> dstPoints;

      for(int i=0; i < charucoIds.size(); ++i)
      {
	 int id = charucoIds[i];
	 dstPoints.push_back(cornerToPoint[id]);

      }


      Mat H = findHomography(charucoCorners, dstPoints, 0);
        cv::imshow("out", imageCopy);
        cv::waitKey(0);
      return H;
  }

};

int main(void)
{
  cout<<"started capture";
  // frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");

  // Check if the correct config file is supplied to frame grabber
   frameGrabber imageTransferObj("./config/video_config/room_light_charuco2.fmt", true,
  	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");

  // frameGrabber imageTransferObj("./config/video_config/room_light_charuco.fmt", false);

  // frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", true,
  //	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image3);



  // images.push_back(dst1);
  // Estimates homography for the given images and writes them to the given directory
  imageStitcher imgStitcher(images);

 } 
