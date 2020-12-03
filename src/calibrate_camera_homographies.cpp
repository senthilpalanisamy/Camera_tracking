#include <iostream>
#include <fstream>
#include <limits>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "simple_capture.hpp"
#include "utility_functions.hpp"



# define DEBUG 0

using namespace cv;
using std::vector;
using cv::detail::MatchesInfo;
using std::cout;
using std::endl;



std::ofstream openFile(string inputFilePath)
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

  vector<Point2f> initialiseChessBoardPoints(Size opSize, int boardWidth=9, 
                                             int boardHeight=7)
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

  imageStitcher(vector<Mat> images, string opPath="./config/camera_homographies")
  {
   size_t i, j;
   size_t opimgWidth = images[0].cols * 3, opimgheight = images[0].rows * 3;
   opSize = Size(opimgWidth, opimgheight);
   chessboardDims.width = 9;
   chessboardDims.height = 6;
   pointsMapping = initialiseChessBoardPoints(opSize, chessboardDims.width, 
		                             chessboardDims.height);


   min_x = std::numeric_limits<int>::max();
   max_x = std::numeric_limits<int>::min();
   min_y = std::numeric_limits<int>::max();
   max_y = std::numeric_limits<int>::min();


   Mat stitchedImage(opSize, CV_8UC1, Scalar(0));

   string command = "mkdir -p "+ opPath;
   system(command.c_str());


   for(i=0; i < images.size(); i++)
   {

   imshow("image", images[i]);
   waitKey(0);

   Mat ipImage = images[i];
   Mat homography = computeHomographyChessBoard(ipImage);
   allHomographies.push_back(homography);
   stitchedImage = stitchImageschessBoard(stitchedImage, ipImage, homography);
   }


   getbiggestBoundingboxImage(stitchedImage);


    for(auto& h:allHomographies)
    {

    // h = h.inv();
    double data[9] = {1, 0, -(double) min_x, 0, 1, -(double) min_y, 0, 0, 1};
    Mat trans = Mat(3, 3, CV_64F, data); 
    //double scale  = h.at<double>(2,2) - h.at<double>(2,0);
    //h.at<double>(0,2) = h.at<double>(0,2) - min_x;
    //h.at<double>(1,2) = h.at<double>(1,2) - min_y;
    h = trans * h;

    //h.at<double>(0,2) = h.at<double>(0,2) - min_x;
    //h.at<double>(1,2) = h.at<double>(1,2) - min_y;
    // h = h.inv();

    }
   finalSize = opSize;
   vector<string> fileNames = {"camera1_to_ground_plane.txt", "camera2_to_ground_plane.txt", 
	                       "camera3_to_ground_plane.txt", "camera4_to_ground_plane.txt"};
   for(auto fileName:fileNames)
   {
     string inputFilePath = opPath + "/" + fileName;
     auto inFile = openFile(inputFilePath);

      for(int i=0; i < 3; ++i)
      {
	 for(int j=0; j < 3; ++j)
	 {
           inFile<<allHomographies[i].at<double>(i, j)<<endl;
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


  Mat computeHomographyChessBoard(Mat image)
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

       //H = findHomography( pointsMapping, detectedPoints, RANSAC);
       // H = findHomography( detectedPoints, pointsMapping, RANSAC, 1);
       H = findHomography( detectedPoints, pointsMapping, LMEDS);

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
};

int main(void)
{
  cout<<"started capture";
  // frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");

  frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", true,
  	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image3);



  // images.push_back(dst1);
  imageStitcher imgStitcher(images);

 } 
