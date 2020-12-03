#include <limits>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/calib3d/calib3d.hpp"


#include "simple_capture.hpp"
#include "utility_functions.hpp"



# define DEBUG 0

using std::to_string;



using namespace cv;
using std::vector;
//using cv::detail::MatchesInfo;
using std::cout;
using namespace std::chrono;
using std::endl;

Mat outputImage;






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

  imageStitcher(vector<Mat> images)
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


   for(i=0; i < images.size(); i++)
   {

   imshow("image", images[i]);
   waitKey(0);

   Mat ipImage = images[i];
   Mat homography = computeHomographyChessBoard(ipImage);
   allHomographies.push_back(homography);
   stitchedImage = stitchImageschessBoard(stitchedImage, ipImage, homography);
   }


   //getbiggestBoundingboxImage(stitchedImage);
   finalSize = opSize;




   // for(auto& h:allHomographies)
   // {

   // // h = h.inv();
   // double data[9] = {1, 0, -(double) min_x, 0, 1, -(double) min_y, 0, 0, 1};
   // Mat trans = Mat(3, 3, CV_64F, data); 
   // //double scale  = h.at<double>(2,2) - h.at<double>(2,0);
   // //h.at<double>(0,2) = h.at<double>(0,2) - min_x;
   // //h.at<double>(1,2) = h.at<double>(1,2) - min_y;
   // h = trans * h;

   // //h.at<double>(0,2) = h.at<double>(0,2) - min_x;
   // //h.at<double>(1,2) = h.at<double>(1,2) - min_y;
   // // h = h.inv();

   // }
  }


  Mat stitchImagesOnline(vector<Mat> images)
  {

   Mat dstImage(finalSize, CV_8U, Scalar(0));
   outputImage = dstImage;

   size_t i;

   pthread_t tid[images.size()];
   ImageStitchData imagedataVector[4];

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


  private:
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

  frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt", true,
  	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image3);



  // images.push_back(dst1);
  cout<<"ended capture";
  imageStitcher imgStitcher(images);
  Mat stitchedImage;

  //namedWindow("stitchedImageop",WINDOW_NORMAL);
  //resizeWindow("stitchedImageop", 600, 600);
  cout<<"OPENCV version"<<CV_VERSION;
  cout<<"Major version"<<CV_MAJOR_VERSION;
  cout<<"\nBuild Information:"<<getBuildInformation();
  //imageTransferObj.displayAllImages();

   while(true)
   {
     auto start = high_resolution_clock::now();
     imageTransferObj.transferAllImagestoPC();
     auto stop = high_resolution_clock::now();
     auto duration = duration_cast<milliseconds>(stop - start);
     cout<<"image acquisition time:"<<duration.count()<<endl;
     vector<Mat> images;
     // images.push_back(imageTransferObj.image0);


     images.push_back(imageTransferObj.image0);
     images.push_back(imageTransferObj.image1);
     images.push_back(imageTransferObj.image2);
     images.push_back(imageTransferObj.image3);

     
      // for new setup
      // double angle = 180;

      // // get rotation matrix for rotating the image around its center in pixel coordinates
      // Mat src = imageTransferObj.image3;
      // cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
      // cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
      // // determine bounding rectangle, center not relevant
      // cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
      // // adjust transformation matrix
      // rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
      // rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

      // cv::Mat dst;
      // cv::warpAffine(src, dst, rot, bbox.size()); 
      // //  

      // images.push_back(dst);

     cout<<"\nstitching image\n";
     start = high_resolution_clock::now();
     stitchedImage = imgStitcher.stitchImagesOnline(images);
     stop = high_resolution_clock::now();
     duration = duration_cast<milliseconds>(stop - start);
     cout<<"Image stitching time:"<<duration.count()<<endl;
     imshow("stitchedImageop", stitchedImage);
     if(waitKey(1) >= 0)
       break;
     //imageTransferObj.displayAllImages();

     imwrite("outputimage.png", stitchedImage);

 } 

}
