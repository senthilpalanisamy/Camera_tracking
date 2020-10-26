#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <limits>

#include "simple_capture.hpp"
#include <iostream>
#include <pthread.h>

//#include "tbb/tbb.h"

//using tbb::parallel_for;


// # define DEBUG 0
//
//
using std::to_string;



using namespace cv;
using std::vector;
using namespace cv::xfeatures2d;
using cv::detail::MatchesInfo;
using std::cout;
using namespace std::chrono;
using std::endl;

pthread_mutex_t lock;

Mat outputImage;


struct ImageStitchData
{
  Mat dstImage;
  Mat inputImage;
  Mat homography;
};


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

          // May be a weak check
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
    warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
                     warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);
    ParallelPixelTransfer parellelPixelTransfer(imageUnwarped, stitchArgs->dstImage);
    parallel_for_(Range(0, imageUnwarped.rows * imageUnwarped.cols), parellelPixelTransfer);
    pthread_mutex_unlock(&lock);

    return NULL;
    }




class imageStitcher
{
  public:

  vector<Point2f> pointsMapping;
  Size opSize;
  Size finalSize;
  vector<Mat> allHomographies;
  int min_x, max_x, min_y, max_y;

  imageStitcher(vector<Mat> images)
  {
   size_t i, j;
   float squareSize = 30;
   int boardLength = 9, boardWidth = 6;
   int opimgWidth = images[0].cols * 2, opimgheight = images[0].rows * 2;
   opSize = Size(opimgWidth, opimgheight);
   float chessCenterx = opimgWidth / 2.0 - boardLength / 2.0 * squareSize;
   float chessCenterY = opimgheight / 2.0 - boardWidth/ 2.0 * squareSize;


   min_x = std::numeric_limits<int>::max();
   max_x = std::numeric_limits<int>::min();
   min_y = std::numeric_limits<int>::max();
   max_y = std::numeric_limits<int>::min();

   for(i=0; i < boardLength; i++)
   {
     for(j=0; j< boardWidth; j++)
     {
       Point2f boardPoint;
       boardPoint.x = chessCenterx + i * squareSize;
       boardPoint.y = chessCenterY + j * squareSize;
       pointsMapping.push_back(boardPoint);

     }
   }

   Mat stitchedImage(opSize, CV_8UC1, Scalar(0));


   for(i=0; i < images.size(); i++)
   {

   Mat ipImage = images[i];
   Mat homography = computeHomographyChessBoard(ipImage);
   allHomographies.push_back(homography);
   stitchedImage = stitchImageschessBoard(stitchedImage, ipImage, homography);
   }
   Mat output;
   output = getbiggestBoundingboxImage(stitchedImage);


   for(auto& h:allHomographies)
   {
     h = h.inv();
   double scale  = h.at<double>(2,2);
   h.at<float>(0,2) = h.at<float>(0,2)  - scale * min_x;
   h.at<double>(1,2) = h.at<double>(1,2) - scale * min_y;
   h = h.inv();

   }
  }


  Mat stitchImagesOnline(vector<Mat> images)
  {
    cout<<"\nhere\n";

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
     cout<<"\nthread not created\n";
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
  Mat getbiggestBoundingboxImage(Mat image)
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

    Mat finalImage(max_y - min_y, max_x- min_x, CV_8U);
    //Mat finalImage(max_x - min_x, max_y- min_y, CV_8U);
    //finalSize = Size(max_x-min_x, max_y-min_y);
    finalSize = finalImage.size();
    for(i=0; i < finalImage.size().height; i++)
    {
      for(j=0; j< finalImage.size().width; j++)
      {
        finalImage.at<char>(i,j) = image.at<char>(min_y+i, min_x + j);
      }
    }
    return finalImage;


  }
  Mat computeHomographyChessBoard(Mat image)
  {
    bool found;
    vector<Point2f> detectedPoints;
    int winSize = 11;
    Mat H;
    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
    found = findChessboardCorners( image, Size(9,6), detectedPoints, chessBoardFlags);
    if ( found )                // If done with success,
    {
       cornerSubPix( image, detectedPoints, Size(winSize,winSize),
                                   Size(-1,-1), TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001  ));
       drawChessboardCorners( image, Size(9,6), Mat(detectedPoints), found  );

       H = findHomography( pointsMapping, detectedPoints, RANSAC);
       // look at this once more
       // H = findHomography( detectedPoints, pointsMapping, RANSAC);

       #ifdef DEBUG
       namedWindow("image",WINDOW_NORMAL);
       resizeWindow("image", 600, 600);
       imshow("image", image);
       waitKey(0);
       #endif
       // H = findHomography( detectedPoints, pointsMapping, 0);
    }
    else
    {
      throw "Chess board not detected";
      // raise error
      //
    //cout<<"status"<<found;
    //return image;
    }

   return H;


  }

  Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography)
  {

    Mat imageUnwarped;
    size_t i=0, j=0;

    Size warpedImageSize = stitchedImage.size();

    warpPerspective (ipImage, imageUnwarped, Homography, warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);

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
  Mat stitchImages(Mat image1, Mat image2, Mat Homography)
   {

    Mat image2_aligned, dst;
    Size imgSize = image1.size();
    //imgSize.width = imgSize.width + image2.size().width; 
    imgSize = imgSize + image2.size();
    int j,i;

    warpPerspective (image2, image2_aligned, Homography, imgSize,INTER_LINEAR + WARP_INVERSE_MAP);
    // check if this logic is little rudimentary
    for(i=0; i < image1.size().height; i++)
     {
        for(j=0; j < image1.size().width;j++)
        {
          if(image1.at<char>(i,j)!=0)
          {
           //image2_aligned(i, j) = images[0](i, j);
           image2_aligned.at<char>(i,j) = image1.at<char>(i,j);
          }
          }
     }

    threshold( image2_aligned, dst, 0, 255,THRESH_BINARY);
    //imshow("stiched", image2_aligned);
    //imshow("thresholded", dst);
    //waitKey(0);

    int min_x = std::numeric_limits<int>::max();
    int max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max();
    int max_y = std::numeric_limits<int>::min();
    for(i=0; i < dst.size().height; i++)
    {
      for(j=0; j < dst.size().width; j++)
      {
        if(dst.at<char>(i,j) !=0)
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

    Mat finalImage(max_y - min_y, max_x- min_x, CV_8U);
    for(i=0; i < finalImage.size().height; i++)
    {
      for(j=0; j< finalImage.size().width; j++)
      {
        finalImage.at<char>(i,j) = image2_aligned.at<char>(min_y+i, min_x + j);
      }
    }

    //imshow("aligned", image2_aligned);
    //imshow("thresholed", dst);
    #ifdef DEBUG
    //namedWindow("WarpedImage",WINDOW_NORMAL);

    namedWindow("WarpedImage",WINDOW_NORMAL);
    namedWindow("StitchedImage",WINDOW_NORMAL);

    resizeWindow("WarpedImage", 600, 600);
    resizeWindow("StitchedImage", 600, 600);

    imshow("WarpedImage", image2_aligned);
    imshow("StitchedImage", finalImage);
    waitKey(0);
    #endif

    return finalImage;
   }

};

int main(void)
{
  cout<<"started capture";
  frameGrabber imageTransferObj("./config/camera2.fmt");
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  //images.push_back(imageTransferObj.image2);
  //images.push_back(imageTransferObj.image1);

  Mat src1 = imageTransferObj.image3;
  double angle1 = 180;
  cv::Point2f center1((src1.cols-1)/2.0, (src1.rows-1)/2.0);
  cv::Mat rot1 = cv::getRotationMatrix2D(center1, angle1, 1.0);
  // determine bounding rectangle, center not relevant
  cv::Rect2f bbox1 = cv::RotatedRect(cv::Point2f(), src1.size(), angle1).boundingRect2f();
  // adjust transformation matrix
  rot1.at<double>(0,2) += bbox1.width/2.0 - src1.cols/2.0;
  rot1.at<double>(1,2) += bbox1.height/2.0 - src1.rows/2.0;

  cv::Mat dst1;
  cv::warpAffine(src1, dst1, rot1, bbox1.size()); 


  images.push_back(dst1);
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
     images.push_back(imageTransferObj.image0);
     //images.push_back(imageTransferObj.image2);
     //images.push_back(imageTransferObj.image1);
     
      // for new setup
      double angle = 180;

      // get rotation matrix for rotating the image around its center in pixel coordinates
      Mat src = imageTransferObj.image3;
      cv::Point2f center((src.cols-1)/2.0, (src.rows-1)/2.0);
      cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
      // determine bounding rectangle, center not relevant
      cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), src.size(), angle).boundingRect2f();
      // adjust transformation matrix
      rot.at<double>(0,2) += bbox.width/2.0 - src.cols/2.0;
      rot.at<double>(1,2) += bbox.height/2.0 - src.rows/2.0;

      cv::Mat dst;
      cv::warpAffine(src, dst, rot, bbox.size()); 
      //  

      images.push_back(dst);

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

     // for(int i=0; i <images.size(); i++) 
     // {
     //  imwrite("image"+to_string(i)+".png", images[i]);

     // }
 } 

}
