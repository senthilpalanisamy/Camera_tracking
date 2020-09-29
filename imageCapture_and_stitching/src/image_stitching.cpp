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


//# define DEBUG 0
//
//

constexpr int CAMERA_COUNT=4;



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
        // cout<<"\n image width:"<<unwarpedImage.cols<<" image height"<<unwarpedImage.rows<<"\n";
        // cout<<"\n"<<range.start<<" "<<range.end<<" "<<r<<" "<<i<<" "<<j<<"\n";

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
    //Size warpedImageSize = stitchArgs->dstImage.size();
    // temp test
    // Size warpedImageSize = Size(3260, 2811);
    cout<<"\ndst size"<<warpedImageSize<<"\n";
    cout<<"\n input image size"<<stitchArgs->inputImage.size();
    cout<<"\n homography size"<<stitchArgs->homography.size();

    warpPerspective (stitchArgs->inputImage, imageUnwarped, stitchArgs->homography,
                     warpedImageSize,INTER_LINEAR + WARP_INVERSE_MAP);
    cout<<"warped image , transferring pixels";

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
     //cout<<"image no"<<i<<"\n";

   Mat ipImage = images[i];
   Mat homography = computeHomographyChessBoard(ipImage);
   allHomographies.push_back(homography);
   stitchedImage = stitchImageschessBoard(stitchedImage, ipImage, homography);


   //imwrite("image"+std::to_string(i)+".jpg", ipImage);

   // imwrite("result.jpg", stitchedImage);
   //imshow("stitchedImage", stitchedImage);
   //imshow("newImage", images[i+1]);
   //waitKey(0);
   }
   //imshow("stitchedImage", stitchedImage);
   //waitKey(0);
   //cout<<"out of loop";
   Mat output;
   output = getbiggestBoundingboxImage(stitchedImage);

   //float w_diff = opSize.width / 2.0 - finalSize.width / 2.0;
   //float h_diff = opSize.height / 2.0 - finalSize.height / 2.0;

   //Mat Ht(Size(3,3), CV_64F, Scalar(0));
   //Ht.at<double>(0,0) = 1.0;
   //Ht.at<double>(1,1) = 1.0;
   //Ht.at<double>(2,2) = 1.0;
   //Ht.at<double>(0,2) = -w_diff;
   //Ht.at<double>(1,2) = -h_diff;

   for(auto& h:allHomographies)
   {
   //  h = Ht * h;

   double scale  = h.at<double>(2,2);
   //h.at<double>(0,0) = h.at<double>(0,0) / scale;
   //h.at<double>(0,1) = h.at<double>(0,1) / scale;
   h.at<float>(0,2) = h.at<float>(0,2)  - scale * min_x;
   //h.at<double>(1,0) = h.at<double>(1,0) / scale;
   //h.at<double>(1,1) = h.at<double>(1,1) / scale;
   h.at<double>(1,2) = h.at<double>(1,2) - scale * min_y;
   //h.at<float>(1,2) = h.at<float>(1,2) / scale - h_diff;

   }
   //imwrite("result.jpg", output);
  }


  Mat stitchImagesOnline(vector<Mat> images)
  {
    cout<<"\nhere\n";

   Mat dstImage(finalSize, CV_8U, Scalar(0));
   outputImage = dstImage;
   //Mat outputImage(finalSize, CV_8U, Scalar(0));


   size_t i;

   pthread_t tid[CAMERA_COUNT];
   cout<<"\nOnline stitching stated\n";
   ImageStitchData imagedataVector[4];

   for(i=0; i < images.size(); i++)
   {
   //Mat homography = allHomographies[i];
   struct ImageStitchData imageStitchData;
   imageStitchData.dstImage = outputImage;
   imageStitchData.homography = allHomographies[i];
   imageStitchData.inputImage = images[i];
   imagedataVector[i] = imageStitchData;
   // cout<<"size  :"<<imageStitchData.inputImage.size();
   int error = pthread_create(&(tid[i]), NULL, WarpandStitchImages, (void *) &imagedataVector[i]);
   if(error != 0)
   {
     cout<<"\nthread not created\n";
      throw "thread not created";
   }

  }
   // cout<<"out of loop";

   //outputImage = getbiggestBoundingboxImage(outputImage);

   //imwrite("stitchedImage.jpg", outputImage);
   for(i=0; i < images.size(); i++)
    {
     // cout<<"joining thread";
     pthread_join(tid[i], NULL);
   }

   cout<<"\n joined threads\n";


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

     //for(i=0; i < imageUnwarped.size().height; i++)
     // {
     //   uchar * imageUnwarpedPtr = imageUnwarped.ptr(i);
     //   uchar * stitchedImagePtr = stitchedImage.ptr(i);
     //    for(j=0; j < imageUnwarped.size().width;j++)
     //    {
     //      // May be a weak check
     //      if(stitchedImagePtr[j]==0)
     //      {
     //       stitchedImagePtr[j] = imageUnwarpedPtr[j];
     //      }
     //      }
     // }

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

  Mat computeHomography(Mat image1, Mat image2)
  {
   vector<Mat> images;
   images.push_back(image1);
   images.push_back(image2);


  vector<Mat> descriptors;
  size_t i=0;
  for(;i<2; i++)
  {
    Mat img_descriptor;
    descriptors.push_back(img_descriptor);
  }
  vector<vector<KeyPoint>> keypoints;

  for(i=0;i<2; i++)
  {
    vector<KeyPoint> imgKeypoint;
    keypoints.push_back(imgKeypoint);
  }
  vector<MatchesInfo> pairwise_matches;

  //auto detector = ORB::create();
  auto detector = SIFT::create();
  size_t num_images = 2;

  for(i=0; i < 2; i++)
   {
    detector->detectAndCompute( images[i], cv::noArray(), keypoints[i], descriptors[i] );
   } 
   BFMatcher brute_force_matcher = cv::BFMatcher(NORM_L2, true);
   vector< cv::DMatch > matches;
   brute_force_matcher.match(descriptors[0], descriptors[1], matches);
   const float ratio_thresh = 0.7f;

    // debugging
    #ifdef DEBUG

    namedWindow("matched_image",WINDOW_NORMAL);
    resizeWindow("matched_image", 600, 600);
    Mat img_matches;
    drawMatches( images[0], keypoints[0], images[1], keypoints[1], matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("matched_image", img_matches);
    //imwrite("mathced_image.jpg", img_matches);
    waitKey(0);
    #endif

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints[0][matches[i].queryIdx].pt);
        scene.push_back(keypoints[1][matches[i].trainIdx].pt);
    }
    Mat H = findHomography( obj, scene, RANSAC );
    return H;

  }




};

int main(void)
{
  //Mat img1 = cv::imread("../data/boat1.jpg", 0);
  //Mat img2 = cv::imread("../data/boat2.jpg", 0);
  //Mat img3 = cv::imread("../data/boat3.jpg", 0);
  //Mat img4 = cv::imread("../data/boat4.jpg", 0);
  //Mat img5 = cv::imread("../data/boat5.jpg", 0);
  //Mat img6 = cv::imread("../data/boat6.jpg", 0);
  //Mat img1 = cv::imread("../data/newspaper1.jpg", 0);
  //Mat img2 = cv::imread("../data/newspaper2.jpg", 0);
  //Mat img3 = cv::imread("../data/newspaper3.jpg", 0);
  //Mat img4 = cv::imread("../data/newspaper4.jpg", 0);
  cout<<"started capture";
  frameGrabber imageTransferObj("../data/camera2.fmt");
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image3);
  cout<<"ended capture";
  //images.push_back(img3);
  //images.push_back(img4);
  //images.push_back(img5);
  //images.push_back(img6);
  imageStitcher imgStitcher(images);
  Mat stitchedImage;

  //namedWindow("stitchedImageop",WINDOW_NORMAL);
  //resizeWindow("stitchedImageop", 600, 600);
  cout<<"OPENCV version"<<CV_VERSION;
  cout<<"Major version"<<CV_MAJOR_VERSION;
  cout<<"\nBuild Information:"<<getBuildInformation();
  while(true)
  {
    auto start = high_resolution_clock::now();
    imageTransferObj.transferAllImagestoPC();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    cout<<"image acquisition time:"<<duration.count()<<endl;
    vector<Mat> images;
    images.push_back(imageTransferObj.image0);
    images.push_back(imageTransferObj.image2);
    images.push_back(imageTransferObj.image1);
    images.push_back(imageTransferObj.image3);

    cout<<"\nstitching image\n";
    start = high_resolution_clock::now();
    stitchedImage = imgStitcher.stitchImagesOnline(images);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    cout<<"Image stitching time:"<<duration.count()<<endl;
    imshow("stitchedImageop", stitchedImage);
    if(waitKey(1) >= 0)
      break;
    imageTransferObj.displayAllImages();
    imwrite("outputimage.png", stitchedImage);
    imwrite("image1.png", images[0]);
    imwrite("image2.png", images[1]);
    imwrite("image3.png", images[2]);
    imwrite("image4.png", images[3]);
}

}
