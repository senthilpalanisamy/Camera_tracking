#include <pthread.h>
#include <fstream>

#include "opencv2/calib3d/calib3d.hpp"
#include <jsoncpp/json/json.h>

#include "utility_functions.hpp"
using std::to_string;



pthread_mutex_t lock;

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

 void performLensCorrection(Mat& image, int imageNo, string lensCorrectionFolderPath)
  {

  cv::Size imageSize(cv::Size(image.cols,image.rows));
  vector<double> distCoeffs(5);
  cv::Mat cameraMatrix(3, 3, CV_64F);

  string json_file_path = lensCorrectionFolderPath + "/" +"camera_"+ to_string(imageNo)+".json";
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

  // Refining the camera matrix using parameters obtained by calibration
  auto new_camera_matrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 0, imageSize, 0);

  // Method 1 to undistort the image
  cv::Mat dst;
  cv::undistort( image, dst, new_camera_matrix, distCoeffs, new_camera_matrix );
  image = dst;
  }






