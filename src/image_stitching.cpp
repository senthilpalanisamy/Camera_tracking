#include <limits>
#include <iostream>
#include <fstream>

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
  bool doLensCorrection;
  string lensCorrectionFolderPath;


  imageStitcher(string configPath="./config/camera_homographies/", bool doLensCorrection_=true,
		string lensCorrectionFolderPath_="./config/camera_intrinsics_1024x1024")
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


  Mat stitchImagesOnline(vector<Mat> images)
  {

   Mat dstImage(finalSize, CV_8U, Scalar(0));
   outputImage = dstImage;

   size_t i;

   pthread_t tid[images.size()];
   ImageStitchData imagedataVector[4];
   if(doLensCorrection)
   {
	   
     for(int i=0; i < 4; ++i)
     {
      performLensCorrection(images[i], i, lensCorrectionFolderPath);
     }
   }

  

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



};

int main(void)
{
  cout<<"started capture";
  // frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");

  //   frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt", true,
  //   	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");
  // 
  //  frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", true,
  //  	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");

  frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", false);
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image3);



  // images.push_back(dst1);
  cout<<"ended capture";
  imageStitcher imgStitcher;
  Mat stitchedImage;


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
