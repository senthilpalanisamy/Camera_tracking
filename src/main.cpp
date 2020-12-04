#include <vector>
#include <fstream>
#include <string>
#include <iostream>

#include "backGroundSubtraction.hpp"
#include "utility_functions.hpp"
#include "simple_capture.hpp"
#include "video_recorder.hpp"

using namespace std::chrono;
using std::vector;
using std::cout;
using std::ifstream;
using std::cout;
using std::string;
using std::to_string;
using cv::Size;
using cv::Point;
using cv::Scalar;



int main()
{
  int cameraCount = 4;
  string experimentPath = getFolderPath();

  vector<cameraCellAssociator> cellAssociation;
  cellAssociation.emplace_back("./config/cell_association/camera0.txt");
  cellAssociation.emplace_back("./config/cell_association/camera1.txt");
  cellAssociation.emplace_back("./config/cell_association/camera2.txt");
  cellAssociation.emplace_back("./config/cell_association/camera3.txt");

  frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt");


  //VideoCapture cap(input_video);

  vector<Mat> frame;
  Method method=MOG2;

  vector<BackGroundSubtractor> bgsubs;
  vector<Mat> frames;

  imageTransferObj.transferAllImagestoPC();

  frames.push_back(imageTransferObj.image0);
  frames.push_back(imageTransferObj.image1);
  frames.push_back(imageTransferObj.image2);
  frames.push_back(imageTransferObj.image3);

  string homographyConfigPath = "./config/camera_homographies/";
  Size stitchedImageSize;

   string imageSizePath = homographyConfigPath + "image_size.txt";
   std::ifstream infile(imageSizePath);
   infile >> stitchedImageSize.width;
   infile >> stitchedImageSize.height;




  // imageStitcher imgStitcher(homographyConfig, true,
  //		           "./config/camera_intrinsics_1024x1024");

  for(int i=0; i<cameraCount; ++i)
  {
    bgsubs.emplace_back(method, frames[i], false);
  }

  string rawVideoPath = experimentPath + "/" + "unprocessed"; 


  auto rawVideoRecorder = videoRecorder(4, "bg_output", frames[0].size(), 10, false,
		                rawVideoPath, true);

  auto stitchedVideorecorder = videoRecorder(1, "stitched_output", stitchedImageSize, 10, false,
		                rawVideoPath, true);



  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  while(true)
  {

    auto start = high_resolution_clock::now();
    vector<Mat> foregroundImages;

    imageTransferObj.transferAllImagestoPC();

    frames[0] = imageTransferObj.image0;
    frames[1] = imageTransferObj.image1;
    frames[2] = imageTransferObj.image2;
    frames[3] = imageTransferObj.image3;



    for(int i=0; i < cameraCount; ++i)
    {

      auto foregroundImage = bgsubs[i].processImage(frames[i]);

      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

      int maxContourId = getMaxAreaContourId(contours), cx, cy;

      if(maxContourId >= 0)
      {
        auto M = moments(contours[maxContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(frames[i], cv::Point(cx , cy), 30, cv::Scalar(255), -1);
	auto associatedCell  = cellAssociation[i].return_closest_cell(cx, cy);
	putText(frames[i], to_string(associatedCell[0]) + ","+ to_string(associatedCell[1]),
		Point(cx, cy-10), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(0, 255, 0), 2);
      }

      //foregroundImages.push_back(move(foregroundImage));


    }

    rawVideoRecorder.writeFrames(frames);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\ntime: "<<duration.count() << endl;
    cout << "\nfps: "<<1.0 / (duration.count() /1000.0 / 1000.0) << endl;


    imshow("image", frames[0]);
    char c = waitKey(2);

    if(c == 27 || c == 10)
	break;
    // Mat frame2 = frame.clone();
    cout<<"here";
  }

  return 0;



}
