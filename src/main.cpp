// Author: Senthil Palanisamy
// This is the main function that lauches all exection
// 1. Does mice tracking using background subtraction
// 2. Does robot tracking by using the black square placed on top of the robot
// 3. Writes raw images acquired from cameras into a video
// 4. Stitched images together ad writes a stitched video

#include <vector>
#include <fstream>
#include <string>
#include <iostream>

#include <opencv2/aruco.hpp>

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

Point2f detectRobotPosition(Mat inputImage)
// Core assumption: the whole maze and everything in the maze is white / bright. Therefore
// the only black square that could exist in the camera's view should belong to the robot.
// The given image is inverse binary thresholded at 30 and the square contour within the 
// thresholded image is found and reported to be the robot
{
    Mat threshold;
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<Point> selectedContour;

    cv::threshold(inputImage, threshold, 30, 255, THRESH_BINARY_INV);
    findContours(threshold, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE );
    int selectedIndex=-1;

    for(int i=0; i < contours.size(); ++i)
     {
	std::vector<cv::Point> approx;
	auto contour = contours[i];
	double area = contourArea(contour);
	if(area < 200 || area > 2000)
	  continue;

	auto peri = cv::arcLength(contour, true);
	cv::approxPolyDP(contour, approx, 0.02 * peri, true);

	if(approx.size() == 4 && fabs(contourArea(Mat(approx))) > 1000 &&
	   isContourConvex(Mat(approx)))
	{
	 selectedContour = contour;
	 selectedIndex = i;
	 cout<<"here";
	}

     }

  if(selectedIndex >= 0)
  {

  auto robotContour = selectedContour;
  auto M = cv::moments(robotContour);
  float cX = float(M.m10 / M.m00);
  float cY = float(M.m01 / M.m00);
  return {cX, cY};

  }
  cout<<"\nnot detected";
  return {-1, -1};
}


int main()
{
  int cameraCount = 4;
  string experimentPath = getFolderPath();

  // Reading all cell association file
  vector<cameraCellAssociator> cellAssociation;
  cellAssociation.emplace_back("./config/cell_association/camera0.txt");
  cellAssociation.emplace_back("./config/cell_association/camera1.txt");
  cellAssociation.emplace_back("./config/cell_association/camera2.txt");
  cellAssociation.emplace_back("./config/cell_association/camera3.txt");

  // check if the camera video config fmt file is correct
  frameGrabber imageTransferObj("./config/video_config/new_white_light.fmt");


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


  // Intialising four background subtractors one for each view of the camera
  for(int i=0; i<cameraCount; ++i)
  {
    bgsubs.emplace_back(method, frames[i], false);
  }

  string rawVideoPath = experimentPath + "/" + "unprocessed"; 

  // Recorder object writing all raw images into videos
  auto rawVideoRecorder = videoRecorder(4, "bg_output", frames[0].size(), 10, false,
		                rawVideoPath, true);
  // Recorder object writing stitched image to a video
  auto stitchedVideorecorder = stitchedVideoRecorder(1, "stitched_output", stitchedImageSize, 10, false,
  		                                     rawVideoPath, true);




  vector<Vec4i> hierarchy;
  vector<future<Point2f>> robotPositionFutures;
  robotPositionFutures.resize(4);


  while(true)
  {

    auto start = high_resolution_clock::now();
    vector<Mat> foregroundImages;

    vector<Point> selectedContour;
    vector<vector<vector<Point>>> allContours;
    // temp logic for disambiguation. Each successive mice position can differ by 
    // maximum of 2. If a mice is not detected in a frame, the maximum allowed difference
    // between previous detected mice position and present mice position is increased and 
    // the maximum allowed difference comes back to 2, when mice is detected in a frame
    // again. This is very hueritic to disambiguate mice position from robot position and 
    // should be scrutanised further
    int diff = 2;

    // start image capture
    auto start1 = high_resolution_clock::now();
    imageTransferObj.transferAllImagestoPC();

    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start);
    cout << "\nAcquisition time: "<< duration1.count() << endl;

    frames[0] = imageTransferObj.image0.clone();
    frames[1] = imageTransferObj.image1.clone();
    frames[2] = imageTransferObj.image2.clone();
    frames[3] = imageTransferObj.image3.clone();

    Mat threshold;

    vector<int> previousClosestCell;

    vector<vector<Point> > contours;

    // launch parallel threads for detecting the robot position by detecting a black
    // square
    for(int i=0; i<4; ++i)
    {
       robotPositionFutures[i] = std::async(std::launch::async, detectRobotPosition,
                                   frames[i]); 
    }


    // do background subtraction and detect all contours on the background subtracted 
    // image
    for(int i=0; i < cameraCount; ++i)
    {

      vector<vector<Point> > contours;
      auto foregroundImage = bgsubs[i].processImage(frames[i]);
      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
      allContours.push_back(std::move(contours));
    }
    auto start2 = high_resolution_clock::now();

    //  Check if robot localisation routnine has ended
    for(int i=0; i < cameraCount; ++i)
    {
        while(robotPositionFutures[i].wait_for(std::chrono::seconds(0)) != std::future_status::ready);
    }

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    cout << "\nwait time for marker detection: "<< duration2.count() << endl;

    // Process contours detected in each image to detect the mice, do cell association 
    // for the mice and draw the mice position on the image

    for(int i=0; i < cameraCount; ++i)
     {
      auto contours = allContours[i];


      auto robotPosition = robotPositionFutures[i].get(); 

      int maxContourId = getMaxAreaContourId(contours, robotPosition), cx, cy;

      circle(frames[i], robotPosition, 30, cv::Scalar(255), -1);

      if(maxContourId >= 0)
      {
        auto M = moments(contours[maxContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(frames[i], cv::Point(cx , cy), 30, cv::Scalar(255), -1);
	auto associatedCell  = cellAssociation[i].return_closest_cell(cx, cy);

  // huerisitic logic for robot mice dismbiguation.
  // temp logic for disambiguation. Each successive mice position can differ by 
  // maximum of 2. If a mice is not detected in a frame, the maximum allowed difference
  // between previous detected mice position and present mice position is increased and 
  // the maximum allowed difference comes back to 2, when mice is detected in a frame
  // again. This is very hueritic to disambiguate mice position from robot position and 
  // should be scrutanised further

	if(previousClosestCell.size() == 0)
	{
	 previousClosestCell = associatedCell;
        }

	if((abs(previousClosestCell[0] - associatedCell[0]) + abs(previousClosestCell[1] - associatedCell[1])) <= diff)
	{
	  putText(frames[i], to_string(associatedCell[0]) + ","+ to_string(associatedCell[1]),
		Point(cx, cy-10), FONT_HERSHEY_COMPLEX_SMALL, 3, Scalar(0, 255, 0), 2);
	  diff = 2;
	  previousClosestCell = associatedCell;
	 }
	else
	{
	  diff += 1;
	}
      }

    }

    // Write videos (Both raw image and stitched image)
    rawVideoRecorder.writeFrames(frames);
    stitchedVideorecorder.writeFrames(frames);
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
