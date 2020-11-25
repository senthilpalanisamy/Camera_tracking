#include <backGroundSubtraction.hpp>
#include "simple_capture.hpp"
#include <video_recorder.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>

#include <ctime>
#include <sys/stat.h>




using namespace std::chrono;
using std::move;
using std::vector;
using std::ifstream;
using std::transform;
using std::back_inserter;
using std::for_each;
using std::pow;
using std::cout;
using std::string;
using cv::Point;
using cv::Scalar;
using std::to_string;

class cameraCellAssociator
{

  public:

    vector<vector<int>> cell_centers;
    vector<vector<int>> cell_index;

    cameraCellAssociator(string fileName)
    {

       //ifstream myfile("./config/camera0.txt");
       ifstream myfile(fileName);
       int pixelx, pixely, cellx, celly;

        while(myfile >> pixelx >> pixely >> cellx >> celly)
       {
         cell_centers.push_back({pixelx, pixely});
         cell_index.push_back({cellx, celly});
       }

    }


vector<int> return_closest_cell(int mice_x, int mice_y)
  {
    vector<double> distances(cell_centers.size());

    for(int i=0; i < cell_centers.size(); ++i)
    {
      auto point = cell_centers[i];
      distances[i] = pow(pow(point[0] - mice_x, 2)+
                     pow(point[1] - mice_y, 2), 0.5);
    }
    int index = min_element(distances.begin(), distances.end()) - distances.begin();
    auto cell = cell_index[index];
    return cell;
  }

};


BackGroundSubtractor::BackGroundSubtractor(Method m_, const Mat& firstImage, 
                                           bool isVisualise_)
  {

    GpuMat d_frame(firstImage);
    //imshow("image", firstImage);
    //waitKey(0);
    if(m == MOG)
    {
      mog = cuda::createBackgroundSubtractorMOG();
      mog->apply(d_frame, d_fgmask, 0.01);
    }
    else
    {
      mog = cuda::createBackgroundSubtractorMOG2();
      mog->apply(d_frame, d_fgmask);
    }

    isVisualise = isVisualise_;

    if(isVisualise)
    {
      namedWindow("image", WINDOW_NORMAL);
      namedWindow("foreground mask", WINDOW_NORMAL);
      namedWindow("foreground image", WINDOW_NORMAL);
      namedWindow("mean background image", WINDOW_NORMAL);
    }
    m = m_;
  }

  Mat BackGroundSubtractor::processImage(const Mat& nextFrame)
  {

    d_frame.upload(nextFrame);
    if(m == MOG)
    {
        mog->apply(d_frame, d_fgmask, 0.01);
    }
    else
    {
       mog->apply(d_frame, d_fgmask);
    }

    mog->getBackgroundImage(d_bgimg);



    d_fgmask.download(fgmask);

    if(isVisualise)
    {
      visualiseImage(nextFrame);
    }
    return fgmask;
  }

  void BackGroundSubtractor::visualiseImage(const Mat& nextFrame)
  {

  d_fgimg.create(d_frame.size(), d_frame.type());
  d_fgimg.setTo(Scalar::all(0));
  d_frame.copyTo(d_fgimg, d_fgmask);

   d_fgimg.download(fgimg);
   if (!d_bgimg.empty())
     d_bgimg.download(bgimg);


  imshow("image", nextFrame);
  imshow("foreground mask", fgmask);
  imshow("foreground image", fgimg);
  if (!bgimg.empty())
      imshow("mean background image", bgimg);
  waitKey(30);
  }


int getMaxAreaContourId(vector <vector<cv::Point>> contours) 
{
    double maxArea = 0;
    int maxAreaContourId = -1;
    for (int j = 0; j < contours.size(); j++) {
        double newArea = cv::contourArea(contours.at(j));
        if (newArea > maxArea) {
            maxArea = newArea;
            maxAreaContourId = j;
        }
    }

    if(maxArea > 600)
    {
      return maxAreaContourId;
    }
    else
    {
      return -1;
    }
}

string return_date_header()
{

   time_t now = time(0);
   tm *ltm = localtime(&now);
   string date_header = to_string(1 + ltm->tm_mon) + "_" + to_string(ltm->tm_mday); 
   return date_header;
		             
}

string getFolderPath()
{

   string folder_name, outputPath;
   bool isFolderPathUnique = true;

   while(isFolderPathUnique)
   {

   cout<<"Please enter a unique folder name for saving results\n";
   cin>>folder_name;
   outputPath = "./samples/" + return_date_header() + "/" + folder_name;
   struct stat buffer;
   isFolderPathUnique = !stat (outputPath.c_str(), &buffer); 
   }
   return outputPath;
}



int main()
{
  int cameraCount = 4;
  string experimentPath = getFolderPath();

  vector<cameraCellAssociator> cellAssociation;
  cellAssociation.emplace_back("./config/camera0.txt");
  cellAssociation.emplace_back("./config/camera1.txt");
  cellAssociation.emplace_back("./config/camera2.txt");
  cellAssociation.emplace_back("./config/camera3.txt");

  frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");


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

  for(int i=0; i<cameraCount; ++i)
  {


    //Mat frame;
    //caps[i].read(frame);
    bgsubs.emplace_back(method, frames[i], false);
    //frames.push_back(frame);
  }

  //BackGroundSubtractor backgroundSubtractor(method, frame, false);
  string rawVideoPath = experimentPath + "/" + "unprocessed"; 


  auto recorder = videoRecorder(4, "bg_output", frames[0].size(), 10, false,
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

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    recorder.writeFrames(frames);
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
