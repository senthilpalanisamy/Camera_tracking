#include <backGroundSubtraction.hpp>
#include <video_recorder.hpp>
#include <iostream>
#include <string>

#include<future>

using namespace std::chrono;
using std::move;
using std::mutex;
using std::async;
using std::future;
using std::ref;


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



void parallelProcessImage1(BackGroundSubtractor& bg, Mat& image,
                          vector<vector<int>>& coOridnates,
                          vector<Mat>& outputImages, int i)
{
  //cout<<"begin";
   //std::this_thread::sleep_for(std::chrono::seconds(3));

     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     auto foregroundImage = bg.processImage(image);

      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

      int  cx=-1, cy=-1;

      double maxArea = 0;
      int maxAreaContourId = -1;

      for (int j = 0; j < contours.size(); j++) {
          double newArea = cv::contourArea(contours.at(j));
          if (newArea > maxArea) {
              maxArea = newArea;
              maxAreaContourId = j;
          }
      }

      if(maxArea < 3200)
      {
        maxAreaContourId = -1;
      }

      if(maxAreaContourId >= 0)
      {
        auto M = moments(contours[maxAreaContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(image, cv::Point(cx , cy), 30, cv::Scalar(255), -1);
      }
      //cout<<"maxArea"<<maxArea<<endl;
      //cout<<"contourId"<<maxAreaContourId<<endl;

      coOridnates[i] = {cx, cy};
      outputImages[i] = image;
}

void parallelProcessImage2(BackGroundSubtractor& bg, Mat& image,
                          vector<vector<int>>& coOridnates,
                          vector<Mat>& outputImages, int i)
{
  //cout<<"begin";
   //std::this_thread::sleep_for(std::chrono::seconds(3));

     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     auto foregroundImage = bg.processImage(image);

      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

      int  cx=-1, cy=-1;

      double maxArea = 0;
      int maxAreaContourId = -1;

      for (int j = 0; j < contours.size(); j++) {
          double newArea = cv::contourArea(contours.at(j));
          if (newArea > maxArea) {
              maxArea = newArea;
              maxAreaContourId = j;
          }
      }

      if(maxArea < 3200)
      {
        maxAreaContourId = -1;
      }

      if(maxAreaContourId >= 0)
      {
        auto M = moments(contours[maxAreaContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(image, cv::Point(cx , cy), 30, cv::Scalar(255), -1);
      }
      //cout<<"maxArea"<<maxArea<<endl;
      //cout<<"contourId"<<maxAreaContourId<<endl;

      coOridnates[i] = {cx, cy};
      outputImages[i] = image;
}

void parallelProcessImage3(BackGroundSubtractor& bg, Mat& image,
                          vector<vector<int>>& coOridnates,
                          vector<Mat>& outputImages, int i)
{
  //cout<<"begin";
   //std::this_thread::sleep_for(std::chrono::seconds(3));

     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     auto foregroundImage = bg.processImage(image);

      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

      int  cx=-1, cy=-1;

      double maxArea = 0;
      int maxAreaContourId = -1;

      for (int j = 0; j < contours.size(); j++) {
          double newArea = cv::contourArea(contours.at(j));
          if (newArea > maxArea) {
              maxArea = newArea;
              maxAreaContourId = j;
          }
      }

      if(maxArea < 3200)
      {
        maxAreaContourId = -1;
      }

      if(maxAreaContourId >= 0)
      {
        auto M = moments(contours[maxAreaContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(image, cv::Point(cx , cy), 30, cv::Scalar(255), -1);
      }
      //cout<<"maxArea"<<maxArea<<endl;
      //cout<<"contourId"<<maxAreaContourId<<endl;

      coOridnates[i] = {cx, cy};
      outputImages[i] = image;
}

void parallelProcessImage4(BackGroundSubtractor& bg, Mat& image,
                          vector<vector<int>>& coOridnates,
                          vector<Mat>& outputImages, int i)
{
  //cout<<"begin";
   //std::this_thread::sleep_for(std::chrono::seconds(3));

     vector<vector<Point> > contours;
     vector<Vec4i> hierarchy;

     auto foregroundImage = bg.processImage(image);

      findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

      int  cx=-1, cy=-1;

      double maxArea = 0;
      int maxAreaContourId = -1;

      for (int j = 0; j < contours.size(); j++) {
          double newArea = cv::contourArea(contours.at(j));
          if (newArea > maxArea) {
              maxArea = newArea;
              maxAreaContourId = j;
          }
      }

      if(maxArea < 3200)
      {
        maxAreaContourId = -1;
      }

      if(maxAreaContourId >= 0)
      {
        auto M = moments(contours[maxAreaContourId]);
        cx = int(M.m10 / M.m00);
        cy = int(M.m01 / M.m00);
        circle(image, cv::Point(cx , cy), 30, cv::Scalar(255), -1);
      }
      //cout<<"maxArea"<<maxArea<<endl;
      //cout<<"contourId"<<maxAreaContourId<<endl;

      coOridnates[i] = {cx, cy};
      outputImages[i] = image;
}






int main()
{
  int cameraCount = 4;
  vector<string> input_video_paths;
  input_video_paths.push_back("/home/senthil/work/Camera_tracking/all_results/results_10/resultssample_trial0.mp4");
  input_video_paths.push_back("/home/senthil/work/Camera_tracking/all_results/results_10/resultssample_trial1.mp4");
  input_video_paths.push_back("/home/senthil/work/Camera_tracking/all_results/results_10/resultssample_trial2.mp4");
  input_video_paths.push_back("/home/senthil/work/Camera_tracking/all_results/results_10/resultssample_trial3.mp4");

  vector<VideoCapture> caps;
  for(int i=0; i <  input_video_paths.size(); ++i)
  {
   caps.emplace_back(input_video_paths[i]);
  }

  //VideoCapture cap(input_video);

  vector<Mat> frame;
  Method method=MOG2;
  for(auto cap: caps)
  {
    if(!cap.isOpened())
    {
      cout<<"video cannot be opened";
      return 0;
    }
  }

  vector<BackGroundSubtractor> bgsubs;
  vector<Mat> frames;
  for(int i=0; i<cameraCount; ++i)
  {
    Mat frame;
    caps[i].read(frame);
    bgsubs.emplace_back(method, frame, false);
    frames.push_back(frame);
  }

  //BackGroundSubtractor backgroundSubtractor(method, frame, false);


  auto recorder = videoRecorder(4, "bg_output", frames[0].size(), 10, true);


  vector<Mat> outputImages(4);
  vector<vector<int>> coOridnates(4);
  //ParallelProcess p;

  while(true)
  {

    vector<future<void>> m_futures;
    vector<Mat> foregroundImages;
    for(int i=0; i< cameraCount; ++i)
    {
      if(!caps[i].read(frames[i]))
      {
        return 0;
      }
    }

    auto start = high_resolution_clock::now();

    //for(int i=0; i < cameraCount; ++i)
    //{

      m_futures.push_back(std::async(std::launch::async, parallelProcessImage1,
                                ref(bgsubs[0]), ref(frames[0]), ref(coOridnates), ref(outputImages), 
                                0));

      m_futures.push_back(std::async(std::launch::async, parallelProcessImage2,
                                ref(bgsubs[1]), ref(frames[1]), ref(coOridnates), ref(outputImages), 
                                1));

      m_futures.push_back(std::async(std::launch::async, parallelProcessImage3,
                                ref(bgsubs[2]), ref(frames[2]), ref(coOridnates), ref(outputImages), 
                                2));

      m_futures.push_back(std::async(std::launch::async, parallelProcessImage4,
                                ref(bgsubs[3]), ref(frames[3]), ref(coOridnates), ref(outputImages), 
                                3));
    //}

    for(int i=0; i < cameraCount; ++i)
    {
      // auto status = m_futures[i].wait_for(0);
      // cout<<"waiting";
      while( m_futures[i].wait_for(std::chrono::seconds(0)) != std::future_status::ready);
      //m_futures[i].get();
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\ntime: "<<duration.count() << endl;
    cout << "\nfps: "<<1.0 / (duration.count() /1000.0 / 1000.0) << endl;


    //imshow("image", frame);
    //waitKey(2);
    // Mat frame2 = frame.clone();
    recorder.writeFrames(frames);
    //cout<<"here";
  }

  return 0;



}
