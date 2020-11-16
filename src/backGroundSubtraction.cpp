#include <backGroundSubtraction.hpp>
#include <video_recorder.hpp>
#include <iostream>
#include <string>

using namespace std::chrono;


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

    if(maxArea > 3200)
    {
      return maxAreaContourId;
    }
    else
    {
      return -1;
    }
}

int main()
{
  string input_video = "/home/senthil/work/Camera_tracking/all_results/results_10/resultssample_trial0.mp4";
  VideoCapture cap(input_video);

  Mat frame;
  Method method=MOG;
  if(!cap.isOpened())
  {
    cout<<"video cannot be opened";
  }

  cap.read(frame);

  BackGroundSubtractor backgroundSubtractor(method, frame, false);

  auto recorder = videoRecorder(1, "bg_output", frame.size(), 10, true);


  vector<vector<Point> > contours;
  vector<Vec4i> hierarchy;

  while(cap.read(frame))
  {
    auto start = high_resolution_clock::now();

    auto foregroundImage = backgroundSubtractor.processImage(frame);
    findContours( foregroundImage, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    int maxContourId = getMaxAreaContourId(contours), cx, cy;

    if(maxContourId >= 0)
    {
      auto M = moments(contours[maxContourId]);
      cx = int(M.m10 / M.m00);
      cy = int(M.m01 / M.m00);
      circle(frame, cv::Point(cx , cy), 30, cv::Scalar(255), -1);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "\ntime: "<<duration.count() << endl;
    cout << "\nfps: "<<1.0 / (duration.count() /1000.0 / 1000.0) << endl;


    //imshow("image", frame);
    //waitKey(2);
    // Mat frame2 = frame.clone();
    recorder.writeFrames({frame});
    cout<<"here";
  }

  return 0;



}
