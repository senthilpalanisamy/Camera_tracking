// Author: Senthil Palanisamy
// A convenience class defining object tracking for images
// TODO: This is an incomplete file and may not even compile

#ifndef OBJECT_TRACKING_INCLUDE_GAURD
#define OBJECT_TRACKING_INCLUDE_GAURD

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/core/ocl.hpp>

#include <opencv2/tracking/tracker.hpp>
#include <opencv2/tracking/tldDataset.hpp>

using cv::Mat;

class objectTracking
{

  Mat frame;
  Ptr<Tracker> tracker;
  Rect2d bbox;
  bool isVisualise;

  objectTracking(const string& trackerType, const Rect2d& bbox_,
                 const Mat& firstFrame, bool isVisualise_);

  void trackObject(Mat& nextFrame);
  void visualiseTracker(Mat& nextFrame, bool isTrackingSuccess);
}


#endif
