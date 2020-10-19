#include<objectTracking.hpp>
using cv::imshow;
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

  objectTracking::objectTracking(const string& trackerType, const Rect2d& bbox_, 
                 const Mat& firstFrame, bool isVisualise_)
  {


    string trackerTypes[8] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT"};

   if (trackerType == "BOOSTING")
       tracker = TrackerBoosting::create();
   if (trackerType == "MIL")
       tracker = TrackerMIL::create();
   if (trackerType == "KCF")
       tracker = TrackerKCF::create();
   if (trackerType == "TLD")
       tracker = TrackerTLD::create();
   if (trackerType == "MEDIANFLOW")
       tracker = TrackerMedianFlow::create();
   if (trackerType == "GOTURN")
       tracker = TrackerGOTURN::create();
   if (trackerType == "MOSSE")
       tracker = TrackerMOSSE::create();
   if (trackerType == "CSRT")
            tracker = TrackerCSRT::create();

   bbox = bbox_;
   tracker->init(frame, bbox);
   isVisualise = isVisualise_;

   if(isVisualise)
   {
      rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 ); 
      imshow("Tracking", firstFrame); 
   }

  }

  void objectTracking::trackObject(Mat& nextFrame)
  {
      bool ok = tracker->update(frame, bbox);
      if(isVisualise)
      {
        visualiseTracker(nextFrame, ok)
      }
  }

  void objectTracking::visualiseTracker(Mat& nextFrame, bool isTrackingSuccess)
  {
    if(isTrackingSuccess)
    {
      rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
    }
    else
    {
       putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
    }

    //putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);

    //putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

     // Display frame.
     imshow("Tracking", frame);
  }

};
