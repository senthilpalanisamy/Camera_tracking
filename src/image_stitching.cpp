#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/calib3d/calib3d.hpp"


using namespace cv;
using std::vector;
using namespace cv::xfeatures2d;
using cv::detail::MatchesInfo;


int main(void)
{
  Mat img1 = cv::imread("../data/b1.png", 0);
  Mat img2 = cv::imread("../data/b2.png", 0);
  vector<Mat> images;
  images.push_back(img1);
  images.push_back(img2);
  size_t noOfImages=2;
  //SurfFeatureDetector surfDetector;


  vector<Mat> descriptors;
  size_t i=0;
  for(;i<noOfImages; i++)
  {
    Mat img_descriptor;
    descriptors.push_back(img_descriptor);
  }
  vector<vector<KeyPoint>> keypoints;

  for(i=0;i<noOfImages; i++)
  {
    vector<KeyPoint> imgKeypoint;
    keypoints.push_back(imgKeypoint);
  }
  vector<MatchesInfo> pairwise_matches;

  //vector<ImageFeatures> features(num_images);

  //auto siftDetector = SIFT_create();
   auto detector = ORB::create();
  size_t num_images = 2;
   
  for(i=0; i < 2; i++)
   {
    detector->detectAndCompute( images[i], cv::noArray(), keypoints[i], descriptors[i] );

    //surfDetector.detectAndCompute(images[i], noArray(), keypoints[i], descriptors[i]);
   } 
   BFMatcher brute_force_matcher = cv::BFMatcher(NORM_L2, true);
   vector< cv::DMatch > matches;
   brute_force_matcher.match(descriptors[0], descriptors[1], matches);
   //Ptr<SURF> detector = SURF::create(0.4);
    const float ratio_thresh = 0.7f;

    // debugging
    Mat img_matches;
    drawMatches( images[0], keypoints[0], images[1], keypoints[1], matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints[0][matches[i].queryIdx].pt);
        scene.push_back(keypoints[1][matches[i].trainIdx].pt);
    }
    Mat H = findHomography( obj, scene, RANSAC );
    Mat image2_aligned;
    Size imgSize = images[0].size();
    imgSize.width = imgSize.width + images[1].size().width; 
    warpPerspective (images[1], image2_aligned, H, imgSize,INTER_LINEAR + WARP_INVERSE_MAP);
    size_t j;
    for(i=0; i < images[0].size().height; i++)
     {
        for(j=0; j < images[0].size().width;j++)
        {
        //image2_aligned(i, j) = images[0](i, j);
        image2_aligned.at<char>(i,j) = images[0].at<char>(i,j);
        }


    }

    //-- Show detected matches
    imshow("Good Matches", image2_aligned);
    waitKey();
}
