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

class imageStitcher
{
  public:
  imageStitcher(vector<Mat> images)
  {
   size_t i;
   Mat stitchedImage = images[0].clone();
   for(i=0; i < images.size()-1; i++)
   {
   Mat homography = computeHomography(stitchedImage, images[i+1]);
   stitchedImage = stitchImages(stitchedImage, images[i+1], homography);

   //imwrite("result.jpg", stitchedImage);
   //imshow("stitchedImage", stitchedImage);
   //imshow("newImage", images[i+1]);
   //waitKey(0);
   }
   //imshow("stitchedImage", stitchedImage);
   //waitKey(0);
   imwrite("result.jpg", stitchedImage);
  }


  private:
  Mat stitchImages(Mat image1, Mat image2, Mat Homography)
   {

    Mat image2_aligned;
    Size imgSize = image1.size();
    imgSize.width = imgSize.width + image2.size().width; 
    warpPerspective (image2, image2_aligned, Homography, imgSize,INTER_LINEAR + WARP_INVERSE_MAP);
    imshow("Warped Image", image2_aligned);
    waitKey(0);
    size_t j,i;
    for(i=0; i < image1.size().height; i++)
     {
        for(j=0; j < image1.size().width;j++)
        {
          if(image1.at<char>(i,j)!=0)
          {
           //image2_aligned(i, j) = images[0](i, j);
           image2_aligned.at<char>(i,j) = image1.at<char>(i,j);
          }
          }
     }

    imshow("stitched Image", image2_aligned);
    waitKey(0);
    return image2_aligned;
   }

  Mat computeHomography(Mat image1, Mat image2)
  {
   vector<Mat> images;
   images.push_back(image1);
   images.push_back(image2);


  vector<Mat> descriptors;
  size_t i=0;
  for(;i<2; i++)
  {
    Mat img_descriptor;
    descriptors.push_back(img_descriptor);
  }
  vector<vector<KeyPoint>> keypoints;

  for(i=0;i<2; i++)
  {
    vector<KeyPoint> imgKeypoint;
    keypoints.push_back(imgKeypoint);
  }
  vector<MatchesInfo> pairwise_matches;

  auto detector = ORB::create();
  size_t num_images = 2;

  for(i=0; i < 2; i++)
   {
    detector->detectAndCompute( images[i], cv::noArray(), keypoints[i], descriptors[i] );
   } 
   BFMatcher brute_force_matcher = cv::BFMatcher(NORM_L2, true);
   vector< cv::DMatch > matches;
   brute_force_matcher.match(descriptors[0], descriptors[1], matches);
   const float ratio_thresh = 0.7f;

    // debugging
    Mat img_matches;
    drawMatches( images[0], keypoints[0], images[1], keypoints[1], matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("matched_image", img_matches);
    waitKey(0);

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back(keypoints[0][matches[i].queryIdx].pt);
        scene.push_back(keypoints[1][matches[i].trainIdx].pt);
    }
    Mat H = findHomography( obj, scene, RANSAC );
    return H;

  }




};

int main(void)
{
  //Mat img1 = cv::imread("../data/boat1.jpg", 0);
  //Mat img2 = cv::imread("../data/boat2.jpg", 0);
  //Mat img3 = cv::imread("../data/boat3.jpg", 0);
  //Mat img4 = cv::imread("../data/boat4.jpg", 0);
  //Mat img5 = cv::imread("../data/boat5.jpg", 0);
  //Mat img6 = cv::imread("../data/boat6.jpg", 0);
  Mat img1 = cv::imread("../data/newspaper1.jpg", 0);
  Mat img2 = cv::imread("../data/newspaper2.jpg", 0);
  Mat img3 = cv::imread("../data/newspaper3.jpg", 0);
  Mat img4 = cv::imread("../data/newspaper4.jpg", 0);
  vector<Mat> images;
  images.push_back(img1);
  images.push_back(img2);
  images.push_back(img3);
  images.push_back(img4);
  //images.push_back(img5);
  //images.push_back(img6);
  imageStitcher imgStitcher(images);
  //SurfFeatureDetector surfDetector;



}
