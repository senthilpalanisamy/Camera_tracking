#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace std::chrono;
using std::cout;
using std::endl;

//using namespace cv::aruco;

int main()
{

  cv::Mat inputImage = cv::imread("marker.jpg", 0);
  std::vector<int> markerIds;
  std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
  cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
  cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
  //cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_7X7_1000);

  auto start = high_resolution_clock::now();
  cv::resize(inputImage, inputImage, cv::Size(inputImage.cols, inputImage.rows), 0, 0, CV_INTER_LINEAR);
  cv::aruco::detectMarkers(inputImage, dictionary, markerCorners, markerIds, parameters, rejectedCandidates);
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  cout << "\ntime: "<<duration.count() << endl;
  cout<<"maker size: "<<markerIds.size()<<endl;

  cv::Mat outputImage = inputImage.clone();
  cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);
  cv::imshow("image", outputImage);
  cv::waitKey(0);
  return 0;
}
