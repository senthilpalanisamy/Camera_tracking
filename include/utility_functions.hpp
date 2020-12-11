// Author: Senthil Palanisamy
// A file describing some utility functions used across all the other scripts

#ifndef UTILITY_FUNCTIONS_INCLUDE_GAURD
#define UTILITY_FUNCTIONS_INCLUDE_GAURD

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using std::vector;
//using cv::detail::MatchesInfo;
using std::cout;
using std::string;
using std::tuple;

extern Mat outputImage;



struct ImageStitchData
{
  Mat dstImage;
  Mat inputImage;
  Mat homography;
};


class cameraCellAssociator
{

  public:

    vector<vector<int>> cell_centers;
    vector<vector<int>> cell_index;

    cameraCellAssociator(string fileName);
    vector<int> return_closest_cell(int mice_x, int mice_y);
};

Mat stitchImageschessBoard(Mat stitchedImage, Mat ipImage, Mat Homography);
void* WarpandStitchImages(void *arguments);

void performLensCorrection(Mat& image, int imageNo, string lensCorrectionFolderPath);
int getMaxAreaContourId(vector <vector<cv::Point>> contours, Point2f);
string return_date_header();
string getFolderPath();
string return_date_time_header();
tuple<vector<double>, Mat> readCameraParameters(string jsonFilePath);


#endif

