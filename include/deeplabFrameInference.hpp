#ifndef DEEP_LAB_FRAME_INFERENCE
#define DEEP_LAB_FRAME_INFERENCE
#include <pybind11/embed.h>
#include <iostream>
#include <vector>
#include <pybind11/stl.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <Python.h>
#include <opencv2/opencv.hpp>


using std::cout;
using std::vector;
using std::list;
using std::endl;
using cv::Vec3b;
using cv::Mat;

class DeepLabFrameInference
{

  PyObject *pModule, *pFunc, *py_array, *pName, *pDict, *pArgs;
  npy_intp dims[3] = {0, 0, 0};
  public:
  DeepLabFrameInference();
  void predictPoseForImage(Mat& image);
  ~DeepLabFrameInference();
};

#endif

