// Author: Senthil Palanisamy
// An example file showing how a python interpreter can be invoked from C++
// using pybind11 and to load deeplabcut model and to do frame inference for predicting
// mice pose through C++
// Be sure to have source the python environment before running these scripts. I have
// maintained python environments in virtual environment. Typing "workon deeplabcut" in 
// the terminal should source the environment. But actual deeplabcut authors have a docker
// container as well. Either structure works but be sure that necessary python environment
// is sourced before the script execution

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

