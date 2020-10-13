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
using cv::imread;
using cv::Vec3b;

namespace py = pybind11;


int main()
{
  //py::module sys = py::module::import("sys");
  //py::print(sys.attr("path"));
  //
  //
  // setenv("PYTHONPATH","../",1);
  // auto image = imread("../test.jpeg");
  // cout<<image.type()<<"\n";
   // int row = 0;
   // float *p = mat1.ptr<float>(row);
   // Vec3b *p = image.ptr<Vec3b>(row);

   // npy_intp dims[3] = { image.rows, image.cols, 3 };
  // PyObject *py_array;

  // import_array();

  // py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, p);
  //
  //
  //

   // py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, p);
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  //py::exec("import deeplabFrameInference");
  auto sysModule = py::module::import("sys");
  py::print(sysModule.attr("path"));
  py::module test = py::module::import("deeplabcut");
  py::module module2 = py::module::import("deeplabFrameInference");
  module2.attr("infer")(0);
  module2.attr("infer")(0);
     //auto image = imread("./test.jpeg");
     //cout<<image.type()<<"\n";
    // int row = 0;
    // //float *p = mat1.ptr<float>(row);
    // Vec3b *p = image.ptr<Vec3b>(row);


   // auto image = imread("../test.jpeg");
   // cout<<image.type()<<"\n";
   // int row = 0;
   // // float *p = mat1.ptr<float>(row);
   // Vec3b *p = image.ptr<Vec3b>(row);

   // npy_intp dims[3] = { image.rows, image.cols, 3 };
   // PyObject *py_array;

   // import_array();
  //
  //py::module test = py::module::import("deeplabFrameInference");



  // auto result = test.attr("infer")(py_array);
  // auto resultNew = py::cast<py::list>(result);
  // // auto resultNew = static_cast<<int> > result;

  // // auto resultNew = test.cast<list>();
  // cout<<"Printing numbers\n";
  // for(auto a: resultNew)
  // {
  //   cout<<a<<"\t";
  // }
  // cout<<"\n";
  cout<<"finish";




}

