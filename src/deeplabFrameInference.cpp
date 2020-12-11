// Author: Senthil Palanisamy
// An example file showing how a python interpreter can be invoked from C++
// using pybind11 and to load deeplabcut model and to do frame inference for predicting
// mice pose through C++
// Be sure to have source the python environment before running these scripts. I have
// maintained python environments in virtual environment. Typing "workon deeplabcut" in 
// the terminal should source the environment. But actual deeplabcut authors have a docker
// container as well. Either structure works but be sure that necessary python environment
// is sourced before the script execution

#include<deeplabFrameInference.hpp>

using cv::imread;

DeepLabFrameInference::DeepLabFrameInference()
  // Sets the python enviroment and invokes  the python interpretor
  {
    setenv("PYTHONPATH","./src",1);
    Py_Initialize ();
    import_array1();
    // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    pName = PyUnicode_FromString ("deeplabFrameInference");

    pModule = PyImport_Import(pName);

    pDict = PyModule_GetDict(pModule);

    pFunc = PyDict_GetItemString (pDict, (char*)"infer");
  }

  void DeepLabFrameInference::predictPoseForImage(Mat& image)
  // call the deeplabcut module by sending the image and get back a numpy array 
  // for predicting mice pose
   {

    dims[0] = image.rows;
    dims[1] = image.cols;
    dims[2] = 3;

    Vec3b *p = image.ptr<Vec3b>(0);
    py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, p);
    pArgs = PyTuple_New (1);
    PyTuple_SetItem (pArgs, 0, py_array);


    if (PyCallable_Check (pFunc))
    {
        PyObject_CallObject(pFunc, pArgs);
    } else
    {
        cout << "Function is not callable !" << endl;
    }

   }

  DeepLabFrameInference::~DeepLabFrameInference()
  // Destroy the python interpretor and the modules imported
   {

    Py_DECREF(pName);
    Py_DECREF (py_array);
    Py_DECREF (pModule);
    Py_DECREF (pDict);
    Py_DECREF (pFunc);
    Py_Finalize ();

   }


// An example showing how deeplab cut could be utilised
//int main (int argc, char *argv[])
//{
//    // float data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
//
//    //Mat mat1 (cv::Size (5, 2), CV_32F, data, Mat::AUTO_STEP);
//
//    auto image = imread("../test.jpeg");
//    cout<<image.type()<<"\n";
//    auto dlc = DeepLabFrameInference(360, 640);
//    dlc.predictPoseForImage(image);
//    dlc.predictPoseForImage(image);
//
//    return 0;
//}
