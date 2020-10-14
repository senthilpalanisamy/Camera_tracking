#include<deeplabFrameInference.hpp>

using cv::imread;

DeepLabFrameInference::DeepLabFrameInference(int rows, int cols)
  {


    dims[0] = rows;
    dims[1] = cols;
    dims[2] = 3;

    setenv("PYTHONPATH","../src",1);
    Py_Initialize ();
    import_array1();
    // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    pName = PyUnicode_FromString ("deeplabFrameInference");

    pModule = PyImport_Import(pName);

    pDict = PyModule_GetDict(pModule);

    pFunc = PyDict_GetItemString (pDict, (char*)"infer");
  }

  void DeepLabFrameInference::predictPoseForImage(Mat& image)
   {

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
   {

    Py_DECREF(pName);
    Py_DECREF (py_array);
    Py_DECREF (pModule);
    Py_DECREF (pDict);
    Py_DECREF (pFunc);
    Py_Finalize ();

   }



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
