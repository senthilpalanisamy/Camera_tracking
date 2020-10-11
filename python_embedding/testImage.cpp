#include <iostream>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main (int argc, char *argv[])
{
    // float data[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    //Mat mat1 (cv::Size (5, 2), CV_32F, data, Mat::AUTO_STEP);

    auto image = imread("test.png");
    cout<<image.type()<<"\n";
    int row = 0;
    //float *p = mat1.ptr<float>(row);
    Vec3b *p = image.ptr<Vec3b>(row);

    //cout << "Mat" << mat1 <<endl;

    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;

    npy_intp dims[3] = { image.rows, image.cols, 3 };
    PyObject *py_array;

    setenv("PYTHONPATH","../",1);
    Py_Initialize ();
    pName = PyUnicode_FromString ("inference");

    pModule = PyImport_Import(pName);

    pDict = PyModule_GetDict(pModule);

    // Required for the C-API : http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
    import_array();

    py_array = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, p);

    pArgs = PyTuple_New (1);
    PyTuple_SetItem (pArgs, 0, py_array);

    pFunc = PyDict_GetItemString (pDict, (char*)"detect_mice_pose"); 

    if (PyCallable_Check (pFunc))
    {
        PyObject_CallObject(pFunc, pArgs);
    } else
    {
        cout << "Function is not callable !" << endl;
    }
    cout<<"out of call";

    Py_DECREF(pName);
    Py_DECREF (py_array);
    Py_DECREF (pModule);
    Py_DECREF (pDict);
    Py_DECREF (pFunc);

    Py_Finalize ();
    cout<<"out";

    return 0;
}
