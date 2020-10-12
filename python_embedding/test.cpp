#include <pybind11/embed.h>
#include <iostream>
#include <vector>
#include <pybind11/stl.h>

using std::cout;
using std::vector;
using std::list;

namespace py = pybind11;


int main()
{
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive
  //py::module sys = py::module::import("sys");
  //py::print(sys.attr("path"));

  py::module test = py::module::import("deeplabFrameInference");
  auto result = test.attr("generate_list")(5);
  auto resultNew = py::cast<py::list>(result);
  // auto resultNew = static_cast<<int> > result;

  // auto resultNew = test.cast<list>();
  cout<<"Printing numbers\n";
  for(auto a: resultNew)
  {
    cout<<a<<"\t";
  }
  cout<<"\n";
  cout<<"finish";




}

