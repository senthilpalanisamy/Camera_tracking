#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <math.h>


using std::vector;
using std::ifstream;
using std::transform;
using std::back_inserter;
using std::for_each;
using std::pow;
using std::cout;
using std::string;

class cameraCellAssociator
{

  public:

    vector<vector<int>> cell_centers;
    vector<vector<int>> cell_index;

    cameraCellAssociator(string fileName)
    {

       //ifstream myfile("./config/camera0.txt");
       ifstream myfile(fileName);
       int pixelx, pixely, cellx, celly;

        while(myfile >> pixelx >> pixely >> cellx >> celly)
       {
         cell_centers.push_back({pixelx, pixely});
         cell_index.push_back({cellx, celly});
       }

    }


vector<int> return_closest_cell(int mice_x, int mice_y)
  {
    vector<double> distances(cell_centers.size());

    for(int i=0; i < cell_centers.size(); ++i)
    {
      auto point = cell_centers[i];
      distances[i] = pow(pow(point[0] - mice_x, 2)+
                     pow(point[1] - mice_y, 2), 0.5);
    }
    int index = min_element(distances.begin(), distances.end()) - distances.begin();
    auto cell = cell_index[index];
    return cell;
  }

};



int main()
{


  int micex=40, micey=700;
  auto camera1 = cameraCellAssociator("./config/camera0.txt");
  auto cell = camera1.return_closest_cell(micex, micey);
  cout<<cell[0]<<"\t"<<cell[1];


}
