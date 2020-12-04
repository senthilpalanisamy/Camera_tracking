#include <iostream>

#include "image_stitching.hpp"
#include "simple_capture.hpp"

using namespace std::chrono;
using std::cout;
using std::endl;

int main(void)
{
  cout<<"started capture";
  // frameGrabber imageTransferObj("./config/red_light_with_binning.fmt");

  //   frameGrabber imageTransferObj("./config/video_config/red_light_with_binning.fmt", true,
  //   	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");
  // 
  //  frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", true,
  //  	                 "/home/senthil/work/Camera_tracking/config/camera_intrinsics_1024x1024");

  frameGrabber imageTransferObj("./config/video_config/room_light_binning.fmt", false);
  imageTransferObj.transferAllImagestoPC();
  vector<Mat> images;
  images.push_back(imageTransferObj.image0);
  images.push_back(imageTransferObj.image1);
  images.push_back(imageTransferObj.image2);
  images.push_back(imageTransferObj.image3);



  // images.push_back(dst1);
  cout<<"ended capture";
  imageStitcher imgStitcher;
  Mat stitchedImage;


   while(true)
   {
     auto start = high_resolution_clock::now();
     imageTransferObj.transferAllImagestoPC();
     auto stop = high_resolution_clock::now();
     auto duration = duration_cast<milliseconds>(stop - start);
     cout<<"image acquisition time:"<<duration.count()<<endl;
     vector<Mat> images;
     // images.push_back(imageTransferObj.image0);


     images.push_back(imageTransferObj.image0);
     images.push_back(imageTransferObj.image1);
     images.push_back(imageTransferObj.image2);
     images.push_back(imageTransferObj.image3);

     

     cout<<"\nstitching image\n";
     start = high_resolution_clock::now();
     stitchedImage = imgStitcher.stitchImagesOnline(images);
     stop = high_resolution_clock::now();
     duration = duration_cast<milliseconds>(stop - start);
     cout<<"Image stitching time:"<<duration.count()<<endl;
     imshow("stitchedImageop", stitchedImage);
     if(waitKey(1) >= 0)
       break;
     //imageTransferObj.displayAllImages();

     imwrite("outputimage.png", stitchedImage);

 } 

}
