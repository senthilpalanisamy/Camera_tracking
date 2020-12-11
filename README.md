This repository contains code for high-speed mice tracking and image stitching
High speed mice trackng is performed by using background subtraction algorithm.
Camera homographies are estimated by placing Charuco boards within the field
of view of the cameras. For a brief overview about the system, visit my portfolio
(https://senthilpalanisamy.github.io./) and visit the post under high speed 
image stitching and mice tracking (A backup for the same has been added under 
the folder web-backups). The post would alsp point you to a daily medium blog 
post on image stitching. The blog post has also been added as images 
within the web-back ups folder

Some quirks (dark secrets) about the system:-
1. Epix drivers for frame grabbers are supported in very specific linux kernels. 
   I have currently been running on 5.0.62 version. If kernel versions change, 
   drivers are not gauranteed to work. While the OS boots up in the system,
   go to advanced options for ubuntu and load 5.0.62 kernel everytime.
   If a different kernel is necessary, you may need to re-install these 
   drivers again.
2. All these drivers should be installed as a root user and hence, these drivers
   work only under the root user. Any program or executable that uses the 
   image capture from the frame grabbers will only compile / work if done
   as the root user.


How to run the system?
1. At the base directory, run cmake .., mkdir build, cd build, make and 
   then run any executable desired. 
2. It must be noted that all these executables must be run from the base
   directory of the repository. For example to run an executable called 
   back_ground_subt within build, you should be at the base directory and then
   do ./build/back_ground_subt. This is because most of the scripts assume a
   relative path for the config files. All config files reside in ./config which
   the scripts won't be able to find if they are run from the build directory

A description of all executables:-
1. image_stitching - Stitch images from all four cameras and show them. This
                     script relies on homography calibrations that were made
                     previously using calibrate_homographies. As long as the
                     cameras have not moved physically, the homographies don't
                     need to be recalibrated. If the cameras have moved,
                     consider running calibrate_homographies before running
                     image_stitching. This script also relies on lens distortion
                     parameters of the camera.
2. sample_capture -  An executable for storing raw and stitched image feed from
                     cameras. This is useful when you are running an experiment
                     but you don't have online tracking but just to record data
                     for future references
3. calibrate_homography - Estimate the homographies between the camera and the
                          stitched image view. The default location for storing
                          the estimated homographies is ./config/homographies
4. online_tracking -  Main execute for doing online tracking. This script does
                      all of the following
                      1. Tracks the position of the mice online
                      2. Tracks the position of the robot online
                      3. Writes the raw data from the camera views into the 
                      folder specified initially
                      4. Stitches images togther and writes the stitched image
                      into the specified folder

Dependencies:-
1. OpenCV - Should work with any version but I was using 4.0.0. More 
            importantly, opencv should be compiled with all optimisation
            options to do fast processing. In particular, these are some of 
            the options that I compiled opencv with 
            cmake -D CMAKE_BUILD_TYPE=RELEASE -D WITH_TBB=ON -D WITH_OPENMP=ON -D WITH_IPP=ON -
            D BUILD_EXAMPLES=OFF -D WITH_NVCUVID=ON D WITH_CUDA=ON D BUILD_DOCS=OFF 
            -D BUILD_PERF_TESTS=OFF -D BUILD_TESTS=OFF -D WITH_CSTRIPES=ON 
            -D WITH_OPENCL=ON      -D CMAKE_INSTALL_PREFIX=/usr/local  
            -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules     
            -D OPENCV_ENABLE_NONFREE=ON ..
            For any system with a GPU and intel processer this should stay the
            same. If the processor is different, check with openCV to figure
            out what options can be specified to utilise the low level
            instruction level parallelism, if any exists for the specified hardware
2. Pybind11 -  This is only needed if you are interested in using the C++
               warpper so that the deeplabcut inference code can be called
               from C++. The specific version that this code has been tested
               against is  v2.2.0
3. xcllib  - This is the epix C++ library for accessing frame grabber and 
             provides a means for getting images programmatically. More
             instructions on setting this up is provided in a seperate README.

