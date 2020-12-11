The whole system depends on a few config files. Use this document to learn
about all the config files that our system depends on
1. ./config/camera_homographies - Folder where homographies of the camera with 
                                  respect to the sitched image view is stored.
                                  The size of the stiched image is also stored 
                                  in the directory
2. ./config/camera_intrinsics   - There are two intrinsics folder - 1024 and 2048 
                                  the json files inside these folders contain
                                  lens distortion parameters and camera intrisic
                                  parameters. The lens distortion parameters stored
                                  in 1024 is very accurate but the one in 2048 is 
                                  not so accurate. If 2048 is needed, consider
                                  recalibrating the camera parameters.
3. ./config/video_config        - the video configuration fmt files describe the 
                                  camera capture and exposure parameters for 
                                  image capture. If the light changes, you will
                                  most likely need to export a new fmt file.
                                  In order to do so open xcap, adjust exposure 
                                  to a satisfactory level and then go to 
                                  vidoe --> video export --> unit 0 (or any 
                                  unit you feel is the perfect one) and then
                                  choose a folder for storing the configuration
                                  file (I have usually been storing it at
                                  /home/senthil/work/Camera_tracking/config/video_config)
                                  for consistency
4. ./config/cell_association    - Files which map each cell center to its 
                                  corresponding neighbour. If the camera positions
                                  change, you will need to open up image from 
                                  each camera in photoshop and manually enter
                                  this config file. This typically takes around
                                  4-5 hours
5. ./config/models              - The deeplabcut models for doing mice pose
                                  tracking. The trained models have not been
                                  commited to the repositories to keep the repo
                                  size reasonable but a model trained on around
                                  200 mice images captured from our lab data
                                  is available in the system
