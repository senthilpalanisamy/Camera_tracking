1. Go to epix download website(https://www.epixinc.com/support/files.php)
2. If the link is down, go to their home website (https://www.epixinc.com/) and
   try searching for software downlads.
3. We have license for xclib (C++ library for accessing and controlling
   frame grabber), XCAP (a software tool for configuring framegrabbers and
   cameras) and PXIPL (A high speed image processing library that acts
   directly on the image stored in frame grabber memory). Never used PXIPL
   since I always preferred OpenCV but give it a try if you think its worth it.
4. Download latest XCAP version (3.8 at the time of writing this post on
   12/11/2020) and latest XCLIB + PXIPL (3.8 at the time of writing this post)
   for your target platform (Most likely it is linux 64)
5. These installers should be run as root(su, type root password and then
   run these installers)
6. When you run both of these, the program normally prompts for a license
   code. Ours is a USB based license and the number is attached at the back 
   of USB (The number is MO2K/XRC5/N9WL). If you are lucky, the installation
   will be successful on the first go
7. If XCAP screen comes alive but the drivers are not installed and they failed, 
   at the top  the window go and 
   select PIXCI --> PIXCI Open / Close --> Driver Assisstant
   for trying out different versions of kernel or pre-compiled kernels
8. The results of the compilation process will be shown on a small white window.
   If something fails, export the whole output from the window as a log file
   and analyse the log file to fix issues

Note:-
Things can go wrong. The installation process was not very smooth and it was
rough. Things to be aware of:0
1. check the linux kernel version. Not all version  of kernel are supported cleanly. 
   Try using different version. I was able to  get this working in 5.0.62 version of kernel. 
2. If it is a brand new machine, confirm that "make" is installed
3. There are two ways to install xcap drivers. Compiling for the current kernel
   or testing if a precompiled version works on the same kernel. If the installation
   doesn't go smoothly, consider trying all options
