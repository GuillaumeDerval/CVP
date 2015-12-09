Important notes
---------------

Requirements to run the whole code :
- Python 2.7 (with numpy and scipy)
- OpenCV WITH ximgproc module
- MATLAB

----------------------

Usage : python exposurefusion.py [-h] [--debug] [--dynamic] [--filterbycolor]
                         [--withiqa] [--sigma SIGMA] [--sigmaColor SIGMACOLOR]
                         [--sigmaSpace SIGMASPACE]
                         source output

-sigmaX parameters are the one defined in report.
-dynamic parameter enable dynamic exposure fusion
-withiqa parameter runs IQA at the end of exposure fusion : Two new files are created.

source : source folder where to find the different exposures of one image
output : output file (fused image)

----------------------

Submodules :

lib.py : Main functions of the gradient-based exposure fusion algorithm
mef_iqa/iqa.m : Main MATLAB program to run IQA. Takes the same arguments as exposurefusion.py as source and output
mef_iqa/mef_ms_ssim.m : Multi-scale IQA algorithm
mef_iqa/mef_ssim.m : Mono-scale IQA algorithm