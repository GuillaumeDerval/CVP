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

----------------------

Tools :

run.py : Runs the exposure fusion + IQA on images folder found in "static_input" folder
         Also runs IQA on corresponding images filename "_method" in the "static2_output" folder
         Used to generate data for comparisons...        
parse_iqa_results.py : Take the output of run.py and generate a nice CSV file for putting on report
optimize.py	: Look for the paramaters that maximizes IQA
optimize_makeplots.py : Make plots from results of optimize.py
image2gif.py : Generate GIF images of fused images varying with paramaters