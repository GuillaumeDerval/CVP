# coding=utf-8
import os
import glob
import subprocess
from multiprocessing import Pool, freeze_support


def launch_expofus(args):
    srcpath, destpath = args
    print "Launching exposition fusion on : " + srcpath
    p = subprocess.Popen("python exposurefusion.py --withiqa \"" + srcpath + "\" \"" + destpath + "\"")
    p.wait()

def launch_matlab(args):
    srcpath, destpath = args
    print "Launching IQA on :" + srcpath + " - " + destpath
    DEVNULL = open(os.devnull, 'wb')
    p = subprocess.Popen("matlab -wait -nodesktop -nosplash -r \"addpath('mef_iqa'); "
                         "iqa('" + srcpath + "', '" + destpath + "', 1, '"+ destpath+"_iqa.fig', '" + destpath + "_iqa.txt')",
                         shell=True, stdout=DEVNULL, stderr=DEVNULL)
    p.wait()
    
mefargs = []
iqaargs = []
for dirname, dirnames, filenames in os.walk('static_input'):
    # For each subdir (image set)    
    for subdirname in dirnames:
        srcpath = os.path.join(dirname, subdirname)
        destpath = os.path.join('static_output', subdirname + '.png')
        
        # Add MEF command
        mefargs.append((srcpath, destpath))
        
        # Add IQA commands
        for filename in glob.glob('static2_output/'+subdirname+'*'):
            iqaargs.append((srcpath, filename))
                        

if __name__ == '__main__':
    freeze_support()        
    Pool(4).map(launch_expofus, mefargs)
    #Pool(4).map(launch_matlab, iqaargs)