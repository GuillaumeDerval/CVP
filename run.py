# coding=utf-8
import os
import subprocess

for dirname, dirnames, filenames in os.walk('static_input'):
    # For each subdir (image set)
    cnt = 0
    p = [0,0,0,0]
    for subdirname in dirnames:
        srcpath = os.path.join(dirname, subdirname)
        destpath = os.path.join('static_output', subdirname + '.png')
        print "Processing" + srcpath
        print "Laucnhing cmd : " + "python exposurefusion.py --withiqa \"" + srcpath + "\" \"" + destpath + "\""
        p[cnt % 4] = subprocess.Popen("python exposurefusion.py --withiqa \"" + srcpath + "\" \"" + destpath + "\"")
        cnt += 1
        
        if cnt % 4 == 0:
            for i in range(0,4):
                p[i].wait()