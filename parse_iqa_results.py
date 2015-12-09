# coding=utf-8
import os
import glob
import csv

data = {}
for dirname, dirnames, filenames in os.walk('static_input'):
    # For each subdir (image set)    
    for subdirname in dirnames:
        srcpath = os.path.join(dirname, subdirname)
        destpath = os.path.join('static_output', subdirname + '.png')
        
        for filename in glob.glob('static2_output/'+subdirname+'*.txt'):
            method = filename[len('static2_output/'+subdirname)+1:-12]
            with open(filename, 'r') as f:
                if subdirname not in data:
                    data[subdirname] = {}
                data[subdirname][method] = f.read()
                
        for filename in glob.glob('static_output/'+subdirname+'*.txt'):
            with open(filename, 'r') as f:
                if subdirname not in data:
                    data[subdirname] = {}
                data[subdirname]["ourmethod"] = f.read()


with open('results.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Image'] + data[data.keys()[0]].keys())
    for image in data.keys():
        print image
        writer.writerow([image] + data[image].values())