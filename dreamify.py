#!/usr/bin/python

# Created by Hawaii Deep Dreams, http://hideepdreams.tumblr.com
# Twitter/Instagram: @hideepdreams | Reddit: 2cats1dog

# Run this script once you've set up the dependencies listed at
# https://github.com/google/deepdream/blob/master/dream.ipynb .
# It should work with the Vagrant setup listed at
# https://www.reddit.com/r/deepdream/comments/3c2s0v/newbie_guide_for_windows/ or
# http://thirdeyesqueegee.com/deepdream/2015/07/19/running-googles-deep-dream-on-windows-with-or-without-cuda-the-easy-way/
# but you'll probably need to change the default save locations (see commends in script).

# Why is this better than Google's IPython notebook?  It's not.  It sucks.
# I'm just more used to using command-line arguments and wanted to adjust this
# script to allow me to do it.  know it's rough; I don't do this professionally.
# I'm just sharing in case someone wants something similar or can use this to
# write a better version.

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe
import sys
import getopt
import os
import re

def usage(): # for getopts
    print 'dreamify.py -o <originalfile> [ -s <savefile> -g <guidefile> -e <end/layer> -t <step_size> -j <jitter> -i <iter_n> -v <octave_n> -l <octave_scale> ]'
    sys.exit(2)

def main(argv): # Set variables for output
    global original_location, guide_location, save_location, end, step_size, jitter, iter_n, octave_n, octave_scale, found_o, found_g, found_s

    found_o = False
    found_g = False
    found_s = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "o:s:g:h:e:t:j:i:v:l:")
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt =='-h':
            usage()
        elif opt in ("-o"):
            original_location = arg
            original_name = os.path.splitext(os.path.basename(original_location))[0]
            found_o = True
        elif opt in ("-s"):
            save_location = arg
            found_s = True
        elif opt in ("-g"):
            guide_location = arg
            guide_name = os.path.splitext(os.path.basename(guide_location))[0]
            found_g = True
        elif opt in ("-e"):
            end = arg
        elif opt in ("-t"):
            step_size = float(arg)
        elif opt in ("-j"):
            jitter = float(arg)
        elif opt in ("-i"):
            iter_n = int(arg)
        elif opt in ("-v"):
            octave_n = int(arg)
        elif opt in ("-l"):
            octave_scale = float(arg)

    end_name = re.sub('[^a-zA-Z0-9-*.]', '-', end)

    if not found_o:
        usage()
    if not found_g:
        guide_name = 'none'
    if not found_s:
        #Default save location will need to be edited for your file structure; on Win (not vagrant/vm) it might look like 'D:/Pictures/deepdream/dream/'
        save_location = '~/Pictures/dream/' + original_name + '_g-' + guide_name + '_e-' + end_name + '_t-' + str(step_size) + '_j-' + str(jitter) + '_i-' + str(iter_n) + '_v-' + str(octave_n) + '_l-' + str(octave_scale) + '.png'

# Set defaults
end = 'inception_4c/output'
step_size = 1.5
jitter = 32
iter_n = 10
octave_n = 4
octave_scale = 1.4

if __name__ == "__main__":
    main(sys.argv[1:])

# This is where the meat of the script starts.

def savearray(a, filename, fmt='png'):
    a = np.uint8(np.clip(a, 0, 255))
    with open(os.path.expanduser(filename), 'wb') as f:
        PIL.Image.fromarray(a).save(f, fmt)
        #display(Image(data=f.getvalue()))

model_path = '../caffe/models/bvlc_googlenet/' # substitute your path here; on Win (not vagrant/vm) it might look like  'caffe/models/bvlc_googlenet/'
net_fn   = model_path + 'deploy.prototxt'
param_fn = model_path + 'bvlc_googlenet.caffemodel'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Classifier('tmp.prototxt', param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def objective_L2(dst):
    dst.diff[:] = dst.data

def make_step(net, step_size=1.5, end='inception_4c/output',
              jitter=32, clip=True, objective=objective_L2):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    objective(dst)  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4,
              end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end, vis.shape
            clear_output(wait=True)

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# Guiding stuff
def objective_guide(dst):
    x = dst.data[0].copy()
    y = guide_features
    ch = x.shape[0]
    x = x.reshape(ch,-1)
    y = y.reshape(ch,-1)
    A = x.T.dot(y) # compute the matrix of dot-products with guide features
    dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

# Open up source image
img = np.float32(PIL.Image.open(original_location))

if found_g:
    # Prep for guiding
    guide = np.float32(PIL.Image.open(guide_location))
    h, w = guide.shape[:2]
    src, dst = net.blobs['data'], net.blobs[end]
    src.reshape(1,3,h,w)
    src.data[0] = preprocess(net, guide)
    net.forward(end=end)
    guide_features = dst.data[0].copy()

    _=deepdream(net, img, end=end, step_size=step_size, jitter=jitter, iter_n=iter_n, objective=objective_guide, octave_n=octave_n, octave_scale=octave_scale)
else:
    _=deepdream(net, img, end=end, step_size=step_size, jitter=jitter, iter_n=iter_n, octave_n=octave_n, octave_scale=octave_scale)

savearray(_, save_location)
