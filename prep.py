import os
import numpy as np
#import matplotlib.pyplot as plt
import fs
from fs import open_fs

dirsep = '/'
csvdelim = ','
basePath="/data/nmops20.0/data/mops/deeplearning/TestQuad/20180518.r40242/CurtsInputs/flattened"
imageText = "image"
inputText = "in"
outputText = "out"
trainCSV = "./neonet.csv"
xdim=32
ydim=32

def normalize(img, bits):
  vlo=np.amin(img)
  vhi=np.amax(img)
  vrg=vhi-vlo
  img = (img - vlo) / vrg * 2**(bits-1)
  return img

def postage_stamp(img,x,y,w,h):
  iy,ix = img.shape
  xo = int(w/2)
  yo = int(h/2)
  cx = min(max(np.floor(x).astype(int)-xo,0),ix-w)
  cy = min(max(np.floor(iy-y).astype(int)-yo,0),iy-h)
  crop=img[cy:cy+h, cx:cx+w]
  return crop, cx, cy

def load_fits(filnam):
  from astropy.io import fits

  hdulist = fits.open(filnam)
  meta = hdulist[0].header
  data = hdulist[0].data
  data = np.nan_to_num(data)
  maxy, maxx = data.shape
  img = data.reshape((maxy, maxx, 1))
#  fits.close(hdulist)
  return maxy, maxx, meta, img

# main
# find input files in the target dir "basePath"
#plt.gray()
outFileHandle=open(trainCSV,'wt')
fsDetection = open_fs(basePath)
for path in fsDetection.walk.files(filter=[inputText]):
  # process each "in" file of detections
  inName=basePath+path
  print('Inspecting %s'%(inName))
  #open the warp warp diff image using "image" file
  sub=path.split(dirsep)
  pointing=sub[1]
  imgName=basePath+dirsep+pointing+dirsep+imageText
  byteArray=bytearray(np.genfromtxt(imgName, 'S'))
  imageFile=byteArray.decode()
  print("Opening image file %s"%(imageFile))
  height, width, imageMeta, imageData = load_fits(imageFile)


  # now the pixels are in the array imageData shape height X width X 1
  # read the truth table from the "out" file
  yName=basePath+dirsep+pointing+dirsep+outputText
  Y=np.genfromtxt(yName, delimiter=",", skip_header=1, usecols=(0,1),names=('detID','actual'))

  detection=np.genfromtxt(inName, delimiter=",", skip_header=1, usecols=(0,60,61),names=('detID','x','y'))
  print('Shape of detection: %s'%(detection.shape))
  for det in detection:
    # for each detection retrieve the 
    print('DetID %d x,y = (%f,%f)'%(det[0], det[1], det[2]))
    stamp, xstamp, ystamp = postage_stamp(imageData[:,:,0], det[1], det[2], xdim, ydim)
    # unroll the 2D postage stamp into a vector
    #plt.imshow(stamp)
    #plt.show()
    stamp = np.reshape(stamp, (ydim*xdim))
    # normalize to range 0,2^bits
    stamp = normalize(stamp, 16)
    #stamp = np.reshape(stamp, (ydim,xdim))
    #img = stamp.astype(int)
    #plt.imshow(img)
    #plt.show()
    #printY[np.where(Y[:] == det[0])]
    y=Y[0]
    detID=y[0]
    actual=y[1]
    values=[detID, actual]
    values=np.concatenate((values, stamp))
    np.savetxt(outFileHandle,values[None],delimiter=csvdelim)

outFileHandle.close()
