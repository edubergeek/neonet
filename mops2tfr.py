from random import shuffle
import glob
import os
import sys
import numpy as np
import fs
from fs import open_fs
import tensorflow as tf

dirsep = '/'
csvdelim = ','
basePath="/data/nmops20.0/data/testquad6370/CurtsInputs/flattened"
imageText = "image"
inputText = "in"
outputText = "out"
#trainCSV = "./neonet.csv"

# size of object postage stamp cutout in pixels
xdim=32
ydim=32

# percentage of total that are held out test examples 
pTest = 0.1
# percentage of total that are validation examples 
pVal = 0.1

train_filename = '../data/train.tfr'  # the TFRecord file containing the training set
val_filename = '../data/val.tfr'      # the TFRecord file containing the validation set
test_filename = '../data/test.tfr'    # the TFRecord file containing the test set

def chunkstring(string, length):
  return (string[0+i:length+i] for i in range(0, len(string), length))

def normalize(img, threshold):
  val = np.percentile(img,threshold)
  img = img / val
  return img

def normalize_bits(img, bits):
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
  meta = {}
  h = list(chunkstring(hdulist[0].header, 80))
  for index, item in enumerate(h):
    m = str(item)
    mh = list(chunkstring(m, 80))
    #print(mh)
    for ix, im in enumerate(mh):
      #print(index, ix, im)
      mm = im.split('/')[0].split('=')
      if len(mm) == 2:
        #print(index, ix, mm[0], mm[1])
        meta[mm[0].strip()] = mm[1].strip()
  nAxes = int(meta['NAXIS'])
  # check this logic in MOPS FITS files
  if nAxes == 0:
    # check metadata before forcing NAXIS
    nAxes = 3
    maxy, maxx = hdulist[1].data.shape
    data = np.empty((maxy, maxx, 3))
    data[:,:,0] = hdulist[1].data
    data[:,:,1] = hdulist[2].data
    data[:,:,2] = hdulist[3].data
  else:
    data = hdulist[0].data
  data = np.nan_to_num(data)
  #img = data.reshape((maxy, maxx, maxz))
  img = data
  if nAxes == 3:
    maxy, maxx, maxz = data.shape
  else:
    maxy, maxx = data.shape
    maxz = 0
  hdulist.close
  return maxy, maxx, maxz, meta, img

# Generator function to walk path and generate 1 SP3D image set at a time
def process_sp3d(basePath):
  prevImageName=''
  level = 0
  fsDetection = open_fs(basePath)
  for path in fsDetection.walk.files(search='breadth', filter=[inputText]):
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
    fitsName = imageFile.split(dirsep)[-1]
    height, width, depth, imageMeta, imageData = load_fits(imageFile)

    # now the pixels are in the array imageData shape height X width X 1
    # read the truth table from the "out" file
    yName=basePath+dirsep+pointing+dirsep+outputText
    Y=np.genfromtxt(yName, delimiter=",", skip_header=1, usecols=(0,1),names=('detID','actual'))
    Y=np.atleast_1d(Y)
  
    detection=np.ndfromtxt(inName, delimiter=",", skip_header=1, usecols=(0,60,61),names=('detID','x','y'))
    #detection=np.ndfromtxt(inName, delimiter=",", skip_header=1)
    detection=np.atleast_1d(detection)
    for det in detection:
      # for each detection retrieve the 
      print('DetID %f x,y = (%f,%f)'%(det[0], det[1], det[2]))
      stamp, xstamp, ystamp = postage_stamp(imageData[:,:], det[1], det[2], xdim, ydim)
      # unroll the 2D postage stamp into a vector
      #plt.imshow(stamp)
      #plt.show()
      #stamp = np.reshape(stamp, (ydim*xdim))
      # normalize to range 0,2^bits
      stamp = normalize_bits(stamp, 16)
      #stamp = np.reshape(stamp, (ydim,xdim))
      #img = stamp.astype(int)
      #plt.imshow(img)
      #plt.show()
      #printY[np.where(Y[:] == det[0])]
      y=Y[0]
      detID=y[0]
      actual=y[1]
      # det is the warp warp diff image parameter vector from IPP
      #values=[detID, actual]
      #values=np.concatenate((values, stamp))

      # New image so wrap up the current image
      # Flip image Y axis
      #img = np.flip(img, axis=1)
      yield fitsName, actual, stamp, detID, det

  fsDetection.close()
  
def _floatvector_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

np.random.seed()

# open the TFRecords file
train_writer = tf.python_io.TFRecordWriter(train_filename)

# open the TFRecords file
val_writer = tf.python_io.TFRecordWriter(val_filename)

# open the TFRecords file
test_writer = tf.python_io.TFRecordWriter(test_filename)

# find input files in the target dir "basePath"
# it is critical that pairs are produced reliably first level2 then level1
# for each level2 (Y) file
i = nExamples = nTrain = nVal = nTest = 0
for name, isNEO, image, detID, params in process_sp3d(basePath):
  # image is a cutout of a detection within a warp warp diff image
  #image = normalize(image, 95)
  print(name, isNEO, detID)
  xa=np.reshape(image,(xdim*ydim))
  ya = isNEO * 1.0
  dID = detID.astype(bytes)
  feature = {'detID': _bytes_feature(dID), 'isNEO': _float_feature(ya), 'cutout': _floatvector_feature(xa.tolist()), 'params': _floatvector_feature(params.tolist())}
  #feature = {'detID': _bytes_feature(np.fromstring(detID, dtype=bytes, count=detID, sep='.')), 'isNEO': _float_feature(ya), 'cutout': _floatvector_feature(xa.tolist()), 'params': _floatvector_feature(params.tolist())}
  # Create an example protocol buffer
  example = tf.train.Example(features=tf.train.Features(feature=feature))
  nExamples += 1

  # roll the dice to see if this is a train, val or test example
  # and write it to the appropriate TFRecordWriter
  roll = np.random.random()
  if roll >= (pVal + pTest):
    # Serialize to string and write on the file
    train_writer.write(example.SerializeToString())
    nTrain += 1
  elif roll >= pTest:
    # Serialize to string and write on the file
    val_writer.write(example.SerializeToString())
    nVal += 1
  else:
    # Serialize to string and write on the file
    test_writer.write(example.SerializeToString())
    nTest += 1

  if not nExamples % 1000:
    print('%d examples: %03.1f%% train, %03.1f%%validate, %03.1f%%test.'%(nExamples, 100.0*nTrain/nExamples, 100.0*nVal/nExamples, 100.0*nTest/nExamples))
    sys.stdout.flush()
  
train_writer.close()
val_writer.close()
test_writer.close()
print('%d examples: %03.1f%% train, %03.1f%%validate, %03.1f%%test.'%(nExamples, 100.0*nTrain/nExamples, 100.0*nVal/nExamples, 100.0*nTest/nExamples))
sys.stdout.flush()

