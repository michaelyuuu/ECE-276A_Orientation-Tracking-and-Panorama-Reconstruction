import pickle
import sys
import time 
import numpy as np
import matplotlib.pyplot as plt

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="9"
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")

# print(type(camd), len(camd) if hasattr(camd, "__len__") else "no len")
# print(type(imud), len(imud) if hasattr(imud, "__len__") else "no len")
# print(type(vicd), len(vicd) if hasattr(vicd, "__len__") else "no len")
# print("cam keys:", camd.keys() if hasattr(camd, "keys") else "no keys")
# print(camd['cam'].shape)
# plt.imshow(camd['cam'][:100,:100,:,0])
# plt.figure()
# plt.imshow(camd['cam'][:,:,:,0])
# plt.show(block=True)