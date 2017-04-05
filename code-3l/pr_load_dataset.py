#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

"""pr_load_dataset.py: This file is to help loading the signs dataset into
TensorFlow more easily. Mainly based on the code here:
  tensorflow/tensorflow/contrib/learn/python/learn/datasets/mnist.py"""

__author__      = "Pablo Rivas"
__copyright__   = "Copyright 2017, Pablo Rivas Perea"
__credits__     = ["pablorp80", "tensorflower-gardener", "ilblackdragon",
                   "velaia", "rohan100jain", "Lezcano", "LaurenLuoYun"]
__license__     = "GPL"
__version__     = "1.0.1"
__maintainer__  = "Pablo Rivas"
__email__       = "Pablo.Rivas@Marist.edu"
__status__      = "Development"

from tensorflow.contrib.learn.python.learn.datasets import base
from PIL import Image
import glob
import numpy as np

class DataSet(object):

  def __init__(self,
               images,
               labels,
               one_hot=False,
               dtype=np.uint8,
               reshape=True):
    """Construct a DataSet.
    `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    if reshape:
      assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2])
    if dtype == np.float32:
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, shuffle=True):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    # Shuffle for the first epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      perm0 = np.arange(self._num_examples)
      np.random.shuffle(perm0)
      self._images = self.images[perm0]
      self._labels = self.labels[perm0]
    # Go to the next epoch
    if start + batch_size > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Get the rest examples in this epoch
      rest_num_examples = self._num_examples - start
      images_rest_part = self._images[start:self._num_examples]
      labels_rest_part = self._labels[start:self._num_examples]
      # Shuffle the data
      if shuffle:
        perm = np.arange(self._num_examples)
        np.random.shuffle(perm)
        self._images = self.images[perm]
        self._labels = self.labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size - rest_num_examples
      end = self._index_in_epoch
      images_new_part = self._images[start:end]
      labels_new_part = self._labels[start:end]
      return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
    else:
      self._index_in_epoch += batch_size
      end = self._index_in_epoch
      return self._images[start:end], self._labels[start:end]




def load_images(f, sub):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
  Args:
    f: A string that matches a specific pattern of files in a directory.
    sub: A list of strings indicating the subjects to process.
  Returns:
    data: A 4D uint8 numpy array [index, y, x, depth].
  """
  print('Loading images ', f, sub)
  filelist = glob.glob(f)
  
  res=[]
  for aFile in filelist:
    for s in sub:
      if s in aFile:
        res.append(aFile)

  filelist = res

  # open the first to get dimensions
  tmp = np.array(Image.open(filelist[0]));
  num_images = len(filelist)
  rows = tmp.shape[0]
  cols = tmp.shape[1]
  print("Number of images: %d" % (num_images))
  print("Number of rows: %d" % (rows))
  print("Number of cols: %d" % (cols))

  data = np.array([np.array(Image.open(fname)) for fname in filelist])
  data = data.reshape(num_images, rows, cols, 1)

  print("Dataset array size: ", data.shape)

  return data

def extract_labels(f, sub, one_hot=False, num_classes=31):
  """Extract the labels into a 1D uint8 numpy array [index].
  Args:
    f: A string that matches a specific pattern of files in a directory.
    sub: A list of strings indicating the subjects to process.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D uint8 numpy array.
  """
  print('Extracting labels from data in ', f, sub)
  filelist = glob.glob(f)

  res=[]
  for aFile in filelist:
    for s in sub:
      if s in aFile:
        res.append(aFile)

  filelist = res


  labels = np.array([np.array(fn[-12:-10], dtype=np.uint8) for fn in filelist])

  if one_hot:
    return dense_to_one_hot(labels, num_classes)

  return labels


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  labels_dense = labels_dense - 1
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def read_data_sets(dataset_dir_pattern,
                   tr_sub, te_sub=[], val_sub=[],
                   one_hot=False,
                   dtype=np.uint8,
                   reshape=True):

  train_images = load_images(dataset_dir_pattern, tr_sub)
  train_labels = extract_labels(dataset_dir_pattern, tr_sub, one_hot=one_hot)

  if (not te_sub) and (not val_sub):
    pct_tr = 0.5  
    pct_te = 0.25
    pct_val = 0.25
    idx = np.arange(train_labels.shape[0])
    np.random.shuffle(idx)
    idx_tr = idx[0:np.int(len(idx)*pct_tr)]
    idx_te = idx[np.int(len(idx)*pct_tr):np.int(len(idx)*(pct_tr+pct_te))]
    idx_val = idx[np.int(len(idx)*(pct_tr+pct_te)):]

    test_images = train_images[idx_te, :, :, :]
    test_labels = train_labels[idx_te, :]
    
    validation_images = train_images[idx_val, :, :, :]
    validation_labels = train_labels[idx_val, :] 
    
    train_images = train_images[idx_tr, :, :, :]
    train_labels = train_labels[idx_tr, :]
  
  else:
    test_images = load_images(dataset_dir_pattern, te_sub)
    test_labels = extract_labels(dataset_dir_pattern, te_sub, one_hot=one_hot)

    validation_images = load_images(dataset_dir_pattern, val_sub)
    validation_labels = extract_labels(dataset_dir_pattern, 
                                       val_sub, 
                                       one_hot=one_hot)

  print(train_images.shape) 
  print(train_labels.shape) 
  print(test_images.shape) 
  print(test_labels.shape) 
  print(validation_images.shape) 
  print(validation_labels.shape) 
  
  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)

