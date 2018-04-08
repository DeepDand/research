import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import time
import glob
import numpy as np
from skimage import io
from sklearn.decomposition import TruncatedSVD 
from sklearn.manifold import TSNE
from ggplot import *
from collections import OrderedDict

aef = sys.argv[1]
#get data
PATH='/home/cc/research/dataset/'
#PATH='/Users/pablorp80/Documents/Faculty/marist/papermlj17/repo/dataset/'
#datafiles = glob.glob(PATH + '*00.jpeg')
datafiles = glob.glob(PATH + '*.jpeg')
img = io.imread(datafiles[0])
N = len(datafiles)
M = img.size
print N
print M

X = np.empty((N,M), dtype=np.float32)
y = []

signs = {'01':'A',
         '02':'B',
         '03':'C',
         '04':'D',
         '05':'E',
         '06':'F',
         '07':'G',
         '08':'H',
         '09':'I',
         '10':'K',
         '11':'L',
         '12':'M',
         '13':'N',
         '14':'O',
         '15':'P',
         '16':'Q',
         '17':'R',
         '18':'S',
         '19':'T',
         '20':'U',
         '21':'2/V',
         '22':'6/W',
         '23':'X',
         '24':'Y',
         '25':'1',
         '26':'3',
         '27':'4',
         '28':'5',
         '29':'7',
         '30':'8',
         '31':'9'}

colmap = {'A': '#008941',
          'B': '#FF2F80',
          'C': '#8FB0FF',
          'D': '#FFAA92',
          'E': '#000000',
          'F': '#4A3B53',
          'G': '#BA0900',
          'H': '#FF4A46',
          'I': '#61615A',
          'K': '#FEFFE6',
          'L': '#FF90C9',
          'M': '#1CE6FF',
          'N': '#FFDBE5',
          'O': '#006FA6',
          'P': '#A30059',
          'Q': '#B903AA',
          'R': '#997D87',
          'S': '#FFFF00',
          'T': '#FF34FF',
          'U': '#5A0007',
        '2/V': '#00C2A0',
        '6/W': '#63FFAC',
          'X': '#6B7900',
          'Y': '#7A4900',
          '1': '#4FC601',
          '3': '#004D43',
          '4': '#3B5DFF',
          '5': '#1B4400',
          '7': '#B79762',
          '8': '#0000A6',
          '9': '#809693'}

n = 0
for filename in sorted(datafiles):
  #print(filename)
  img = io.imread(filename)
  X[n,:] = img.flatten() / 255.0
  cls = filename[-12:-10]
  #print cls
  y.append(signs[cls])
  n = n+1

feat_cols = ['pixel'+str(i) for i in range(M)]
df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y
df['colors'] = df['label'].map(colmap)

X, y = None, None

print 'Size of the dataframe: {}'.format(df.shape)
rndperm = np.random.permutation(N)


time_start = time.time()
datafiles = sorted(glob.glob('./'+aef+'/chkpts/filters?.npy'))
ae_results = np.array([])
layers = 0
for fname in datafiles:
  print fname
  layers = layers + 1
  ae = np.load(fname)
  b = np.load('./'+aef+'/chkpts/biases'+ fname[20] +'.npy')
  if ae_results.size == 0:
    ae_results = df[feat_cols].values.dot(ae) + b
  else:
    ae_results = ae_results.dot(ae) + b
  print ae_results.shape

print ae_results.shape
print 'autoencoder done! Time elapsed: {} seconds'.format(time.time()-time_start)
df['x-ae'] = ae_results[:,0]
df['y-ae'] = ae_results[:,1]
ae, ae_results = None, None

chart = ggplot( df, aes(x='x-ae', y='y-ae', color='label') ) \
        + geom_point(size=50,alpha=0.5) \
        + ggtitle("Autoencoder dimensions colored by Sign")
chart.make()
fig = plt.gcf()
ax = plt.gca()
plt.savefig("ae-"+str(layers)+".png", dpi=350)
plt.close()
chart = None
print "first chart done!"

for index, row in df[['x-ae','y-ae','colors','label']].iterrows():
  print index
  x = row['x-ae']
  y = row['y-ae']
  co = row['colors']
  lab = row['label']
  plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
        borderaxespad=0, fontsize= 'xx-small')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.title("Autoencoder visualization colored by Sign")
plt.savefig("ae-"+str(layers)+"-m.png", dpi=350)
plt.close()

handles, labels, by_label = None, None, None
gc.collect()
print "second chart done!"



time_start = time.time()
datafiles = sorted(glob.glob('./'+aef+'/chkpts/filters?ft.npy'))
ae_results = np.array([])
layers = 0
for fname in datafiles:
  print fname
  layers = layers + 1
  ae = np.load(fname)
  b = np.load('./'+aef+'/chkpts/biases'+ fname[20] +'ft.npy')
  if ae_results.size == 0:
    ae_results = df[feat_cols].values.dot(ae) + b
  else:
    ae_results = ae_results.dot(ae) + b
  print ae_results.shape

print ae_results.shape
print 'autoencoder done! Time elapsed: {} seconds'.format(time.time()-time_start)
df['x-ae'] = ae_results[:,0]
df['y-ae'] = ae_results[:,1]
ae, ae_results = None, None

chart = ggplot( df, aes(x='x-ae', y='y-ae', color='label') ) \
        + geom_point(size=50,alpha=0.5) \
        + ggtitle("Autoencoder dimensions colored by Sign")
chart.make()
fig = plt.gcf()
ax = plt.gca()
plt.savefig("ae-ft-"+str(layers)+".png", dpi=350)
plt.close()
chart = None
print "first chart done!"

for index, row in df[['x-ae','y-ae','colors','label']].iterrows():
  print index
  x = row['x-ae']
  y = row['y-ae']
  co = row['colors']
  lab = row['label']
  plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
        borderaxespad=0, fontsize= 'xx-small')
plt.xlabel('First dimension')
plt.ylabel('Second dimension')
plt.title("Autoencoder visualization colored by Sign")
plt.savefig("ae-ft-"+str(layers)+"-m.png", dpi=350)
plt.close()

handles, labels, by_label = None, None, None
gc.collect()
print "second chart done!"



#for n in np.logspace(2, 16, num=15, base=2.0, dtype=np.int32):
#  print n
#  pca_n = TruncatedSVD(n_components=n, algorithm='arpack')
#  pca_result_n = pca_n.fit_transform(df[feat_cols].values)
#  print 'Explained variation per principal component (PCA): {}'.format(np.sum(pca_n.explained_variance_ratio_))
#  time_start = time.time()
#  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#  tsne_pca_results = tsne.fit_transform(pca_result_n)
#  pca_n, pca_result_n = None, None
#  print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
#  df['x-tsne-pca'] = tsne_pca_results[:,0]
#  df['y-tsne-pca'] = tsne_pca_results[:,1]
#  tsne_pca_results, tsne = None, None
#  
#  chart = ggplot( df, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
#          + geom_point(size=50,alpha=0.5) \
#          + ggtitle("tSNE dimensions colored by Sign (PCA)")
#  chart.make()
#  fig = plt.gcf()
#  ax = plt.gca()
#  plt.savefig("pca-tsne"+str(n)+".png", dpi=350)
#  plt.close()
#  chart = None
#  
#  for index, row in df[['x-tsne-pca','y-tsne-pca','colors','label']].iterrows():
#      x = row['x-tsne-pca']
#      y = row['y-tsne-pca']
#      co = row['colors']
#      lab = row['label']
#      plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)
#  
#  handles, labels = plt.gca().get_legend_handles_labels()
#  by_label = OrderedDict(zip(labels, handles))
#  plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
#          borderaxespad=0, fontsize= 'xx-small')
#  plt.xlabel('First dimension')
#  plt.ylabel('Second dimension')
#  plt.title("PCA-tSNE visualization colored by Sign")
#  plt.savefig("pca-tsne-m"+str(n)+".png", dpi=350)
#  plt.close()
#  
#  handles, labels, by_label = None, None, None
#  gc.collect()


#pca = TruncatedSVD(n_components=2, algorithm='arpack')
#pca_result = pca.fit_transform(df[feat_cols].values)
#df['pca-one'] = pca_result[:,0]
#df['pca-two'] = pca_result[:,1] 
#print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)
#pca, pca_result = None, None
#
#chart = ggplot( df, aes(x='pca-one', y='pca-two', color='label') ) \
#        + geom_point(size=50,alpha=0.5) \
#        + ggtitle("First and Second Principal Components colored by sign")
#chart.make()
#fig = plt.gcf()
#ax = plt.gca()
#plt.savefig("pca.png", dpi=350)
#plt.close()
#chart = None
#
#for index, row in df[['pca-one','pca-two','colors','label']].iterrows():
#    x = row['pca-one']
#    y = row['pca-two']
#    co = row['colors']
#    lab = row['label']
#    plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)
#
#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = OrderedDict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
#        borderaxespad=0, fontsize= 'xx-small')
#plt.xlabel('First dimension')
#plt.ylabel('Second dimension')
#plt.title("PCA visualization colored by Sign")
#plt.savefig("pca-m.png", dpi=350)
#plt.close()
#
#handles, labels, by_label = None, None, None
#gc.collect()


#for n in np.logspace(13.91, 14, num=10, base=2.0, dtype=np.int32):
#  print n
#  time_start = time.time()
#  tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#  tsne_results = tsne.fit_transform(df.loc[rndperm[:n],feat_cols].values)
#  print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
#  df_tsne = df.loc[rndperm[:n],:].copy()
#  df_tsne['x-tsne'] = tsne_results[:,0]
#  df_tsne['y-tsne'] = tsne_results[:,1]
#  tsne, tsne_results = None, None
#  gc.collect()
#  
#  chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
#          + geom_point(size=50,alpha=0.5) \
#          + ggtitle("tSNE dimensions colored by sign")
#  chart.make()
#  fig = plt.gcf()
#  ax = plt.gca()
#  plt.savefig("tsne-"+str(n)+".png", dpi=350)
#  plt.close()
#  
#  chart = None
#  gc.collect()
#  
#  for index, row in df_tsne[['x-tsne','y-tsne','colors','label']].iterrows():
#      x = row['x-tsne']
#      y = row['y-tsne']
#      co = row['colors']
#      lab = row['label']
#      plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)
#  
#  handles, labels = plt.gca().get_legend_handles_labels()
#  by_label = OrderedDict(zip(labels, handles))
#  plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
#          borderaxespad=0, fontsize= 'xx-small')
#  plt.xlabel('First dimension')
#  plt.ylabel('Second dimension')
#  plt.title("tSNE visualization colored by Sign")
#  plt.savefig("tsne-m-"+str(n)+".png", dpi=350)
#  plt.close()
#  
#  handles, labels, by_label, df_tsne = None, None, None, None
#  gc.collect()



#time_start = time.time()
#tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
#tsne_results = tsne.fit_transform(df.loc[:,feat_cols].values)
#print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
#df['x-tsne'] = tsne_results[:,0]
#df['y-tsne'] = tsne_results[:,1]
#tsne, tsne_results = None, None
#gc.collect()
#
#chart = ggplot( df, aes(x='x-tsne', y='y-tsne', color='label') ) \
#        + geom_point(size=50,alpha=0.5) \
#        + ggtitle("tSNE dimensions colored by sign")
#chart.make()
#fig = plt.gcf()
#ax = plt.gca()
#plt.savefig("tsne.png", dpi=350)
#plt.close()
#
#chart = None
#gc.collect()
#
#for index, row in df[['x-tsne','y-tsne','colors','label']].iterrows():
#    x = row['x-tsne']
#    y = row['y-tsne']
#    co = row['colors']
#    lab = row['label']
#    plt.scatter(x, y, c=co, label=lab, s=10, alpha=0.5)
#
#handles, labels = plt.gca().get_legend_handles_labels()
#by_label = OrderedDict(zip(labels, handles))
#plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.1,1),
#        borderaxespad=0, fontsize= 'xx-small')
#plt.xlabel('First dimension')
#plt.ylabel('Second dimension')
#plt.title("tSNE visualization colored by Sign")
#plt.savefig("tsne-m.png", dpi=350)
#plt.close()
#
#handles, labels, by_label = None, None, None
#gc.collect()

