import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA
import os
from PIL import Image
import csv
import time

def create_labels(labeled_data):
    label=''
    if labeled_data['Eyeglasses'] == 1: label+='Eyeglasses_'
    if labeled_data['Mustache'] == 1: label+='Mustache_'  
    if labeled_data['Smiling']  == 1: label+='Smiling_' 
    if labeled_data['Wearing_Hat'] == 1: label+='Wearing_Hat'
    if label=='': label='Neutral'
    return label

def create_digits(labeled_data):
    label=0
    if labeled_data['Eyeglasses'] == 1: label = 1 
    if labeled_data['Mustache'] == 1: label = 2  
    if labeled_data['Smiling']  == 1: label = 3 
    if labeled_data['Wearing_Hat'] == 1: label = 4
    return label

def convert_64_64(data):
    dirname = '../img_align_celeba/'
    converted_64_64 = []

    outputCsvFileName = 'converted_64_64.csv'
    file = open(outputCsvFileName,"w")
    writer = csv.writer(file)
    for i in range(len(train)):
        #for i in range(0,50):
        im_file = train.loc[i].Image
        #    print(file)
        image = Image.open(os.path.join(dirname, im_file))
        image = image.convert('L')
        image = image.resize((64, 64),Image.BICUBIC)
        image.load()
        image = np.array(image)
        image = image.reshape(4096)
        writer.writerow(image)
        if i % n == 0:
            print("Processed :", i)
        file.close()
        print('Table saved to the file ',outputCsvFileName)
        
def do_balanced(data, samples):
    Y = data.apply(create_labels, axis = 1)
    Y_Neutral =Y[Y == 'Neutral']
    print('Neutral:',Y_Neutral.size)
    Y_Smiling =Y[Y == 'Smiling_']
    print('Smiling:',Y_Smiling.size)
    neutral_removes = Y_Neutral.sample(n=Y_Neutral.size-samples)
    smiling_removes = Y_Smiling.sample(n=Y_Smiling.size-samples)
    df_rest = Y.loc[~(Y.index.isin(neutral_removes.index)|Y.index.isin(smiling_removes.index))]
    Y_data_balanced = data.loc[data.index.isin(df_rest.index)]
    return Y_data_balanced

def do_minimal(data, samples):
    Y = data.apply(create_labels, axis = 1)
    Y_Neutral =Y[Y == 'Neutral']
    print('Neutral:',Y_Neutral.size)
    Y_Eyeglasses =Y[Y == 'Eyeglasses_']
    print('Eyeglasses:',Y_Eyeglasses.size)
    Y_Mustache =Y[Y == 'Mustache_']
    print('Mustache:',Y_Mustache.size)
    Y_Wearing_Hat =Y[Y == 'Wearing_Hat']
    print('Wearing_Hat:',Y_Wearing_Hat.size)
    Y_Smiling =Y[Y == 'Smiling_']
    print('Smiling:',Y_Smiling.size)
    neutral = Y_Neutral.sample(n=samples)
    eyeglasses = Y_Eyeglasses.sample(n=samples)
    mustache = Y_Mustache.sample(n=samples)
    wearing_hat = Y_Wearing_Hat.sample(n=samples)
    smiling = Y_Smiling.sample(n=samples)
    df_sub = Y.loc[(Y.index.isin(neutral.index)|Y.index.isin(eyeglasses.index)|Y.index.isin(mustache.index)
                     |Y.index.isin(wearing_hat.index)|Y.index.isin(smiling.index))]
    Y_data_balanced = data.loc[data.index.isin(df_sub.index)]
    return Y_data_balanced

def convert_64_64(data):
    dirname = '../img_align_celeba/'
    print('Data size:',len(data))
    n= 5000
    outputCsvFileName = 'converted_64_64.csv'
    file = open(outputCsvFileName,"w")
    writer = csv.writer(file)
    for i in range(len(data)):
#    for i in range(0,5):
        im_file = data.loc[i].Image
#        print(im_file)
        image = Image.open(os.path.join(dirname, im_file))
        image = image.convert('L')
        image = image.resize((64, 64),Image.BICUBIC)
        image.load()
        image = np.array(image)
        image = image.reshape(4096)
        writer.writerow(image)
        if i % n == 0:
            print("Processed :", i)
    file.close()
    print('Data temporary saved to the file ',outputCsvFileName)
    conv_64 = pd.read_csv(outputCsvFileName,header = None)
    data_conv_64 = pd.merge(data, conv_64, left_index=True, right_index=True, how='outer')
    return data_conv_64

def convert_128_128(data):
    dirname = '../img_align_celeba/'
    print('Data size:',len(data))
    n= 5000
    outputCsvFileName = 'converted_128.csv'
    file = open(outputCsvFileName,"w")
    writer = csv.writer(file)
    for i in range(len(data)):
#    for i in range(0,5):
        im_file = data.loc[i].Image
#    print(file)
        image = Image.open(os.path.join(dirname, im_file))
        image = image.convert('L')
        image = image.resize((128, 128),Image.BICUBIC)
        image.load()
        image = np.array(image)
        image = image.reshape(16384)
        writer.writerow(image)
        if i % n == 0:
            print("Processed :", i)
    file.close()
    print('Data temporary saved to the file ',outputCsvFileName)
    conv_128 = pd.read_csv(outputCsvFileName,header = None)
    data_conv_128 = pd.merge(data, conv_128, left_index=True, right_index=True, how='outer')
    return data_conv_128

def make_NN_dataset_128(data):
    X = data.drop(['Image','Eyeglasses','Mustache','Smiling','Wearing_Hat','Type'], axis=1)
    X = X.astype(np.float32)
    X = np.array(X)
    
    Y_data = data[['Eyeglasses','Mustache','Smiling','Wearing_Hat']]
#    Y = data[['Eyeglasses','Mustache','Smiling','Wearing_Hat']]
#    Y[Y <0] = 0.
#    Y = Y.astype(np.float)
#    Y = np.array(Y)
    Y = Y_data.apply(create_digits, axis = 1)
    
    inputs = [np.reshape(x, (16384, 1)) for x in X]
    inputs =[np.float32(x) for x in inputs]

    results = [vectorized_result(y) for y in Y]
#    results = [np.reshape(y, (4, 1)) for y in Y]

    dataset = zip(inputs, results)
    return dataset

def make_NN_dataset_64(data):
    X = data.drop(['Image','Eyeglasses','Mustache','Smiling','Wearing_Hat','Type'], axis=1)
    X = X.astype(np.float32)
    X = np.array(X)
    
#    Y = data[['Eyeglasses','Mustache','Smiling','Wearing_Hat']]
#    Y[Y <0] = 0.
#    Y = Y.astype(np.float)
#    Y = np.array(Y)
    Y_data = data[['Eyeglasses','Mustache','Smiling','Wearing_Hat']]
    Y = Y_data.apply(create_digits, axis = 1)
    
    inputs = [np.reshape(x, (4096, 1)) for x in X]
    inputs =[np.float32(x) for x in inputs]

    results = [vectorized_result(y) for y in Y]
#    results = [np.reshape(y, (4, 1)) for y in Y]

    dataset = zip(inputs, results)
    return dataset

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((5, 1))
#    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
