# -*- coding: utf-8 -*-
"""
Created on Sat Sep 03 12:10:03 2016

@author: Sergey
"""

import numpy as np
import sklearn
import sklearn.mixture
from sklearn.decomposition import PCA
import sys
import os
import matplotlib.pyplot as plt

#******************DOWNLOADING_THE_DATA - from lasagne tutorial***********
def load_dataset():
    # N - number of instances to download
    # mode = "random" or "ranged"
    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 28*28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train,  X_test, y_test
    
    
    
def data_partition(N, reg):
    #N - numer of instances
    if (reg == 'random'):
        indices = np.random.permutation(X_train.shape[0])
        X_train_new = X_train[indices, :]
        y_train_new = y_train[indices]
        X_train_new = X_train_new[:N, :]
        y_train_new = y_train_new[:N]
        
        k = N/10
        
        T_X = [X_train_new[k * i: k * (i + 1), : ] for i in range(10)]
        T_y = [y_train_new[k * i: k * (i + 1) ] for i in range(10)]
        
        return T_X, T_y 
        
    elif (reg == 'label'): 
        groups_by_label = [[i  for i in range(len(y_train)) if (y_train[i] == j)] for j in range(10)]
        rand_permutation_inside_groups = [np.random.permutation(groups_by_label[i]) for i in range(10)]
        choice_of_instances_X = [X_train[rand_permutation_inside_groups[i][:N], :] for i in range(10)]
        choice_of_instances_y = [y_train[rand_permutation_inside_groups[i][:N]] for i in range(10)]
        
        return choice_of_instances_X, choice_of_instances_y

    else:
        raise ValueError('Inserted type of partition does not exist.')
        
    
def data_choice(N):
    #N - numer of instances
    indices = np.random.permutation(X_train.shape[0])
    X_train_new = X_train[indices, :]
    y_train_new = y_train[indices]
    X_train_new = X_train_new[:N, :]
    y_train_new = y_train_new[:N]
    
    return X_train_new, y_train_new
#*****************************************************************************








#********************Local_MLE**********************************

def local_mle(data_test_type, data_partition_type):
    #data_test_type = 'train'/'test'
    #data_partition_type = 'random'/'lable' - type of partition
    print('******LOCAL_AVERAGE for ' + data_partition_type + ' and ' + data_test_type + ' ******')
    Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
    trial = 10
    L = [0 for i in range(len(Numbers))]
    model = sklearn.mixture.GMM(n_components = 10)
    for j in range(len(Numbers)):
        print 'Count for N = {}...'.format(Numbers[j])
        total_average = 0
        for i in range(trial):
            
            X, Y = data_partition(Numbers[j], data_partition_type)
            s = 0
            x_concat = np.concatenate(X)
            y_concat = np.concatenate(Y)
            for k in range(10):
                model.fit(X[k],Y[k])
                if (data_test_type == 'test'):
                    s = s + sum(model.score(X_test, y_test))/100000#X_test, y_test
                elif (data_test_type == 'train'):
                    s = s + (sum(model.score(x_concat, y_concat))/10)/Numbers[j]
                else:
                    raise ValueError('Inserted type of test data does not exist.')
            total_average = total_average + s/trial
            
        L[j] =  total_average
    
    return L
  
#**********************Global MLE**********************************  
    
def global_mle(data_test_type):
    print('********Global_MLE for ' + data_test_type + ' ******')
    Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
    trial = 10
    L_on_full = [0 for i in range(len(Numbers))]
    model = sklearn.mixture.GMM(n_components = 10)
    for j in range(len(Numbers)):
        print 'Count for N = {}...'.format(Numbers[j])
        total_average = 0
        
        for i in range(trial):
            
            X,Y = data_choice(Numbers[j])
            model.fit(X,Y)
            
            if (data_test_type == 'test'):
                total_average = total_average + (sum(model.score(X_test, y_test))/10000)/trial
            elif (data_test_type == 'train'):
                total_average = total_average + (sum(model.score(X))/len(Y))/trial
            else:
                raise ValueError('Inserted type of test data does not exist.')
    
            
        L_on_full[j] = total_average
    
    return L_on_full
                       

#**********************KL-ESTIMATOR**********************************  


def kl_estimator(data_test_type, data_partition_type):
    print('********KL_ESTIMATOR for ' + data_partition_type + ' and ' + data_test_type + ' ******')
    Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
    trial = 10
    L_KL = [0 for i in range(len(Numbers))]
    model = sklearn.mixture.GMM(n_components = 10)
    
    for j in range(len(Numbers)):
        print 'Count for N = {}...'.format(Numbers[j])
        total_average = 0
            
        for i in range(trial):
            
            X, Y = data_partition(Numbers[j], data_partition_type)
            
            model.fit(X[0],Y[0])
            L_curr = model.sample(500)           
            
            for k in range(1, 10):
                model.fit(X[k],Y[k])
                L_curr = np.concatenate( (L_curr, model.sample(500)))            
            
            model.fit(L_curr)
            
            if (data_test_type == 'test'):
                total_average = total_average + (sum(model.score(X_test, y_test))/10000)/trial 
            elif (data_test_type == 'train'):
                total_average = total_average + (sum(model.score(X[k], Y[k]))/len(Y[k]))/trial
            else:
                raise ValueError('Inserted type of test data does not exist.')
      
        L_KL[j] = total_average      
   
    return L_KL



#********************NAIVE_LINEAR_AVERAGE*******************************************

def naive_linear_average(data_test_type, data_partition_type):
    print('******NAIVE_LINEAR for ' + data_partition_type + ' and ' + data_test_type + ' ******')
    Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
    trial = 10
    L_NAIVE_LINEAR = [0 for i in range(len(Numbers))]
    model = sklearn.mixture.GMM(n_components = 10)
    for j in range(len(Numbers)):
        print 'Count for N = {}...'.format(Numbers[j])
        total_average = 0
           
        for i in range(trial):
            X, Y = data_partition(Numbers[j], data_partition_type)
            model.fit(X[0],Y[0])
            
            
            weights = model.weights_ / 10
            means = model.means_ / 10
            covars = model.covars_ / 10
            
            for k in range(1,10):
                
                model.fit(X[k],Y[k])
                weights = weights + model.weights_/10
                means = means + model.means_/10
                covars = covars + model.covars_/10
            
            model.means_ = means
            model.weights_ = weights
            model.covars_ = covars
            
            if (data_test_type == 'test'):
                total_average = total_average + (sum(model.score(X_test, y_test))/10000)/trial 
            elif (data_test_type == 'train'):
                concat_train_X = np.concatenate(X)
                concat_train_Y = np.concatenate(Y)
                total_average = total_average + (sum(model.score(concat_train_X, concat_train_Y))/len(concat_train_Y))/trial
            else:
                raise ValueError('Inserted type of test data does not exist.')
            
        
        L_NAIVE_LINEAR[j] = total_average 
    
    return L_NAIVE_LINEAR


#*******************************ALPHA-DIVERGENCE*************************

def f_alpha(alpha,x):
    if (alpha == 1.0):
        return math.log(x)
    else:
        return (2/(1 - alpha))*(x**((1 - alpha)/2))
        
def f_alpha_inv(alpha, x):
    if (alpha == 1.0):
        return math.exp(x)
    else:
        return (x * (1 - alpha)/2)**(2/(1 - alpha))
    
def alpha_div(alpha, data_partition_type):
    print('******ALPHA DiV for ' + data_partition_type + ' and ' +  'alpha = '+ str(alpha) )
    Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
    Models = [sklearn.mixture.GMM(n_components = 10) for i in range(10)]
    L_ALPHA = [0 for i in range(len(Numbers))]
    
    for j in range(len(Numbers)):
        print(Numbers[j])
        X, Y = data_partition(Numbers[j], data_partition_type)
        Models_fit = [Models[i].fit(X[i], Y[i]) for i in range(10)]
        Scores = [Models_fit[i].score(X_test, y_test) for i in range(10)]
        summ = 0.0
        
        for i in range(len(X_test)):
            summ = summ + math.log(f_alpha_inv(alpha, 1.0/10.0 * sum([f_alpha(alpha, math.exp(Scores[k][i])) for k in range(10)])))/10000.0
        
        L_ALPHA[j] = summ
        
    return L_ALPHA
        
        
        
#*******************PCA_ANALISYS******************************************
"""
print('************FIXING DATA....***********')

X_train, y_train, X_test, y_test = load_dataset()
#print(X_train.shape) 
pca = PCA(n_components=100)
Z = pca.fit_transform(np.concatenate((X_train, X_test)))
X_train = Z[:60000, :]
X_test = Z[60000:, :]
#*************************************************************************
"""
print('******START OF LEARNING*****')

#L_MLE_rand_train = local_mle('train', 'random')
#L_MLE_rand_test = local_mle('test', 'random')
#L_MLE_label_train = local_mle('train', 'label')
#L_MLE_label_test = local_mle('test', 'label')

#G_MLE_train = global_mle('train')
#G_MLE_test = global_mle('test')

#L_AVG_rand_train = naive_linear_average('train', 'random')
#L_AVG_rand_test = naive_linear_average('test', 'random')
#L_AVG_label_train = naive_linear_average('train', 'label')
#L_AVG_label_test = naive_linear_average('test', 'label')

#KL_AVG_rand_train = kl_estimator('train', 'random')
#KL_AVG_rand_test = kl_estimator('test', 'random')
#KL_AVG_label_train = kl_estimator('train', 'label')
#KL_AVG_label_test = kl_estimator('test', 'label')

Numbers = [500]+[1000*i for i in range(1,10)]+[10000*i for i in range(1,6)]
"""
print('************************RAND/TRAIN********************************')


line1 = plt.plot(Numbers, G_MLE_train, 's', linestyle = '--', color = 'b', label = 'G_MLE'); 
line2 = plt.plot(Numbers, L_MLE_rand_train, '*', linestyle = '-', color = 'b', label = 'L_MLE'); 
line3 = plt.plot(Numbers, KL_AVG_rand_train, 's',linestyle = '-', color = 'g', label = 'KL_AVG' );
line4 = plt.plot(Numbers, L_AVG_rand_train , 's',linestyle = '-', color = 'r', label = 'L_AVG' );
plt.xlabel('Train LL(random partition) ')
plt.xscale('log')
plt.legend(loc='down left')
plt.savefig('rand_train.png')

"""
"""
print('************************RAND/TEST********************************')

import matplotlib.pyplot as plt
line1 = plt.plot(Numbers, G_MLE_test, 's', linestyle = '--', color = 'b', label = 'G_MLE'); 
line2 = plt.plot(Numbers, L_MLE_rand_test, '*', linestyle = '-', color = 'b', label = 'L_MLE'); 
line3 = plt.plot(Numbers, KL_AVG_rand_test, 's',linestyle = '-', color = 'g', label = 'KL_AVG' );
line4 = plt.plot(Numbers, L_AVG_rand_test , 's',linestyle = '-', color = 'r', label = 'L_AVG' );
plt.xlabel('Test LL(random partition) ')
plt.xscale('log')
plt.legend(loc='down left')
plt.savefig('rand_test.png')

"""
"""
print('************************LABEL/TRAIN********************************')

import matplotlib.pyplot as plt
line1 = plt.plot(Numbers, G_MLE_train, 's', linestyle = '--', color = 'b', label = 'G_MLE'); 
line2 = plt.plot(Numbers[12:15], L_MLE_label_train[12:15], '*', linestyle = '-', label = 'L_MLE'); 
line3 = plt.plot(Numbers, KL_AVG_label_train, 's',linestyle = '-', color = 'g', label = 'KL_AVG' );
line4 = plt.plot(Numbers, L_AVG_label_train , 's',linestyle = '-', color = 'r', label = 'L_AVG' );
plt.xlabel('Train LL(label-wise partition) ')
plt.xscale('log')
plt.legend(loc='down left')
plt.savefig('label_train.png')

"""
"""
print('************************LABEL/TEST********************************') 

import matplotlib.pyplot as plt
line1 = plt.plot(Numbers, G_MLE_test, 's', linestyle = '--', color = 'b', label = 'G_MLE'); 
line2 = plt.plot(Numbers, L_MLE_label_test, '*', linestyle = '-', label = 'L_MLE'); 
line3 = plt.plot(Numbers, KL_AVG_label_test, 's',linestyle = '-', color = 'g', label = 'KL_AVG' );
line4 = plt.plot(Numbers, L_AVG_label_test , 's',linestyle = '-', color = 'r', label = 'L_AVG' );
plt.xlabel('Test LL(label-wise partition)')
plt.xscale('log')
plt.legend(loc='down left')
plt.savefig('label_test.png')
"""
    


def pp():
    
    from matplotlib import pyplot
    
    """
    line5 = plt.plot(Numbers, Mn_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, Mn_10e06, 'o', linestyle = '-', color = 'r', label = '-10e-06');
    line3 = plt.plot(Numbers, Mn_01, 'o',linestyle = '-', color = 'g', label = '-0.1' );
    line4 = plt.plot(Numbers, Mn_02, '^',linestyle = '-', color = 'r', label = '-0.2' );
    line5 = plt.plot(Numbers, Mn_05, 'o',linestyle = '-', color = 'y', label = '-0.5' );
    """
    """
    line5 = plt.plot(Numbers,Mn_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, Mn_07, 'o', linestyle = '-', color = 'r', label = '-0.7');
    line3 = plt.plot(Numbers, Mn_1, 'o',linestyle = '-', color = 'g', label = '-1.0' );
    line4 = plt.plot(Numbers, Mn_3, '^',linestyle = '-', color = 'r', label = '-3.0' );
    line5 = plt.plot(Numbers, Mn_5, 'o',linestyle = '-', color = 'y', label = '-5.0' );    
    """
    
    line5 = plt.plot(Numbers,Mn_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, Mp_01, 'o', linestyle = '-', color = 'r', label = '0.1');
    line3 = plt.plot(Numbers, Mp_05, 'o',linestyle = '-', color = 'g', label = '0.5' );
    line5 = plt.plot(Numbers, Mp_07, '*',linestyle = '-', color = 'y', label = '0.7' );
    line4 = plt.plot(Numbers, Mp_1, '^',linestyle = '-', color = 'r', label = '1.0' );
    line5 = plt.plot(Numbers, Mp_2, 'o',linestyle = '-', color = 'y', label = '2.0' );
    line5 = plt.plot(Numbers, Mp_5, '*',linestyle = '-', color = 'g', label = '5.0' );
    
    """
    line5 = plt.plot(Numbers, L_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, L_10e06, 'o', linestyle = '-', color = 'r', label = '-10e-06');
    line3 = plt.plot(Numbers, Lm01, 'o',linestyle = '-', color = 'g', label = '-0.1' );
    line4 = plt.plot(Numbers, Lm02, '^',linestyle = '-', color = 'r', label = '-0.2' );
    line5 = plt.plot(Numbers, Lm05, 'o',linestyle = '-', color = 'y', label = '-0.5' );
    """
    """
    line5 = plt.plot(Numbers,L_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, Lm07, 'o', linestyle = '-', color = 'r', label = '-0.7');
    line3 = plt.plot(Numbers, Lm1_0, 'o',linestyle = '-', color = 'g', label = '-1.0' );
    line4 = plt.plot(Numbers, Lm2, '^',linestyle = '-', color = 'r', label = '-2.0' );
    line5 = plt.plot(Numbers, Lm3, 'o',linestyle = '-', color = 'y', label = '-3.0' );
    line5 = plt.plot(Numbers, Lm5, '*',linestyle = '-', color = 'y', label = '-5.0' ); 
    """
    """
    line5 = plt.plot(Numbers,L_0, 's',linestyle = '--', color = 'b', label = '0.0' );
    line5 = plt.plot(Numbers, G_MLE_test, '*',linestyle = '--', color = 'b', label = 'G_mle' );
    line2 = plt.plot(Numbers, Lp01, 'o', linestyle = '-', color = 'r', label = '0.1');
    line3 = plt.plot(Numbers, Lp0_5, 'o',linestyle = '-', color = 'g', label = '0.5' );
    line5 = plt.plot(Numbers, Lp1, '*',linestyle = '-', color = 'y', label = '1.0' );
    line4 = plt.plot(Numbers, Lp2, '^',linestyle = '-', color = 'r', label = '2.0' );
    line5 = plt.plot(Numbers, Lp5, 'o',linestyle = '-', color = 'y', label = '5.0' );
    """
    
    pyplot.legend(loc='down left')
    pyplot.xscale('log')
    pyplot.xlabel('Alpha = 0.0:5.0 - rand partition ')
    pyplot.savefig('trytry.png')
    
"""
Mn_10e06 = alpha_div(10e-06, 'label')
Mn_01 = alpha_div(-0.1, 'label')
Mn_02 = alpha_div(-0.2, 'label')
Mn_05 = alpha_div(-0.5, 'label')
Mn_07 = alpha_div(-0.7, 'label')
Mn_09 = alpha_div(-0.9, 'label')
Mn_1 = alpha_div(-1.0, 'label')
Mn_2 = alpha_div(-2.0, 'label')
Mn_3 = alpha_div(-3.0, 'label')
Mn_5 = alpha_div(-5.0, 'label')
Mp_01 = alpha_div(0.1, 'label')
Mp_05 = alpha_div(0.5, 'label')
Mp_1 = alpha_div(1.0, 'label')
Mp_2 = alpha_div(2.0, 'label')
Mp_5 = alpha_div(5.0, 'label')
"""