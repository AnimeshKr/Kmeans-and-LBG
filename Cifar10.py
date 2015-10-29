import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32'
#os.chdir('/Users/neerajkumar/Documents/DeepKernels')

import time
import sys
import theano 
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet import conv
from logistic_sgd import LogisticRegression #, load_data
from mlp_bak import HiddenLayer #SigmoidLayer, BinarizationLayer
#from DropoutHiddenLayer import 
from theano.tensor.signal import downsample
import numpy as np
from scipy import linalg
from sklearn.utils import array2d, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator

srng = RandomStreams()

import numpy

rng =numpy.random.RandomState(23455)
import pickle
with open('modelParams.pickle') as f:
   layer3params,layer2params, layer1params, layer0params = pickle.load(f)
print 'Previous weights loaded'


class convpool(object):
    def __init__(self,rng,input_1,input_2, filter_shape, image_shape, poolsize, dropout, parameters):
        """ Parameters of cnn class
        input: 4d tensor of input images
        output: labels
        filter_shape: 4d tensor- (number of filters, number of input feature
                                    maps, filter height, filter width)
        image_shape: 4d tensor- (batch size, number of input fetaure maps
                                image height, image width)
        pool size 2x2 pooling"""
        
        assert image_shape[1]==filter_shape[1]
        
        self.input_1 = input_1
        self.input_2 = input_2
        #self.output=output
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize=poolsize
       
        #rng = numpy.random.RandomState(23455)
        
        
        if parameters is None:
            fan_in=numpy.prod(filter_shape[1:])
            fan_out= numpy.prod(filter_shape[0]*numpy.prod(filter_shape[1:])/numpy.prod(poolsize))

        ## Random initialization of weights
            W_bound= numpy.sqrt(6./(fan_in + fan_out))
            self.W_1 = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high= W_bound, size= filter_shape),
                                             dtype=theano.config.floatX), borrow=True)
        
        #self.W_2 = theano.shared(numpy.asarray(rng.uniform(low=-W_bound, high= W_bound, size= filter_shape),
#                                             dtype=theano.config.floatX), borrow=True)
                                             
        ## Initializing bias which is 1D tensor--> one bias per output feature
            b_values = numpy.zeros(filter_shape[0], dtype=theano.config.floatX)
            self.b_1=theano.shared(value=b_values,borrow=True)
            
        else:
            self.W_1 = parameters[0]   
            self.b_1 = parameters[1]  
        #self.b_2=theano.shared(value=b_values,borrow=True)
        # convolve input feature maps with filters for input_1
        conv_out_1= conv.conv2d(
                  input=input_1,
                  filters=self.W_1,
                  filter_shape=filter_shape,
                  image_shape=image_shape)
                  
        # Appying maxpooling to each feature map
        pooled_output_1=downsample.max_pool_2d(
                      input=conv_out_1,
                      ds=poolsize,
                      ignore_border= True)
        #Applying Dropout              
        if dropout > 0.0:
            retain_prob = 1 - dropout
            pooled_output_1 *= srng.binomial(pooled_output_1.shape, p=retain_prob, dtype=theano.config.floatX)
            pooled_output_1 /= retain_prob
                      
        #Calculating final output by adding bias term
        self.output_1= T.tanh(pooled_output_1 + self.b_1.dimshuffle('x',0,'x','x'))
        
        # convolve input feature maps with filters for input_2
        conv_out_2= conv.conv2d(
                  input=input_2,
                  filters=self.W_1,
                  filter_shape=filter_shape,
                  image_shape=image_shape)
                  
        # Appying maxpooling to each feature map
        pooled_output_2=downsample.max_pool_2d(
                      input=conv_out_2,
                      ds=poolsize,
                      ignore_border= True)
        #Applying Dropout              
        if dropout > 0.0:
            retain_prob = 1 - dropout
            pooled_output_2 *= srng.binomial(pooled_output_2.shape, p=retain_prob, dtype=theano.config.floatX)
            pooled_output_2 /= retain_prob
                      
        #Calculating final output by adding bias term
        self.output_2= T.tanh(pooled_output_2 + self.b_1.dimshuffle('x',0,'x','x'))
        
        #store parameters of this layer
        self.params =[self.W_1, self.b_1]

######################### BUILD ACTUAL MODEL ################################
## Model specifications


learning_rate=0.12
n_epochs=1200
#Number of filters in each layer
nkerns=[96, 128]
batch_size=500
    
    

rng = numpy.random.RandomState(23455)

#loading the dataset
from random import randint
import load_CIFAR10
from sklearn.utils import shuffle
x_train, y_train, x_test, y_test = load_CIFAR10.cifar10()
trainX = np.concatenate((x_train,x_test), axis=0)
trainY = np.concatenate((y_train,y_test), axis=0)

#flength=len(trainY)/2

flength = 30000
ntrdiffX1 = np.zeros(shape=(flength,32*32*3))
ntrdiffX2 = np.zeros(shape=(flength,32*32*3))
ntrdiffY = np.zeros(shape=(flength,))

print 'Randomly picking',flength, 'pairs of dissimilar data (label = 0)'
#producing diff data i.e labels are different

i=0

#for i in xrange(0,flength/2):
while i < (flength):
    q=randint(0,flength-1)
    r=randint(0,flength-1)
        
    d=trainY[q]
    e=trainY[r]
        
    if d==e:
        i= i-1
        continue
    else:
        #print d,e
        ntrdiffY[i]=0
        a=trainX[q]
        b=trainX[r]
        ntrdiffX1[i]=a
        ntrdiffX2[i]=b
    i+=1

                
print "Randomly picking", flength, "pairs of similar data (label = 1)"
#print ntrdiffY
#producing equal data i.e labels are same
ntrsameX1 = np.zeros(shape=(flength,32*32*3))
ntrsameX2 = np.zeros(shape=(flength,32*32*3))
ntrsameY = np.zeros(shape=(flength,))
k=0

#
while k < (flength):
    q=randint(0,flength-1)
    r=randint(0,flength-1)
        
    d=trainY[q]
    e=trainY[r]
    if d == e:
        #print d,e
        ntrsameY[k]=1
        a=trainX[q]
        b=trainX[r]
        ntrsameX1[k]=a
        ntrsameX2[k]=b
    else:
        k= k-1
#        continue
    k=k+1
    
x_train_1 = np.concatenate((ntrsameX1[:25000],ntrdiffX1[:25000]),axis =0)
x_train_2 = np.concatenate((ntrsameX2[:25000],ntrdiffX2[:25000]),axis =0)
y_train = np.concatenate((ntrsameY[:25000],ntrdiffY[:25000]),axis =0)

x_test_1 = np.concatenate((ntrsameX1[25000:],ntrdiffX1[25000:]),axis =0)
x_test_2 = np.concatenate((ntrsameX2[25000:],ntrdiffX2[25000:]),axis =0)
y_test = np.concatenate((ntrsameY[25000:],ntrdiffY[25000:]),axis =0)

x_train_1= x_train_1.astype('float32',copy=True)
x_train_2= x_train_2.astype('float32',copy=True)

x_test_1= x_test_1.astype('float32',copy=True)
x_test_1= x_test_1.astype('float32',copy=True)

y_train= y_train.astype('int64',copy=True)
y_test= y_test.astype('int64',copy=True)

x_train_1, x_train_2, y_train = shuffle(x_train_1, x_train_2, y_train)

x_test_1, x_test_2, y_test = shuffle(x_test_1, x_test_2, y_test)

print 'Data loading done!'


print 'ZCA Whitening...'

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = array2d(X)
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=1).transpose()
        X = X.transpose()
        #self.mean_ =  np.mean(X, axis=0)
        #print 'mean subraction..'
        X -= self.mean_
        sigma = np.dot(X,X.T) / X.shape[1]
        #print 'applying SVD...'
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        print 'fit done'
        return self

    def transform(self, X):
        X = array2d(X)
        #print 'mean subtraction'
        X_transformed = X.transpose() - self.mean_
        X_transformed = X_transformed.transpose()
        print 'applying transform..'
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed

        
ZCA_processor_train = ZCA(regularization= 10**-5,copy='False')
## Whitening the training data
ZCA_processor_train.fit(X=x_train_1, y=None)
x_train_1_ZCA=ZCA_processor_train.transform(X=x_train_1)

ZCA_processor_train.fit(X=x_train_2, y=None)
x_train_2_ZCA=ZCA_processor_train.transform(X=x_train_2)
print 'Training data whitened!'
## Whitening the testing data
ZCA_processor_test = ZCA(regularization= 10**-5,copy='False')
ZCA_processor_test.fit(X=x_test_1, y=None)
x_test_1_ZCA=ZCA_processor_test.transform(X=x_test_1)

ZCA_processor_test.fit(X=x_test_2, y=None)
x_test_2_ZCA=ZCA_processor_test.transform(X=x_test_2)
print 'Test data whitened!'
#
#with open('CIFAR10BinaryLabelDataZCA.pickle','w') as f:
#   pickle.dump([x_train_1_ZCA, x_train_2_ZCA,y_train, x_test_1_ZCA,x_test_2_ZCA, y_test], f)
#print 'Loading previously saved data...'
#
#with open('CIFAR10BinaryLabelDataZCA.pickle') as f:
#   x_train_1_ZCA, x_train_2_ZCA,y_train, x_test_1_ZCA,x_test_2_ZCA, y_test = pickle.load(f)
#
train_set = (x_train_1_ZCA, x_train_2_ZCA, y_train)
test_set = (x_test_1_ZCA,x_test_2_ZCA, y_test)

print 'Data Loaded!'


# Formatting data in theano shared variables
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x_1,data_x_2, data_y = data_xy
        shared_x_1 = theano.shared(numpy.asarray(data_x_1,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
                                 
        shared_x_2 = theano.shared(numpy.asarray(data_x_2,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)                         
                                 
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x_1,shared_x_2, T.cast(shared_y, 'int32')
        
test_set_x_1,test_set_x_2, test_set_y = shared_dataset(test_set)
#valid_set_x, valid_set_y = shared_dataset(valid_set)
train_set_x_1,train_set_x_2, train_set_y = shared_dataset(train_set)


# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x_1.get_value(borrow=True).shape[0]
#n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x_1.get_value(borrow=True).shape[0]
n_train_batches /= batch_size
#n_valid_batches /= batch_size
n_test_batches /= batch_size

    # allocate symbolic variables for the data
index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
x_1 = T.matrix('x_1')   # the data is presented as rasterized images
x_2 = T.matrix('x_2') 
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
print '... building the model'
#
#    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
#    # to a 4D tensor, compatible with our LeNetConvPoolLayer
#    # (28, 28) is the size of MNIST images.
layer0_input_1 = x_1.reshape((batch_size, 3, 32, 32))
layer0_input_2 = x_2.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1 , 32-5+1) = (28, 28)
    # maxpooling reduces this further to (28/2, 28/2) = (14, 14)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 14, 14)
layer0 = convpool(
        rng,
        input_1=layer0_input_1,
        input_2=layer0_input_2,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2), dropout=0.1,
        parameters = layer0params
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (14-5+1, 14-5+1) = (10, 10)
    # maxpooling reduces this further to (10/2, 10/2) = (5, 5)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 5, 5)
layer1 = convpool(
        rng,
        input_1=layer0.output_1,
        input_2=layer0.output_2,
        image_shape=(batch_size, nkerns[0], 14, 14),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2), dropout = 0.25,
        parameters = layer1params
    )

layer2_input_1 = layer1.output_1.flatten(2)
layer2_input_2 = layer1.output_2.flatten(2)

layer2_input = T.concatenate([layer2_input_1, layer2_input_2], axis=1)
    # construct a fully-connected sigmoidal layer


layer2 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 2* 5*5,
        n_out=4000,
        activation=T.tanh,
        dropout = 0.5,
        parameters = None
        
    )
    
layerH2 = HiddenLayer(
        rng,
        input=layer2.output,
        n_in=4000,
        n_out=2000,
        activation=T.tanh,
        dropout = 0.5,
        parameters = None
    )

    # classify the values of the fully-connected sigmoidal layer
layer3 = LogisticRegression(input=layerH2.output, n_in=2000, n_out=2,parameters=None)

    # the cost we minimize during training is the NLL of the model
cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
test_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x_1: test_set_x_1[index * batch_size: (index + 1) * batch_size],
            x_2: test_set_x_2[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )



#params = layer3.params +  layerH2.params +layer2.params + layer1.params + layer0.params
params = layer3.params +  layerH2.params +layer2.params 

    # create a list of gradients for all model parameters
grads = T.grad(cost, params)


updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x_1: train_set_x_1[index * batch_size: (index + 1) * batch_size],
            x_2: train_set_x_2[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #epoch
    ###############
print '... training'

test_score = 0.
start_time = time.clock()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    epoch = epoch + 1
    
    for minibatch_index in xrange(n_train_batches):
        iter = (epoch-1)*n_train_batches + minibatch_index
        if iter % 100 == 0:
            print 'Training @ iter= ', iter 
        #if minibatch_index % 10==0:
        #print 'Processing minibatch # %d of epoch # %d.' % (minibatch_index, epoch)
        cost_ij= train_model(minibatch_index)
        
        ## Testing the model
        if (iter + 1) % 100 == 0:

            test_losses = [test_model(i)for i in xrange(n_test_batches)]
            test_score = numpy.mean(test_losses)
            print(('epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))
    #learning_rate= learning_rate-0.001
#        if patience <= iter:
#            done_looping = True
#            break
     
end_time = time.clock()
print('Optimization complete.')

print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.)) 

