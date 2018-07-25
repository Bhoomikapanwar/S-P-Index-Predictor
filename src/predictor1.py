import matplotlib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

def get_data():
    #import data
    data = pd.read_csv("data_stocks.csv")

    #print data 5v rows and columns
    #print(data.head(5))
    #print(data.columns)

    #drop date variable
    data = data.drop(labels='DATE',axis = 1)
    print(data.head(1))

    #print dimensions of data set
    #n = data.shape[0]
    #p = data.shape[1]
    print(data.shape)

    """
    plt.plot(data['SP500'])
    plt.show()
    """

    #make data a numpy array
    data = data.values
    #print(data[0])
    return data

def make_train_test_sets(data):
    #divide data in training and test set
    n = data.shape[0]
    train_start = 0
    train_end = int(np.floor(n*0.8))
    test_start = train_end
    print(train_end,test_start)
    test_end = n
    data_train = data[np.arange(train_start,train_end),:]
    data_test = data[np.arange(test_start,test_end),:]
    #print(data_train[-1,0:20])
    #print(data_test[0,0:20])
    return data_train,data_test

def scale_data(data_train,data_test):
    #scale the data bew -1,1
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_test = scaler.transform(data_test)
    return data_train,data_test

def build_X_Y(data_train,data_test):
    #build X and Y
    X_train = data_train[:,1:]
    Y_train = data_train[:,0]
    X_test = data_test[:,1:]
    Y_test = data_test[:,0]
    return X_train,Y_train,X_test,Y_test

def predictor():
    data = get_data()
    data_train , data_test = make_train_test_sets(data)
    data_train , data_test = scale_data(data_train,data_test)
    X_train,Y_train,X_test,Y_test = build_X_Y(data_train,data_test)
    #no of stocks in training data
    n_stocks = X_train.shape[1]
    #print(n_stocks)
    #print(X_train.shape)

    #Neurons
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    #Session
    net = tf.InteractiveSession()

    #PlaceHolder
    X = tf.placeholder(dtype =tf.float32,shape=[None,n_stocks])
    Y = tf.placeholder(dtype = tf.float32,shape=[None])

    #Initializers
    sigma = 1
    weight_initializers = tf.variance_scaling_initializer(mode = "fan_avg",distribution="uniform",scale=sigma)
    bias_initializers = tf.zeros_initializer()

    #hidden weights
    W_hidden_1 = tf.Variable(weight_initializers([n_stocks,n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializers([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializers([n_neurons_1,n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializers([n_neurons_2]))
    W_hidden_3 = tf.Variable(weight_initializers([n_neurons_2,n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializers([n_neurons_3]))
    W_hidden_4 = tf.Variable(weight_initializers([n_neurons_3,n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializers([n_neurons_4]))

    #output weights
    W_out = tf.Variable(weight_initializers([n_neurons_4,n_target]))
    bias_out = tf.Variable(bias_initializers([n_target]))

    #hidden layer
    hidden1 = tf.nn.relu(tf.add(tf.matmul(X,W_hidden_1),bias_hidden_1))
    hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1,W_hidden_2),bias_hidden_2))
    hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2,W_hidden_3),bias_hidden_3))
    hidden4 = tf.nn.relu(tf.add(tf.matmul(hidden3,W_hidden_4),bias_hidden_4))

    #output layer
    out = tf.transpose(tf.add(tf.matmul(hidden4,W_out),bias_out))

    #cost function
    mse = tf.reduce_mean(tf.squared_difference(out,Y))

    #optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Init
    net.run(tf.global_variables_initializer())

    #setup pyplot
    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    line1, = ax1.plot(Y_test)
    line2, = ax1.plot(Y_test *0.5)

    #fit neural network
    batch_size = 256
    mse_train = []
    mse_test = []

    #run
    epochs = 10
    for e in range(epochs):

        #shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
        X_train = X_train[shuffle_indices]
        Y_train = Y_train[shuffle_indices]

        #minibatch training
        for i in range(0,len(Y_train)//batch_size):
            start = i*batch_size
            batch_x = X_train[start:start+batch_size]
            batch_y = Y_train[start:start+batch_size]
            #run optimizer
            net.run(opt,feed_dict={X:batch_x,Y:batch_y})

            #show progress
            if np.mod(i,50) == 0:
                #MSE train and test
                mse_train.append(net.run(mse,feed_dict={X:X_train,Y:Y_train}))
                mse_test.append(net.run(mse,feed_dict={X:X_test,Y:Y_test}))
                print('MSE Train: ', mse_train[-1])
                print('MSE Test: ', mse_test[-1])

                #prediction
                pred = net.run(out,feed_dict={X:X_test})
                line2.set_ydata(pred)
                plt.title('Epoch '+ str(e)+ ', Batch'+ str(i))
                file_name = 'img/epoch_' + str(e) + '_batch_' + str(i) + '.png'
                plt.savefig(file_name)
                plt.pause(0.01)

    # Print final MSE after Training
    mse_final = net.run(mse, feed_dict={X: X_test, Y: Y_test})
    print(mse_final)

if __name__ == '__main__':
    predictor()
