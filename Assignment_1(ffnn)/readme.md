<<<<<<< HEAD
>> Question 1 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy 
used subplots in matplotlib library and plotted all 10 images of each distinct class 
0 to 9.

corresponding label names are :  
lable_names=[":t-shirt/top", ':trouser/pants', ':pullover shirt', ':dress', ':coat', ':sandal', ':shirt', ':sneaker', ':bag', ':ankle boot']

link of direct colab code :
﻿https:﻿//colab.research.google.com/drive/1wfPqs2UEewiyLpC-s3zhXsJ7cQcNf4u2?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom. (for wandb to work correctly for you, you may need to log in to your own project with login key)

**already attached code output of wandb in report.












>> Question 2 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy
loadData() function is to load the above from fashion_mnist and after a random shuffling make 90% train data, 10% validation data splits.
In this question the motto is initialise the weights and biases with random/normal/xavier and after fixing an activation function(sigmoid, relu , tanh)
just feedforward it using "justForwarding" function.

link of direct colab code :
﻿https:﻿//colab.research.google.com/drive/1RGmG5Jn0uD-xiHmm2vePVBfRvpEsedK8?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom.

my main class name is "MLSN"
it takes 10 arguments -->
flattened input size of data (784 in this case; i.e. 28*28), 
hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
number of available classes for a data point; (here 10)
actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initialisation function(string), use "random" or "xavier"
loss function(string), use "crossentropy" or "squarederror"
regularizer coefficient(float), eg: 0.00005, 0.0002 etc
last 4 arguments are valx, valy, testx, testy

example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)
then you have to call a function named "justForwarding" .
it takes one argument -->
input training data points as a list.
example>>> as in code I declared as => def justForwarding(self, xs):
           calling by the prev. made object should be like  obj.justForwarding(trainx[0:20]) 
           here I have used just the first 20 trainx data points.
Output of code is self explainatory.













>> Question 3 & 4 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy

link of direct colab code :
﻿https:﻿//colab.research.google.com/drive/1wZZRIwWy_so6GmDRYIzq_XssXSjAhqAD?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom.(for wandb sweep to work correctly for you, you may need to log in to your own project with login key)

my main class name is "MLSN"
it takes 10 arguments -->
szInp=flattened input size of data (784 in this case; i.e. 28*28), 
hiddenL=hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
numOp=number of available classes for a data point; (here 10)
actvfunc=actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initfunc=initialisation function(string), use "random" or "xavier"
lossfunc=loss function(string), use "crossentropy" or "squarederror"
lmda=regularizer coefficient(float), eg: 0.00005, 0.0002 etc
vx, vy, xtest, ytest = these last 4 arguments are valx, valy, testx, testy
example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)

then using this object you have to call "trainParent" function.
example>>> as in code I declared as =>   def trainParent(self, xs, ys, strr, run, batchsize, rate):
            calling function with prev. made object should be like => obj.trainParent(trainx,trainy, "NAGD", 10, 32, 3)

            above function takes 6 arguments while calling; 
            xs=training data(trainx), 
            ys=training lables(trainy), 
            strr=optimisation func. (string: use any one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM'])
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : passing rate=3 in above example but I use dynamic learning rate ; 
                                    so effective learning rate is (rate/batchsize); here in above example it is (3/32).

question 3,4 runs with wandb sweep; if you want to change sweep dictionary then change the "sweep_config" variable(python dict variable) written in one of the cells above in ipynb.
If you want to incorporate a new optimisation function such as adaGrad: do the following
========================================================================================
in class MLSN there is a distict train function for each one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM']
eg:   def trainNADAM(self, xs, ys, run, batchsize, rate),
      def trainADAM(self, xs, ys, run, batchsize, rate),
      def trainMBGD(self, xs, ys, run, batchsize, rate) etc.
you just have to define adaGrad like   def trainADAGRAD(self, xs, ys, run, batchsize, rate):
arguments of the functions are 
            xs=training data, 
            ys=training lables, 
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : I use dynamic learning rate ; if rate=r
                                         effective learning rate is (r/batchsize).

example structure:

  def trainADAGRAD(self, xs, ys, run, batchsize, rate):
    #*initialise anyvariable you want for ADAGRAD*
    w_s=[np.zeros(xx.shape) for xx in self.delLdelW]
    b_s=[np.zeros(xx.shape) for xx in self.delLdelB]
    for _ in range(run): # epochs loop
      
      self.reinitdels() # delLdelW's and delLdelB's are initialized to all 0
      
      for i in range(len(xs)): 3 loop for each train data point
        curinp=xs[i] # i-th data point
        self.feedFrwd(curinp,self.weights, self.biases) # doing forwarding
        self.backProp(i, ys) # doing backprop
        if i%(batchsize-1)==0 and i!=0: # checking for batch size
          #*code for ADAGRAD optimiser*
          # the only variables of my code you need is self.weights[*use proper indexing*], self.biases[*use proper indexing*],  self.delLdelW[*use proper indexing*],  self.delLdelB[*use proper indexing*]
          # all 4 of the variables above are a 3D list, with number of elements = number of hidden layers+1 ; each element is a 2D matrix of size (no.of inputs X no.of outputs)
          self.reinitdels() # delLdelW's and delLdelB's are initialized to all 0

      self.AccuracyLossLogging(_, xs, ys) # logging train, test accuracy and loss
    self.TestAccuracy(self.xtest, self.ytest) # for test data confusion matrix

========================================================================================
after this add this line => elif strr=="NADAM" : self.trainADAGRAD(xs, ys, run, batchsize, rate) to "trainParent" function of MLSN class.
That's all.
========================================================================================



















>> Question 5 information >>
For this question wandb automatically plots validation accuracy vs date created.
in report I attached all 350+ plots summary



















>> Question 6 information >>
For this question wandb automatically plots parellel coordinate plots with importance and correlation summary..
in report I attached all results.























>> Question 7 information >>
For this question wandb automatically plots confusion matrix.
in report I attached confusion matrix of one of the best results among my 350+ runs.
Also created a heatmap(another way to view confusion matrix) of the same run as above.



















>> Question 8 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy

link of direct colab code :
﻿https:﻿//colab.research.google.com/drive/1wZZRIwWy_so6GmDRYIzq_XssXSjAhqAD?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom.(for wandb sweep to work correctly for you, you may need to log in to your own project with login key)

my main class name is "MLSN"
it takes 10 arguments -->
szInp=flattened input size of data (784 in this case; i.e. 28*28), 
hiddenL=hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
numOp=number of available classes for a data point; (here 10)
actvfunc=actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initfunc=initialisation function(string), use "random" or "xavier"
lossfunc=loss function(string), use "crossentropy" or "squarederror"
lmda=regularizer coefficient(float), eg: 0.00005, 0.0002 etc
vx, vy, xtest, ytest = these last 4 arguments are valx, valy, testx, testy
example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)

then using this object you have to call "trainParent" function.
example>>> as in code I declared as =>   def trainParent(self, xs, ys, strr, run, batchsize, rate):
            calling function with prev. made object should be like => obj.trainParent(trainx,trainy, "NAGD", 10, 32, 3)

            above function takes 6 arguments while calling; 
            xs=training data(trainx), 
            ys=training lables(trainy), 
            strr=optimisation func. (string: use any one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM'])
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : passing rate=3 in above example but I use dynamic learning rate ; 
                                    so effective learning rate is (rate/batchsize); here in above example it is (3/32).


I have used,(you may see after clicking the above link)
             obj=MLSN(dim_inp, [32], 10, "reLU", "xavier", "crossentropy", 0.0001, valx, valy, testx, testy)
             a1,b1,c1=obj.trainParent(trainx,trainy, "GD", 5, 32, 3.1)
             obj1=MLSN(dim_inp, [32], 10, "reLU", "xavier", "squarederror", 0.0001, valx, valy, testx, testy)
             a2,b2,c2=obj1.trainParent(trainx,trainy, "GD", 5, 32, 3.1)
i.e. activation reLU, batch_size 32, 10 epochs, activation function xavier, hidden layer architecture [32], 
     optimization function Gradient Descant, effective learning rate (3.1/batch_size), regularizing coeff. 0.0001,
     loss crossentropy. 

## a1, a2 training accuracy figure : 2 (crossentropy vs squarederror)
## b1, b2 validation accuracy figure : 1 (crossentropy vs squarederror)
## c1, c2 validation accuracy figure : 3 (crossentropy vs squarederror)
## I plotted each pair with plottttt function(user defined)
like this>>>
Plottttt(a1, a2, "Training accuracy(crossentropy vs squarederror)")
Plottttt(b1, b2, "validation accuracy(crossentropy vs squarederror)")
Plottttt(c1, c2, "Test accuracy(crossentropy vs squarederror)")

We can conclude that crossentropy is a better loss function here than squarederror.
You may change the parameters given in function in colab code (link above) according to readme.md instructions for question 8 
and see how the plots come.





















>> Question 9 information >>
# github codes link
https://github.com/sayan-chandra/CS6910_deepLearning
# go to Assignment_1(ffnn) folder.
=======
# by Sayan Chandra
# Roll : CS20M057
# IITM
# Assignment 1 deep learning by prof. Mitesh M. Khapra

Link to wandb report>>
https://wandb.ai/blackcloud/cs6910_dl_assignment_1/reports/CS6910-Assignment-1-solutions---Vmlldzo1MTk0MTU

#>> Question 1 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy 
used subplots in matplotlib library and plotted all 10 images of each distinct class 
0 to 9.

corresponding label names are :  
lable_names=[":t-shirt/top", ':trouser/pants', ':pullover shirt', ':dress', ':coat', ':sandal', ':shirt', ':sneaker', ':bag', ':ankle boot']

link of direct colab code :
https://colab.research.google.com/drive/1wfPqs2UEewiyLpC-s3zhXsJ7cQcNf4u2?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom. (for wandb to work correctly for you, you may need to log in to your own project with login key)

**already attached code output of wandb in report.












>> Question 2 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy
loadData() function is to load the above from fashion_mnist and after a random shuffling make 90% train data, 10% validation data splits.
In this question the motto is initialise the weights and biases with random/normal/xavier and after fixing an activation function(sigmoid, relu , tanh)
just feedforward it using "justForwarding" function.

link of direct colab code :
https://colab.research.google.com/drive/1RGmG5Jn0uD-xiHmm2vePVBfRvpEsedK8?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom.

my main class name is "MLSN"
it takes 10 arguments -->
flattened input size of data (784 in this case; i.e. 28*28), 
hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
number of available classes for a data point; (here 10)
actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initialisation function(string), use "random" or "xavier"
loss function(string), use "crossentropy" or "squarederror"
regularizer coefficient(float), eg: 0.00005, 0.0002 etc
last 4 arguments are valx, valy, testx, testy

example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)
then you have to call a function named "justForwarding" .
it takes one argument -->
input training data points as a list.
example>>> as in code I declared as => def justForwarding(self, xs):
           calling by the prev. made object should be like  obj.justForwarding(trainx[0:20]) 
           here I have used just the first 20 trainx data points.
Output of code is self explainatory.













>> Question 3 & 4 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy

link of direct colab code :
https://colab.research.google.com/drive/1wZZRIwWy_so6GmDRYIzq_XssXSjAhqAD?usp=sharing

other instructions to run(in case):
just run each code cell one after another from top to bottom.(for wandb sweep to work correctly for you, you may need to log in to your own project with login key)

my main class name is "MLSN"
it takes 10 arguments -->
szInp=flattened input size of data (784 in this case; i.e. 28*28), 
hiddenL=hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
numOp=number of available classes for a data point; (here 10)
actvfunc=actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initfunc=initialisation function(string), use "random" or "xavier"
lossfunc=loss function(string), use "crossentropy" or "squarederror"
lmda=regularizer coefficient(float), eg: 0.00005, 0.0002 etc
vx, vy, xtest, ytest = these last 4 arguments are valx, valy, testx, testy
example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)

then using this object you have to call "trainParent" function.
example>>> as in code I declared as =>   def trainParent(self, xs, ys, strr, run, batchsize, rate):
            calling function with prev. made object should be like => obj.trainParent(trainx,trainy, "NAGD", 10, 32, 3)

            above function takes 6 arguments while calling; 
            xs=training data(trainx), 
            ys=training lables(trainy), 
            strr=optimisation func. (string: use any one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM'])
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : passing rate=3 in above example but I use dynamic learning rate ; 
                                    so effective learning rate is (rate/batchsize); here in above example it is (3/32).

question 3,4 runs with wandb sweep; if you want to change sweep dictionary then change the "sweep_config" variable(python dict variable) written in one of the cells above in ipynb.
If you want to incorporate a new optimisation function such as adaGrad: do the following
========================================================================================
in class MLSN there is a distict train function for each one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM']
eg:   def trainNADAM(self, xs, ys, run, batchsize, rate),
      def trainADAM(self, xs, ys, run, batchsize, rate),
      def trainMBGD(self, xs, ys, run, batchsize, rate) etc.
you just have to define adaGrad like   def trainADAGRAD(self, xs, ys, run, batchsize, rate):
arguments of the functions are 
            xs=training data, 
            ys=training lables, 
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : I use dynamic learning rate ; if rate=r
                                         effective learning rate is (r/batchsize).

example structure:

  def trainADAGRAD(self, xs, ys, run, batchsize, rate): #start of func
  
    #*initialise anyvariable you want for ADAGRAD*
    
    w_s=[np.zeros(xx.shape) for xx in self.delLdelW]
    
    b_s=[np.zeros(xx.shape) for xx in self.delLdelB]
    
    for _ in range(run): # epochs loop      
    
      self.reinitdels() # delLdelW's and delLdelB's are initialized to all 0  
      
      for i in range(len(xs)): 3 loop for each train data point
        curinp=xs[i] # i-th data point
        
        self.feedFrwd(curinp,self.weights, self.biases) # doing forwarding
        
        self.backProp(i, ys) # doing backprop
        
        if i%(batchsize-1)==0 and i!=0: # checking for batch size
        
          #*code for ADAGRAD optimiser*
          
          # the only variables of my code you need is self.weights[*use proper indexing*], self.biases[*use proper indexing*],  self.delLdelW[*use proper indexing*],  self.delLdelB[*use proper indexing*]
          
          # all 4 of the variables above are a 3D list, with number of elements = number of hidden layers+1 ; each element is a 2D matrix of size (no.of inputs X no.of outputs)
          
          self.reinitdels() # delLdelW's and delLdelB's are initialized to all 0
          
      self.AccuracyLossLogging(_, xs, ys) # logging train, test accuracy and loss
      
    self.TestAccuracy(self.xtest, self.ytest) # for test data confusion matrix

========================================================================================
after this add this line => elif strr=="NADAM" : self.trainADAGRAD(xs, ys, run, batchsize, rate) to "trainParent" function of MLSN class.
That's all.
========================================================================================



















>> Question 5 information >>
For this question wandb automatically plots validation accuracy vs date created.
in report I attached all 350+ plots summary



















>> Question 6 information >>
For this question wandb automatically plots parellel coordinate plots with importance and correlation summary..
in report I attached all results.























>> Question 7 information >>
For this question wandb automatically plots confusion matrix.
in report I attached confusion matrix of one of the best results among my 350+ runs.
Also created a heatmap(another way to view confusion matrix) of the same run as above.



















>> Question 8 information >>
i imported all necessary modules first including fashion_mnist 
loaded trainx, trainy, testx, testy

link of direct colab code :
https://colab.research.google.com/drive/1j1Z7LwWiN-rewVRjrUvL7BzUve8fLK8C
other instructions to run(in case):
just run each code cell one after another from top to bottom.(for wandb sweep to work correctly for you, you may need to log in to your own project with login key)

my main class name is "MLSN"
it takes 10 arguments -->
szInp=flattened input size of data (784 in this case; i.e. 28*28), 
hiddenL=hidden layer architechture(eg: if you want 2 layers with 64, 128 as number of neurones then pass [64, 128], a python list),
numOp=number of available classes for a data point; (here 10)
actvfunc=actiation function(string), use "sigmoid" ot "tanh" or "reLU"
initfunc=initialisation function(string), use "random" or "xavier"
lossfunc=loss function(string), use "crossentropy" or "squarederror"
lmda=regularizer coefficient(float), eg: 0.00005, 0.0002 etc
vx, vy, xtest, ytest = these last 4 arguments are valx, valy, testx, testy
example>>> as in code I declared as => def __init__(self, szInp, hiddenL, numOp, actvfunc, initfunc, lossfunc, lmda, vx, vy, xtest, ytest)
           making object is like =>    obj=MLSN(dim_inp, [16, 32], 10, "sigmoid", "xavier", "crossentropy", 0.0002, valx, valy, testx, testy)

then using this object you have to call "trainParent" function.
example>>> as in code I declared as =>   def trainParent(self, xs, ys, strr, run, batchsize, rate):
            calling function with prev. made object should be like => obj.trainParent(trainx,trainy, "NAGD", 10, 32, 3)

            above function takes 6 arguments while calling; 
            xs=training data(trainx), 
            ys=training lables(trainy), 
            strr=optimisation func. (string: use any one of ['GD', 'MBGD', 'NAGD', 'SGD', 'RMSPROP', 'ADAM', 'NADAM'])
            run=number of epochs(integer) : eg: 5, 7, 10 etc
            batchsize=batch size(integer) : eg: 32, 64, 128 etc
            rate=learning rate (float) : passing rate=3 in above example but I use dynamic learning rate ; 
                                    so effective learning rate is (rate/batchsize); here in above example it is (3/32).


I have used,(you may see after clicking the above link)
             obj=MLSN(dim_inp, [32], 10, "reLU", "xavier", "crossentropy", 0.0001, valx, valy, testx, testy)
             a1,b1,c1=obj.trainParent(trainx,trainy, "GD", 5, 32, 3.1)
             obj1=MLSN(dim_inp, [32], 10, "reLU", "xavier", "squarederror", 0.0001, valx, valy, testx, testy)
             a2,b2,c2=obj1.trainParent(trainx,trainy, "GD", 5, 32, 3.1)
i.e. activation reLU, batch_size 32, 10 epochs, activation function xavier, hidden layer architecture [32], 
     optimization function Gradient Descant, effective learning rate (3.1/batch_size), regularizing coeff. 0.0001,
     loss crossentropy. 

## a1, a2 training accuracy figure : 2 (crossentropy vs squarederror)
## b1, b2 validation accuracy figure : 1 (crossentropy vs squarederror)
## c1, c2 validation accuracy figure : 3 (crossentropy vs squarederror)
## I plotted each pair with plottttt function(user defined)
like this>>>
Plottttt(a1, a2, "Training accuracy(crossentropy vs squarederror)")
Plottttt(b1, b2, "validation accuracy(crossentropy vs squarederror)")
Plottttt(c1, c2, "Test accuracy(crossentropy vs squarederror)")

We can conclude that crossentropy is a better loss function here than squarederror.
You may change the parameters given in function in colab code (link above) according to readme.md instructions for question 8 
and see how the plots come.





















>> Question 9 information >>
# github codes link
https://github.com/sayan-chandra/CS6910_deepLearning
# go to Assignment_1(ffnn) folder.
>>>>>>> ad1b0c332302326a34bad5e70dd21c0eb17a2ae8
