# Link For Github => [click for github repo](https://github.com/sayan-chandra/CS6910_deepLearning.git)
# Link For Wandb Report => [click for report](https://wandb.ai/blackcloud/cs6910_dl_assignment_2/reports/DL_CS6910_Assignment-2_IITM--Vmlldzo1OTYxNDE?accessToken=g26e1nk4hyx3trur7jo452o5xm16eogt5rak27d2c7d0nhb51z5878ewlg7khker)
# [Go to All Links section](#all-links)
>>>>## `Part_A(Question-2)`
>1) Run all the cells one after another.
>2) Used split-folders module to split train into train and val so that all classes has same number of images.
>3) `loadTrain_Val_TestData()` function makes 6 things, train, test and val data and train, test and val data loader with required batch size and shuffling if required.
>4) `CNN` is my defined class fot the architecture of convolutional neural net.
>5) I have set a variable called WANDB=0; so while checking it won't run sweeps. It will ony run one good configuration that is set by me.
>> ## How To Run Customised Configuration.
>> USE THE `SweepConfig` FUNCTION ( in else part that means WANDB IS SET TO 0 ) TO CHANGE ALL THE VARIABLES MENTIONED BELOW FROM POINT 6 TO 11.
>6) class CNN takes arguments cnn_config, in_channels=3, num_classes=10, denselayer=64, prob=0.2 where in_channels and num_classes are fixed with 3 and 10 because that is mandetory. You can change denselayer size and prob(drop out probability, prob=1 means all links dropped out, prob=0 means all links retained).
The cnn_config is a list of lists. Each list = in_channels, out_channels(number of filters), convkernel size, conv kernel stride, padding, maxpoolkernel size, maxpool stride so i_th list = for i_th conv-relu-maxpool block

                                eg: 
                                    cnn_config=[[3, 64, 11, 2, 0, 2, 1],
                                               [64, 64, 7, 2, 0, 2, 1],
                                               [64, 32, 5, 2, 0, 2, 1],
                                               [32, 32, 3, 1, 0, 2, 1],
                                               [32, 32, 3, 1, 0, 2, 1]]
>7) now we make an object of class CNN i.e.

                                   cnnModel=CNN(cnn_config, denselayer=64, prob=0.2)
> 8) bsz=batch size variable
>9) For optimiser I am using SGD with cosine annealing with warm restarts. You can change them if you want with other optimiser, learning rate, weightdecay, momentum and all.

                                   optimizer = torch.optim.SGD(cnnModel.parameters(), 0.0022, momentum=0.94, weight_decay=0)
                                   scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(len(train_data_loader)/bsz)))
                                   
>10) epk = epoch variable, lossfunc=cnn.CrossEntropyLoss() because we are dealing with multi class classification.
>11) After all these are set then we call trainMyModel() that takes these arguments : epk, cnnModel, train_data_loader, heyGPU, optimizer, scheduler, val_data_loader, batchnorm(1 means yes 0 means no), lossfunc (change whatever variables from point 6 to 11 as stated above & then this function will get called)
>12) After each epoch finishes it will print Current epoch, Training loss, val accuracy and val loss.


>>>>## `Part_A(Question-4)`
>1) Run all the cells one after another.
>2) This code is anyway wandb free code.
>3) here I set the cnn_config as stated above with the model that gave best accuracy according to sweep.
>4) We get the test accuracy after training the model for 14 epochs.
>5) Then we take 10 shuffled test images and plot the image with true label, predicted label and with bar plot of output probability.
>6) Then we take the learned filter weights (64 filters in 8 X 8 grid) and plot them.


>>>>## `Part_A(Question-5)`
>1) Run all the cells one after another.
>2) For this I selected 10 images of train_data and did guided backprop using output layer.
>3) This code is anyway wandb free code.
>4) The object of class `guidedBackprop()` takes my cnnModel as input and do this guided backpropagation. After this I plot the gradients using imshow().


>>>>## `Part_B(Question-1-2-3)`
>1) Run all the cells one after another.
>2) Used split-folders module to split train into train and val so that all classes has same number of images.
>3) `loadTrain_Val_TestData()` function makes 6 things, train, test and val data and train, test and val data loader with required batch size and shuffling if required.
>4) `CNN` is my defined class fot the architecture of convolutional neural net.
>5) I have set a variable called WANDB=0; so while checking it won't run sweeps. It will ony run one good configuration that is set by me.
>> ## How To Run Customised Configuration.
>> USE THE `SweepConfig` FUNCTION ( in else part that means WANDB IS SET TO 0 ) TO CHANGE ALL THE VARIABLES MENTIONED BELOW .
 
      cnnModel, inpsz=initialize_model('resnet50') ========> change model name using any of ['resnet50', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inceptionv3']
      cnnModel = freezesome(cnnModel, _true=1, k=0) =====> change how many layers from the last(parameter, k) you want to unfreeze, rest will be automatically freezed.
      bsz=16 =====> change batch size
      train_data, test_data, val_data, train_data_loader, test_data_loader, val_data_loader = loadTrain_Val_TestData(bsz, inpsz)
      heyGPU=torch.cuda.is_available()
      if heyGPU: cnnModel=cnnModel.cuda()
      lossfunc=cnn.CrossEntropyLoss()
      print(heyGPU)
      optimizer = torch.optim.SGD(cnnModel.parameters(), lr=0.0002, momentum=0.94, weight_decay=0.00002) ======> change optimiser func., learning rate, momentum, weightdecay
      scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(np.ceil(len(train_data_loader)/bsz)))
      epk=8 =======> change epochs
      
>6) After all these are set then we call trainMyModel() that takes these arguments : epk, cnnModel, train_data_loader, heyGPU, optimizer, scheduler, val_data_loader, lossfunc
>7) After each epoch finishes it will print Current epoch, Training loss, val accuracy and val loss

# All links
[Google colab Assgn 2 part A question 1](https://colab.research.google.com/drive/1i10wa6inYnYftLRR93_BL1YN1J7j2C9p?usp=sharing)

[Google colab Assgn 2 part A question 2]( https://colab.research.google.com/drive/1wrIz0Qn_VuoTGJ0yZ3IAUJLe6rFSaItb?usp=sharing)

[Google colab Assgn 2 part A question 4 a,b,c]( https://colab.research.google.com/drive/12pai2eNiu-U9E09QWohrXRiBNCe538Be?usp=sharing)

[Google colab Assgn 2 part A question 5](https://colab.research.google.com/drive/1AX969M79hXwdP_VYSI_r2SUL70mZcsLl?usp=sharing)

[Google colab Assgn 2 part B question 1,2,3](https://colab.research.google.com/drive/1lFXfKf4p_N7Ne3aiIq9YhhlYt-9ixq3e?usp=sharing)

[YouTube Video for face mask detection](https://www.youtube.com/watch?v=Y1T9et7j2x8)

[YouTube Video for person, car, dog detection](https://www.youtube.com/watch?v=5Lafwa2rgi8)
