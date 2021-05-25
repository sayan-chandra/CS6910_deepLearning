# Link For Github => [click for github repo](https://github.com/sayan-chandra/CS6910_deepLearning.git)
# Link For Wandb Report => [click for report](https://wandb.ai/blackcloud/cs6910_dl_assignment_3/reports/DL_CS6910-Assignment-3--Vmlldzo3MDMzMzI?accessToken=u501zvl5gwejmdvqsl1fvm4sp1rxkhira5w7c7ysjd2f1vm2h18pch5zn3ozbj4x)
# [Go to All Links section](#all-links)

>>>>## `(Question-1)`
[Google colab Assgn 3 question 1](https://colab.research.google.com/drive/1PqaFzsJfsrYL6SUnjrvSUh0p1BgFclII?usp=sharing)
>1) This just making of the cnn architechture as mentioned in question 1. this code is not meant for running.


>>>>## `(Question-2)`
[Google colab Assgn 3 question 2]( https://colab.research.google.com/drive/1_7Lbtxa8mjMJA6BgT2ShLjbdpEwtw8vn?usp=sharing)
>1) Run all the cells one after another.
>2) Used !tar and !wget to download and unzip the dataset.
>3) setWandb() and download_unzip() function autometically works without any argument.
>4) A global variable WANDB is set to 0 by afforsaid functions and hence the code will run for a default configuration that I set. You can change it also.
>4) preprocessing_TrainValTest()` function makes 5 things dataType, datapath, mode, isTrain, en_format.
>4) setPaths() function sets all required root , train, val, test paths
>5) In the ReccNeuralNet() function I make desired rnn with enough flexibilityu given as required.
>> ## the arguments of above function.
>> en = number of encoder/decoder layers (int type), input_embedd_sz=output size of encoder and dcecoder embedding (int type), latent_dim=hidden layer size of encoder and decoder(vanillaRnn or lstm or gru) (int type), drop=dropout percentage for enoders/decoders (float type), cell=type of rnn cell (string type)
>6) The inferenceModel() function is for re-constructing the inference model from the actual model according to the layers used i.e. embedding, encoder and decoder.
>> ## the arguments and working of above function.
>> it takes 5 arguments mdl=keras model (keras.model type), ld=list of encoder/decoder layers latent dim i.e. hidden layer size (list of int type), emb_var=list of 2 things; encoder embedding size and decoder embedding size (list of int type), attn=attention yes or no (string type, for question 2 set to "no"), cell=type of rnn cell (string type).
>7) beam_search() is the decoder helper function taking 2 argument ; a matrix (numpy array type) and beam width (int type).
>8) charByCharDecoding() is the post_used_function after inferenceModel for testing on validation and test data that takes one-hot-encoding of input english word as input and sequentially predict hindi words. This function is needed because ideally for test data we do not have the output hindi word, hence teacher-forcing is never meant for test data.
>9) WriteToCSVFile() is a function to save all predicted test words in a ".csv" file automatically after all test prediction.
>10) summary_architechture() is just to print the architechture of the model made by custom configuration. Takes argument a model <keras.model type>
>11) plotModel() returns a very good block representation of model by which one can understand how the i/p and o/p of each layer works.
>12) class funcForWordAcc() is a custom callback function implemented class overriding a keras function Callback, what it does is takes each predicted and input word after each epoch and match it totally with all chars, if full word matches it does count+=1.
>13) the TRAIN() function is to train the custom model we made , this is the main training function that uses model.fit(). 
>> ## Arguments taken
>> model=custom made model with desired combination <keras.model type>, opti=optimiser function (string type, "adam" or "sgd" or "rmsprop" only.), lr=learning rate (default value 1e-3;change while calling the func.), float type), epk=number of epochs (int type, default value 15;change it while calling func.)), bs=(int type, default value 64;change it while calling func.))
>14) WORDACCVAL() computes inference on validation data using beam search(width 3) after all epochs are done (after model.fit is done basically) for any model given. It takes sigficant time hence commented out in my default model. No arguments needed while calling, all set as default.
>15) Test_Acc_With_Best_Model_In_Sweep() computes inference on test data using beam search(width 3/4) after model is done with training basically.
>16) the SweepParent() function is for running the sweeps is global var WANDB=1 else it will run the custom configuration for model det by me(You can change configs , INSTRUCTIONS are below).
>> ## How To Run Customised Configuration.
>> USE THE `SweepParent()` FUNCTION ( go to the `else` part that runs when `WANDB IS SET TO 0` ) TO CHANGE ALL THE VARIABLES MENTIONED BELOW FROM POINT 17 to 19.
>17) set 5 variables a(int),b(int),c(int),d(float),e(string; "vanillaRnn" or "lstm" or "gru") i.e respectively number of encoder/decoder layers , input_embedding_size for encoder/decoder embedding, hidden_layer_size for encoder/decoder rnn cells ,dropout for encoder/decoder rnn cells , cell_type as type of rnn cell,
>18) set 4 variables f(string; "adam" or "sgd" or "rmsprop"),g(float),h(int),i(int) i.e. respectively optimizer function, learning_rate, number of epochs, batch_size
>19) you can find this 5+4=9 variables in the `else` part of SweepParent function `24th` and `25th` line. Set them as per your wish if you want according to instruction given in point 17 and 18 above.

                                eg: 
                                    a,b,c,d,e=2, 128, 256, 0.5, "lstm"
                                    f,g,h,i="rmsprop", 3e-3, 17, 32

>20) After each epoch finishes it will print trainig and val char accuracy and loss, Training word accuracy, val word accuracy,


>>>>## `(Question-5-6)`
>>>>[Google colab Assgn 3 question 5, 6]( https://colab.research.google.com/drive/1w337FBwZz0231G24H4hqdIamYOneHo-V?usp=sharing)
>1) Run all the cells one after another.
>2) Used !tar and !wget to download and unzip the dataset.
>3) setWandb() and download_unzip() function autometically works without any argument.
>4) A global variable WANDB is set to 0 by afforsaid functions and hence the code will run for a default configuration that I set. You can change it also.
>4) preprocessing_TrainValTest()` function makes 5 things dataType, datapath, mode, isTrain, en_format.
>4) setPaths() function sets all required root , train, val, test paths
>5) In the ReccNeuralNet() function I make desired rnn with enough flexibilityu given as required.
>> ## the arguments of above function.
>> en = number of encoder/decoder layers (int type), input_embedd_sz=output size of encoder and dcecoder embedding (int type), latent_dim=hidden layer size of encoder and decoder(vanillaRnn or lstm or gru) (int type), drop=dropout percentage for enoders/decoders (float type), cell=type of rnn cell (string type)
>6) The inferenceModel() function is for re-constructing the inference model from the actual model according to the layers used i.e. embedding, encoder and decoder.
>> ## the arguments and working of above function.
>> it takes 5 arguments mdl=keras model (keras.model type), ld=list of encoder/decoder layers latent dim i.e. hidden layer size (list of int type), emb_var=list of 2 things; encoder embedding size and decoder embedding size (list of int type), attn=attention yes or no (string type, for question 2 set to "yes"), cell=type of rnn cell (string type).
>7) beam_search() is the decoder helper function taking 2 argument ; a matrix (numpy array type) and beam width (int type).
>8) charByCharDecoding() is the post_used_function after inferenceModel for testing on validation and test data that takes one-hot-encoding of input english word as input and sequentially predict hindi words. This function is needed because ideally for test data we do not have the output hindi word, hence teacher-forcing is never meant for test data.
>9) WriteToCSVFile() is a function to save all predicted test words in a ".csv" file automatically after all test prediction.
>10) summary_architechture() is just to print the architechture of the model made by custom configuration. Takes argument a model <keras.model type>
>11) plotModel() returns a very good block representation of model by which one can understand how the i/p and o/p of each layer works.
>12) class funcForWordAcc() is a custom callback function implemented class overriding a keras function Callback, what it does is takes each predicted and input word after each epoch and match it totally with all chars, if full word matches it does count+=1.
>13) the TRAIN() function is to train the custom model we made , this is the main training function that uses model.fit(). 
>> ## Arguments taken
>> model=custom made model with desired combination <keras.model type>, opti=optimiser function (string type, "adam" or "sgd" or "rmsprop" only.), lr=learning rate (default value 1e-3;change while calling the func.), float type), epk=number of epochs (int type, default value 15;change it while calling func.)), bs=(int type, default value 64;change it while calling func.))
>14) WORDACCVAL() computes inference on validation data using beam search(width 3) after all epochs are done (after model.fit is done basically) for any model given. It takes sigficant time hence commented out in my default model. No arguments needed while calling, all set as default.
>15) Test_Acc_With_Best_Model_In_Sweep() computes inference on test data using beam search(width 3/4) after model is done with training basically.
>16) the SweepParent() function is for running the sweeps is global var WANDB=1 else it will run the custom configuration for model det by me(You can change configs , INSTRUCTIONS are below).
>> ## How To Run Customised Configuration.
>> USE THE `SweepParent()` FUNCTION ( go to the `else` part that runs when `WANDB IS SET TO 0` ) TO CHANGE ALL THE VARIABLES MENTIONED BELOW FROM POINT 17 to 19.
>17) set 5 variables a(int),b(int),c(int),d(float),e(string; "vanillaRnn" or "lstm" or "gru") i.e respectively number of encoder/decoder layers , input_embedding_size for encoder/decoder embedding, hidden_layer_size for encoder/decoder rnn cells ,dropout for encoder/decoder rnn cells , cell_type as type of rnn cell,
>18) set 4 variables f(string; "adam" or "sgd" or "rmsprop"),g(float),h(int),i(int) i.e. respectively optimizer function, learning_rate, number of epochs, batch_size
>19) you can find this 5+4=9 variables in the `else` part of SweepParent function `24th` and `25th` line. Set them as per your wish if you want according to instruction given in point 17 and 18 above.

                                eg: 
                                    a,b,c,d,e=2, 64, 512, 0.2, "lstm"
                                    f,g,h,i="adam", 3e-3, 17, 32

>20) After each epoch finishes it will print trainig and val char accuracy and loss, Training word accuracy, val word accuracy,
>21) for 6th question Vizualisation code is also given in this code, vizualisation gif is added in wandb report

# All links
[Google colab Assgn 3 question 1](https://colab.research.google.com/drive/1PqaFzsJfsrYL6SUnjrvSUh0p1BgFclII?usp=sharing)

[Google colab Assgn 3 question 2]( https://colab.research.google.com/drive/1_7Lbtxa8mjMJA6BgT2ShLjbdpEwtw8vn?usp=sharing)

[Google colab Assgn 3 question 5, 6]( https://colab.research.google.com/drive/1w337FBwZz0231G24H4hqdIamYOneHo-V?usp=sharing)
