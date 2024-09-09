# Automatic Modulation Classification
 Train a CNN for calssifying digital modulation
This is a part of my Master's thesis where I tested and developed some models for modulation classification.

Here I am sharing how to train a simple CNN model using pytorch and also generate you own dataset using Matlab.

I used the benchmark dataset RADIOML 2016.A (https://www.deepsig.ai/datasets/) ,  (https://github.com/radioML/dataset)

they used GNU Radio using python, but I used matlab and found thier channel characteristics to generate samples as close to theirs. the Matlab code for generating is mostly from (https://www.mathworks.com/help/deeplearning/ug/modulation-classification-with-deep-learning.html) , but I added Reyleigh channel and omitted the analogue modulation.

the link to RADIOML 2016.A dataset is ---> (https://pubs.gnuradio.org/index.php/grcon/article/view/11)
As for the CNN model in this repository I used the paper ---> (https://doi.org/10.48550/arXiv.1602.04105), its a great start to get your hands dirty and then check and develope other models.

There is this awsome survey paper from 2022 which basically covers all of the models developed up-untill 2022, which gives a great perspective and a good starting point ---> (https://doi.org/10.48550/arXiv.2207.09647)

I ran this model on 1050TI and took 40 minutes for 50 epoch with batch size of 1024.
you can find the confusion matrices and model's weights in this repository under **cnn_models** and **confusion_matrices**
The train and test losses per epoch is saved as dict in **models_results**
the matlab code and a python script for packaging data as pickle files in found at **Generating_data_Matlab**
python scripts, **CNN_training_validation** inculdes notebook for training and validating the model
**data_setup** provides functions to import, filter, create train and test splits specific for RADIOML 2016.A dataset
**utills** provides functions to save, load, some plottings and etc...

*Note: the authors of the dataset recommend to use generated datasets and real collected data : 
'''
These datasets are from early academic research work in 2016/2017, they have several known errata and are NOT currently used within DeepSig products. We HIGHLY recommend researchers develop their own datasets using basic modulation tools such as in MATLAB or GNU Radio, or use REAL data recorded from over the air! DeepSig provides several supported and vetted datasets for commercial customers which are not provided here -- unfortunately we are not able to provide support, revisions or assistance for these open datasets due to overwhelming demand!
'''
After all I hope this repository would be helpful ! 
