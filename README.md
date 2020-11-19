# Bayesian-Convolution-Neural-Network-Road-Crack-Detection
A Bayesian Convolutional neural network to detect cracks in concrete structures from images. The Bayesian Convolution Neural Networks takes the uncertainty of the weight parameters into account which potentially could lead to more reliable decisions. Theoretically, we model a probability distribution over the model weights to include a form of  uncertainty. According to Gal and Ghahramani (2015a), adding drop-out layers after every convolution layer, during training and testing, is approximately corresponds to Variational Inference and enables us to sample from the posterior distribution of the predicted output. 

## Road crack dataset
The datasets contains images of concrete surfaces with(postive) and without(negative) cracks. Each class contains 20000 image in a seperate folder, postive and negative, leading to a total of
40000 images. The image are in RBG channel and have the following size, 227 x 227 x 3.  The data is publically availible at Mendely, 

https://data.mendeley.com/datasets/5y9wdsg2zt/2

The data is previously used for crack-detection;

2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin. 


Lei Zhang , Fan Yang , Yimin Daniel Zhang, and Y. J. Z., Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road Crack Detection Using Deep Convolutional Neural Network. In 2016 IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052

Gal, Y. and Ghahramani, Z. (2015a). Bayesian convolutional neural networks with bernoulli approximate variational inference. arXiv preprint arXiv:1506.02158.

## Data preparation 
The data is transformed to a 100 x 100 x 1 dimension and is split up into train, validation and test data sets (75% - 15% -15%). Furtermore, each image is given its  corresponding label, positive (1) or negative (0) and we random shuffled the order of the images. In the folder Codes one can find the code to prepare the raw downloaded dataset in "Prepare_data.py"

## Models 
In the folder codes one can find the codes for each model.
- CNN: Convolution neural network
- CNN_D: Convolution neural network with regularization dropout
- BCNN: Bayesian convolution neural network with dropout layers after every convolution operation to cast the bernouille approximation variational inference with Monte Carlo dropout

## Output
Final the executed code in the format of a Juypter notebook are stored under the file Output.ipyn 

## Summary obtained results
... TODO..


