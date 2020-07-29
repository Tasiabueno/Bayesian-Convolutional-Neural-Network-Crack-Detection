# Bayesian-Convolution-Neural-Network-road-cracks
A Bayesian Convolutional neural network to detect cracks in concrete structures. The Bayesian Convolution Neural Networks takes the uncertainty of the weight parameters into account which potentially leads to more reliable decisions. We model probability distribution over the model weights to include the uncertainty. The concept follows the Bayesian update rule and uses the theory of Variational Inference. According to Gal and Ghahramani (2015a), adding drop-out layers to the afters each convolutin layer approximate the Variational Inference and thus the posterior distribution of the model weights. 

## Road crack dataset
The datasets contains images of concrete surfaces with(postive) and without(negative) cracks. Each class contains 20000 image in a seperate folder, postive and negative, leading to a total of
40000 images. The image are in RBG channel and have the following size, 227 x 227 pixel. The data is publically availible at Mendely, 

https://data.mendeley.com/datasets/5y9wdsg2zt/2

The data is previously used for crack-detection;

2018 – Özgenel, Ç.F., Gönenç Sorguç, A. “Performance Comparison of Pretrained Convolutional Neural Networks on Crack Detection in Buildings”, ISARC 2018, Berlin. 


Lei Zhang , Fan Yang , Yimin Daniel Zhang, and Y. J. Z., Zhang, L., Yang, F., Zhang, Y. D., & Zhu, Y. J. (2016). Road Crack Detection Using Deep Convolutional Neural Network. In 2016 IEEE International Conference on Image Processing (ICIP). http://doi.org/10.1109/ICIP.2016.7533052

Gal, Y. and Ghahramani, Z. (2015a). Bayesian convolutional neural networks with bernoulli approximate variational inference. arXiv preprint arXiv:1506.02158.

## CNN Architecture
Below you can find each step which was applied in order to obtain the results in the paper. 

### Data preparation 



