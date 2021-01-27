# Bayesian-Convolutional-Neural-Network-Crack-Detection
Maintenance is an essential task in the industrial sector—the rise of image-detection techniques shows the potential to improve maintenance processes. Machine learning algorithms are appealing with their excellence in prediction accuracy and scalability. However, their” black box” behavior and the inability to include uncertainty in the predicted values are the main drawbacks. The estimated class likelihood from the softmax activation lacks model confidence and can be misleading in decision-making. The Bayesian convolution network uses dropout to cast variational inference and obtains an approximated predictive posterior distribution. The predictive variances aggregate from samples of the approximated predictive posterior and serve as a measure of uncertainty. The Bayesian convolution neural network has advantages in image-based maintenance applications. Domain experts affirm the novel method’s potential—model uncertainty gains an increased interest in the image-based maintenance industry sector and is often a discussion point with their clients. The predictive posterior variance as an evaluation criterion eliminates the number of false-negative predictions and illuminates uncertain predictions.

![alt text](https://github.com/Tasiabueno/Bayesian-Convolution-Neural-Network-Road-Crack-Detection/blob/master/Data/B-cnn.png)

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
We compare three different convolution neural networks in terms of predictive accuracy and uncertainty analysis. The experimental set-up provides evidence to answer the research question; can we quantify the uncertainty in image-based crack detection for concrete structures using Bayesian convolution neural networks? If we can quantify the uncertainty, our interest lies in the practical application of maintenance strategies. Can the image-based detection method using BCNN contribute to maintenance strategies? 

In the folder codes one can find the codes for each model.
- CNN: Convolution neural network
- CNN_D: Convolution neural network with regularization dropout
- BCNN: Bayesian convolution neural network with dropout layers after every convolution operation to cast the bernouille approximation variational inference with Monte Carlo dropout

## Output
In order to reproduce the obtained results take the following steps:
- Download the data set
- Prepare and clean the data with the code "Prepare_data.py"
- Download the codes of the applied models; "CNN.py","CNN_D.py","BCNN.py"
- Download the final executed output code, in the format of a Juypter notebook are stored under the file "Output.ipyn"

! Make sure you store the downloaded codes and the final output notebook in the same folder

## Summary obtained results
Comparing the three convolution neural networks, we criticize on three components; predictive accuracy, uncertainty assessment, and practical applicability. The Bayesian method slightly outperforms the CNN and CNN_ D in terms of prediciton accuracy. Furthemore, the novel method can measure uncertainty through its predictive posterior variance and eliminate the number of false negatives relative to the CNN method. We verify and substantiate the obtained results with the application of CQM, a quantitative consultancy company that previously applied a CNN method for a maintenance project. Summarizing the information obtained from the domain expert, we can conlude the folloinwg statements; The company is intressented in applying novel method to include a source of uncertainty. Uncertainty evaluation is an often discussed topic among their clients. The trade-off to appy the method for futur project depends on the application and the costs of false postives. 

The field of uncertainty in deep learning applications is significantly advancing and is a topic with considerable potential for further research. More specifically, it might be interesting to include visualizing results for maintenance applications in the future. A technique that indicates the origin of the uncertainty directly on the image distinguishes the aleatoric and epistemic uncertainty. Furthermore, an exciting field of research for maintenance applications is the combination of uncertainty evaluation and Humans in the loop. Humans in the loop is a combination of the machine learning model and human interaction with continuous feedback loops, also known as active learning [Settles, 2009]. The combinations of active learning and BCNN could potentially be the step towards automated image- based maintenance strategies.


## Thesis Defense
In the presentation below you can find my thesis defense for my Masters in Econometrics & Operations Research at the Erasmus University. The presentation will give you a short overview of the motivation to apply the BCNN and some guidance that can help you to follow my codes and the architecture. If you have any question regarding the thesis or presentation, please feel free to reach out to me!



![alt text](https://github.com/Tasiabueno/Bayesian-Convolution-Neural-Network-Road-Crack-Detection/blob/master/Data/Powerpoint.pptx)
