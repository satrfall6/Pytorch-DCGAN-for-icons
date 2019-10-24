
# Automatic Generation of Icons

## Description
This is a study of how to generate various kinds of icons. GAN is notorious for its instability during training. In this study, we tested parameters of model that we can adjust, to stablize the model. Included the number of layers, learning rate, label smoothing, adding noise, and using spectral normal.

 In addition, we also compare the results of DCGAN to the conditional DCGAN. This dataset does not have labels. In order to implement conditional DCGAN, we use Gaussian Mixture Model to create 100 clusters and label the images based on which cluster they belong to. Then, we merge the labels into the convolution layers to train the model.

## Environment
* Python version: 3.7
* torch version : 1.2.0
* tests are run on colab notebook

## Contents
* Dataloader with Transform 
* DCGAN
* Autoencoder  
* GMM  
* Conditional DCGAN

### __Data Loader__ 
The data is stored as .hdf5 format, and we use this data loader to load the data. The dataloader is developed by . We added the transform into the data loader so it can resize, center, and normalize the image.



### __DCGAN__
In the DCGAN, we improve the stability of the model by 4 things. Firstly, We use Two Time-Scale Update Rule, which means by using different learning rate for the generator and the discrminator. Secondly, we add the spectral norm to each layer of the dscriminator and the generator. Thirdly, we add noise the the inputs of the discriminator. Finally, the set the real and fake labels to the numbers that are close to 0 and 1, instead of using exactly 0 and 1.



* Stucture of DCGAN:
![DCGAN](https://i.imgur.com/JpmG9km.png)
Figure 1 the structure of DCGAN

### __Autoencoder__ 
By using the autoencoder, we can keep the relation between pixel and pixel, then we cluster the image from the latent space.

* Stucture of autoencoder:
![AE](https://i.imgur.com/ADThmZB.png)
Figure 2

### __GMM__
We use GMM to create 100 clusters, then label them from 1 to 100 base on which cluster the images belong to. The AIC and BIC of GMM is shown as below

* Labels from the GMM:\
![GMM](https://i.imgur.com/naQ5ICb.png) \
Figure 3 

* AIC and BIC of the GMM: \
![AICBIC](https://i.imgur.com/sAFcObS.png)\
Figure 4

### __Conditional DCGAN__
The label from the GMM is coded as onehot label, and merged to the convolutional layers in the way as shown in Figure 5. 

* Stucture of Conditional DCGAN:
![CDCGAN](https://i.imgur.com/A8ehR9f.png)
Figure 5 

## Results

* Results of DCGAN:\
![resultDCGAN](https://i.imgur.com/tMJtQGI.png)\
Figure 6 


* Results of Conditional DCGAN:\
![resultCDCGAN](https://i.imgur.com/DxxF7F5.png)\
Figure 7 

## Reference




