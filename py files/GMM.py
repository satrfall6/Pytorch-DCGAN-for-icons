#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:08:31 2019

@author: s4503302
"""


"""##Kmeans clustering

#### Have to deal with cuda memory issue first
if want to train autoencoder again, have to re-run this part
"""


import sys
sys.path.append('/Users/s4503302/Documents/LLD_DCGAN/')
from Autoencoder import Autoencoder
from sklearn.mixture import GaussianMixture as GMM
import os




document = os.path.join(os.path.expanduser("~"), "/Users/s4503302/Documents/LLD_DCGAN")
loadPath_2x = os.path.join(document, "icon_2x.pt")
loadPath_4x = os.path.join(document, "icon_4x.pt")

#encoded result are obtained from the Autoencoder
encoded = autoencoder(icon_combined.to(device))[0]
dim =encoded.shape
encoded=encoded.view(dim[0],dim[1]*dim[2]*dim[3])
encoded = encoded.detach().cpu().numpy()
encoded.shape

AIC = {}
BIC = {}
GMM_index = {}
for k in range(80, 101):
    gmm = GMM(n_components=k, max_iter=5, random_state = 4503).fit(encoded)
    AIC[k] = gmm.aic(encoded) 
    BIC[k] = gmm.bic(encoded)
    GMM_index[k] = gmm.predict(encoded)
plt.figure()
plt.plot(list(AIC.values()),AIC.keys())
plt.plot(list(BIC.values()),BIC.keys())        
plt.xlabel("Number of Components")
plt.ylabel("AIC & BIC")
plt.show()

font = {
        'color':  'blue',
        'size': 16,
        }

plt.figure()
plt.plot(list(AIC.keys()),list(AIC.values()),label = "AIC")

plt.plot(list(BIC.keys()),list(BIC.values()),label = "BIC")   
plt.legend(loc="best")
plt.xlabel("Number of Components",fontdict=font)
plt.ylabel("value",fontdict=font)
plt.title("AIC & BIC for GMM",fontdict=font)
plt.show()


AIC_path = os.path.join(document,'AIC_80_100')
BIC_path = os.path.join(document,'BIC_80_100')
GMMindex_path = os.path.join(document,'clusterIdx_80_100')
torch.save(AIC, AIC_path)
torch.save(BIC, BIC_path)
torch.save(GMM_index, GMMindex_path)
