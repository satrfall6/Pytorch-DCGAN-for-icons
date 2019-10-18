#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:09:29 2019

@author: s4503302
"""
import torch
import os
import torch.nn as nn
#%%
#change to the path that saves "AE_weights_3C_4", and load the pre-trainined weights
document = os.path.join(os.path.expanduser("~"), "/Users/s4503302/Documents/LLD_DCGAN")
AEweights_Path = os.path.join(document, "AE_weights_3C_4")


#autoencoder to encode the input image for clustering 
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #input 32x32
            nn.Conv2d(nc, nf_auto, 4, 2, 1), 
            nn.ReLU(True),
            #state size 16x16
            nn.Conv2d(nf_auto, nf_auto*2 , 4, 2, 1),  
            nn.ReLU(True),
            #state size 8x8
            nn.Conv2d(nf_auto*2, nf_auto*2 , 4, 2, 1),  
            nn.ReLU(True),
            #output 4*4
        )
        
        self.decoder = nn.Sequential(
            #input is 4x4
            nn.ConvTranspose2d(nf_auto*2, nf_auto*2, 4, 2, 1),  
            nn.ReLU(True),
            #state size 8x8
            nn.ConvTranspose2d(nf_auto*2, nf_auto, 4, 2, 1),  
            nn.ReLU(True),
            #state size 16x16
            nn.ConvTranspose2d(nf_auto, nc, 4, 2, 1),  
            nn.Tanh()
            #output 16x16
        )
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

# number of feature maps for autoencoder  
nf_auto = 4
# 3 if the image is color, 1 if b&w
nc = 3

#set parameters for autoencoder 
autoencoder = Autoencoder()

criterion_auto = nn.MSELoss()
optimizer_auto = torch.optim.Adam(autoencoder.parameters(), lr=1e-4, betas=(0.85, 0.999),eps = 1e-5)

#apply initial weights for autoencoder
#autoencoder.apply(weights_init)


#apply saved weights for autoencoder 

Autoencoder_weights=torch.load(AEweights_Path,map_location=torch.device('cpu'))
autoencoder.load_state_dict(Autoencoder_weights['state_dict'])
optimizer_auto.load_state_dict(Autoencoder_weights['optimizer'])


#train autoencoder 
for epoch in range(num_epochs):
    for i, data in enumerate(LLDloader, 0):
      
        optimizer_auto.zero_grad()
        
        img = data.to(device)
        # get the result from the decoder
        output = autoencoder(img)[1]
        #calculae the loss between original image and the decoder
        loss_auto = criterion_auto(output, img)
        #backpropogation 
        loss_auto.backward()
        optimizer_auto.step()
        
        if i % 3450 == 0:
            print('[%d/%d][%d/%d]' % (epoch, num_epochs, i, len(LLDloader)))
                     
    if epoch % 2==0:
        print("original image: ")
        showImages(data[0:8][:][:][:])
        print("decoded image: ")
        decoded = output.detach()
        showImages(decoded[0:8][:][:][:])


if __name__=='__main__':


    check_decoder = autoencoder(icon_combined[0:64].to(device))[1]
    check_decoder.shape
    showImages(check_decoder.detach())


