#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 22:10:47 2019

@author: s4503302
"""


"""##loading and preprocessing data"""
import h5py
import torch.utils.data as tud
import torch
import torchvision.transforms as transforms
import os

#change the path to where both files are
document = os.path.join(os.path.expanduser("~"), "/Users/s4503302/Documents/LLD_DCGAN")
loadHDF5_Path_2x = os.path.join(document, "LLD-icon-sharp.hdf5")
loadHDF5_Path_4x = os.path.join(document, "LLD-icon.hdf5")

## 


class loadHDF5(tud.Dataset):
    '''
    Image loading and processing: load data from HDF5 and normalize, then save 
    as a tensor
    Input: the location of file
    
    Output: [n,3,32,32] tensor without normalizing 
    '''
    def __init__(self, file_path,transform=None):
        super(loadHDF5, self).__init__()
        h5_file = h5py.File(file_path)
        self.data = h5_file.get('data')
        self.target = h5_file.get('label')
        self.transform = transform
        
    def __getitem__(self):   

        return torch.from_numpy((self.data[:,:,:,:])).float()


    def __len__(self):
        return self.data.shape[0]

      
## loading image from HDF5 
image_size = 32
transform=transforms.Compose([
                               transforms.ToPILImage(),
                               transforms.Resize(image_size),#
                               transforms.CenterCrop(image_size),#
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


#run through every image and apply transform to them
if __name__=='__main__':
    
    ld5=loadHDF5(loadHDF5_Path_2x)
    img = ld5.__getitem__() 
    outputs=[]
    for i,ch in enumerate(range(img.size(0)), 0):
        tensor = transform(img[ch,:,:,:])
        tensor = tensor.unsqueeze(0)
        outputs.append(tensor)
        
    icons32_2x = torch.cat(outputs, dim=0)
    loadPath_2x = os.path.join(document,'icon_2x.pt')
    torch.save(icons32_2x, loadPath_2x)
    
    ld5=loadHDF5(loadHDF5_Path_4x)
    img = ld5.__getitem__()
    outputs=[]    
        
    for i,ch in enumerate(range(img.size(0)), 0):
        tensor = transform(img[ch,:,:,:])
        tensor = tensor.unsqueeze(0)
        outputs.append(tensor)
        
    icons32_4x = torch.cat(outputs, dim=0)
    loadPath_4x = os.path.join(document,'icon_4x.pt')
    torch.save(icons32_4x, loadPath_4x)
    
    icon_combined = torch.cat((icons32_2x,icons32_4x), dim=0)
    icons_32_4x = None
    icons_32_2x = None
