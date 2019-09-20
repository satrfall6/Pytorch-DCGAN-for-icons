# GAN_LLD_32x32







# Automatic Generation of Icons

The model here is to show the idea of how to set the parameters for each model. They might be differ form the modle I use for generating the results.  

## Environment
Pytorch version:

## Contents


1.dataloader \
2.DCGAN\
3.Autoencoder  \
4.Clustering  \
5.Comparisons of different parameters of NNs 

## Data Loader 

### Loading 
Below is the data loader for loading HDF5. it requires the package "h5py" and "torch.utils.data"

```
import h5py
import torch.utils.data as tud
```

```
class loadHDF5(tud.Dataset):
    '''
    this will output the [n,3,32,32] tensor without normalizing 
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
```
For example, we can set the path and file name into this python class, and use the methos "__getitem__()" to load the image. This method would return a 4D tensor, EX: [64:3:32:32].
```
ld5=loadHDF5('/your path/file_name.hdf5')
img = ld5.__getitem__() 
```
### Transform 
Since the file it returns is a 4D tensor, I could not apply transform, i.e. resize, normalize, etc. So I use a for loop to transform all of the images inside then cat them back to a 4D tensor. Probably there is more efficient way, but I didn't put too much time on it. This way occupies a lot of memories, so have to clean the list after cat them to tensor.  
```
outputs=[]
for i,ch in enumerate(range(img.size(0)), 0):
    tensor = transform(img[ch,:,:,:])
    tensor = tensor.unsqueeze(0)
    outputs.append(tensor)
    
icons_32 = torch.cat(outputs, dim=0)
```


## DCGAN


  ### Hyperparameters
  ```
# Batch size during training
batch_size = 64
# size of images
image_size = 32
# Number of training epochs
num_epochs =50
# Number of channels in the training images. For color images this is 3, black & white is 1.
nc = 3
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number feature maps(filters) in generator
ngf = 64
# Number feature maps(filters)  in discriminator
ndf = 64
# Beta1 hyperparam for Adam optimizers, default is 0.9
beta1 = 0.5
# Learning rate for optimizers
lr = 0.00003
  ```


  ### Generator
Unlike Keras, we have to give the number of input channels and the number of padding in Pytorch. Take "nn.Conv2d(nc, ndf, 4, 2, 1), " in the discrminator as an example. The First variable "nc" represents the number of channels, for colored image it is 3. The second variable "ndf" represent the number of feature maps to output, which is shown in Figure 1. \
The last 3 variables define the size of output image size, kernel size(k) = 4, stride(s) = 2, and padding(p) = 1. Let l = 32 be the length of input image. The length of output image(l')  is equal to (l-k)/2+1 = 16 in this case. Since the feature maps and the image size need to be matched for each layer, we can use the equation to make sure every size is correct for each layer.   

```
#generator
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 4, 4, 1, 0), 
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1), 
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d( ngf * 2, ngf * 1, 4, 2, 1), 
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),
            # state size. (ngf*1) x 16 x 16
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1), 
            nn.Tanh()
            # state size. (nc) x 32 x 32
            )
        
            # forward method
    def forward(self, input):

        return self.main(input)
```
  ### Discriminator


```
#discriminator   
class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0),  
            nn.Sigmoid()
        )

            # forward method
    def forward(self, input):
        return self.main(input)
```



  ### Setting
  We have to save the model to a variable for running first. The Criterion here is the loss function, we can choose different loss functions such as, CrossEntropyh, MSE, etc. Notice that the loss function is from the package "torch.nn". \
  \
  Fort he optimizer, we can choose like Gradient Descent or Stochastic Gradient Descent, etc. The optimizer comes from the "torch.optim" package. \
  \
  The fixed_noise is just for checking the output for the generator. Notice that it has to be set ".to(device)" for GPU to run. The real_label and fake_label is to calculate the loss for the discriminator's judgement for the images from the real dataset and the generator. It doesn't mean this is a kind of supervised algorithm.  \
  \
  I stored the loss for the discriminator and the generator to check if the model runs properly during training. However, low loss does not mean the models can generate images properly, we still need to check the results that are generated by the generator.\
  \
  Using ".cuda()", we can set our model to use the resource of GPU during training. \
  

   
```  
#save the model
G=generator()
D=discriminator()

#set the loss function
criterion = nn.BCELoss()

# Setup Adam optimizers for both G and D
optimizerD = opt.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = opt.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

#fixed size of noise for plotting manually or testing 
fixed_noise = torch.randn(64, nz, 1, 1).to(device) 

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

#just for tracing the performance of model
G_losses = []
D_losses = []

# set model to GPU
D.cuda()
G.cuda()
```
  ### Set the weights
For applying weights, we can either apply for random weights (generated from normal distribution), or the weights saved from training process. 
  
```
#apply initialize weights
G.apply(weights_init)
D.apply(weights_init)

```
```
#apply saved weights
stateG_icon32x32=torch.load('/your path/file_name')
G.load_state_dict(stateG_icon32x32['state_dict'])
optimizerG.load_state_dict(stateG_icon32x32['optimizer'])

stateD_icon32x32=torch.load('/your path/file_name')
D.load_state_dict(stateD_icon32x32['state_dict'])
optimizerD.load_state_dict(stateD_icon32x32['optimizer'])
```
  ### main
  
  ```
  for epoch in range(num_epochs):

    #save the loss for showing stats
    running_lossG=[]
    running_lossD=[]
    # For each batch in the dataloader
    for i, data in enumerate(cluster_0_Loader, 0):

        '''
        ###########################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        '''
        
        '''
        ## Train with all-real batch
        '''      
        # Firstly, have to set grad to zero
        D.zero_grad()

        real_img = data.to(device) 
        # Set the batch size same as the image batch size we input every time
        #sometimes it would not be same as the setting at original e.g.64, 60000/64 will left 32)
        b_size = real_img.size(0)
        # Create the label for images from true dataset   
        one_label = torch.full((b_size,), real_label).to(device)  
        # Forward pass real image batch through D
        output = D(real_img).view(-1).to(device) 
        # Calculate loss on all-real batch
        errD_real = criterion(output, one_label)
        # Calculate gradients for D and backpropagate
        errD_real.backward()
        D_x = output.mean().item()
        
        '''
        ## Train D with all-fake batch
        '''                
        # Generate batch of latent vectors(100-vector)
        noise = torch.randn(b_size, nz, 1, 1).to(device) 
        # Generate fake image batch with G
        fake = G(noise)

        zero_label = torch.full((b_size,), fake_label).to(device) 
        # Input all fake image batch to D
        output = D(fake.detach()).view(-1).to(device) 
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, zero_label)
        # Calculate the gradients for this fake batch and backpropagate
        errD_fake.backward()
     
        ########### for statistics ############
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        #######################################
        
        # Update D
        optimizerD.step()
        '''
        ###########################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        '''
        G.zero_grad()

        # Since we just updated D, perform another forward pass of all-fake batch through D

        output = D(fake).view(-1)
        # Calculate G's loss based on this output, G should make D thinks this is real(one label)
        errG = criterion(output, one_label)
        # Calculate gradients for G and backpropagate
        errG.backward()
        
        ########### for statistics ############
        D_G_z2 = output.mean().item()
        #######################################
        
        # Update G
        optimizerG.step()
        
        
        #print the loss for G and D
        running_lossG.append(errG.item())
        running_lossD.append(errD.item())
        if i % 3450 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(cluster_0_Loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

    # Save Losses for plotting later
    if epoch % 1==0:
        G_losses.append(np.mean(errG))
        D_losses.append(np.mean(errD))
        
        
    #saving weights every 2 epochs.   
    if epoch % 2==0:
        #samples = fake.detach()
        #samples = samples.view(samples.size(0), 3, 32, 32)
        #showImages(samples)

        stateG_icon32x32 = {
            'epoch': epoch,
            'state_dict': G.state_dict(),
            'optimizer': optimizerG.state_dict(),
        }
        torch.save(stateG_icon32x32, '/your path/file_name')

        stateD_icon32x32 = {
            'epoch': epoch,
            'state_dict': D.state_dict(),
            'optimizer': optimizerD.state_dict(),
        }
        torch.save(stateD_icon32x32, '/your path/file_name')
  ```
## Autoencoder 

  ### Autoencoder

```
#number of feature maps for Autoencoder 
nf_auto = 4

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
            nn.Conv2d(nf_auto*2, nf_auto*4 , 4, 2, 1),  
            nn.ReLU(True),
            #output 4*4
        )
        
        self.decoder = nn.Sequential(
            #input is 4x4
            nn.ConvTranspose2d(nf_auto*4, nf_auto*2, 4, 2, 1),  
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
```

   ### setting 
   ```
#set parameters for autoencoder 
autoencoder = Autoencoder()
autoencoder.cuda()
criterion_auto = nn.MSELoss()
optimizer_auto = torch.optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))
   ```
### main
```
#train autoencoder 
for epoch in range(num_epochs):
    for i, data in enumerate(LLDLoader, 0):
              
        optimizer_auto.zero_grad()
        
        img = data.to(device)
        # get the result from the decoder
        output = autoencoder(img)[1]
        #calculae the loss between original image and the decoder
        loss_auto = criterion_auto(output, img)
        #backpropogation 
        loss_auto.backward()
        optimizer_auto.step()
```
## Clustering


  ### Memory issue on Colab
  Since the dataset is too large for Colab to encode at once, so I encode them seperately, then cat them. If there is no memory issue, can ignore this part. 
  ```
  KMLoader1 = torch.utils.data.DataLoader(icons32_2x, shuffle=False, batch_size=10000)
  KMLoader2 = torch.utils.data.DataLoader(icons32_4x, shuffle=False, batch_size=10000)
  ```
  
  ```
encoded = None
for i, data in enumerate(KMLoader1, 0): #40,000first, then 20,000
  dim_reducted = data.to(device)
  dim_reducted = autoencoder(dim_reducted)[0]  
  dim = dim_reducted.shape
  dim_reducted = dim_reducted.view(dim[0],dim[1]*dim[2]*dim[3])
  if i == 0:
    encoded = dim_reducted 
  else:
    encoded = torch.cat((encoded,dim_reducted),dim = 0)  
  ```
  
  ```
  torch.save(encoded, '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt')
  ```
  
  ```
  encoded_4x = torch.load( '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt')
  encoded = torch.cat((encoded,encoded_4x),dim = 0)
  ```
  

  
  ### KMeans cluster
I do KMeans clustering to the encoded images.

```
b = encoded.detach().cpu().numpy()
sse = {}
index = {}
for k in range(1, 8):
    kmeans = KMeans(n_clusters=k, max_iter=10, random_state = 4503).fit(b)
    #b["clusters"] = kmeans.labels_

    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    index[k] = kmeans.labels_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of clusters")
plt.ylabel("SSE")
plt.show()
```


## Comparisons of results using different parameters









