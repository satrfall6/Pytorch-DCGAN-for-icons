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

torch.cuda.empty_cache()

icons32_2x=torch.load( '/content/drive/My Drive/Colab Notebooks/LLD_32/icon_sharp.pt')
icons32_4x=torch.load( '/content/drive/My Drive/Colab Notebooks/LLD_32/icon_32.pt')

KMLoader1 = torch.utils.data.DataLoader(icons32_2x, shuffle=False, batch_size=10000)
KMLoader2 = torch.utils.data.DataLoader(icons32_4x, shuffle=False, batch_size=10000)

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

torch.save(encoded, '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt')

encoded_4x = torch.load( '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt')

encoded = torch.cat((encoded,encoded_4x),dim = 0) #but here, I cat 20,000 first then 40,000
encoded.shape

torch.save(encoded, '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt') # so the file here is 200000+40000
icons32_2x=None
icons32_4x = None

"""################above is to deal with cuda memory issue##########################"""

#2x first, then 4x
encoded = torch.load( '/content/drive/My Drive/Colab Notebooks/LLD_32/icon32_KMeans.pt')
encoded.shape

b = encoded.detach().cpu().numpy()
b.shape

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

np.save('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster0_index',np.where(index[4]==0)[0])
np.save('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster1_index',np.where(index[4]==1)[0])
np.save('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster2_index',np.where(index[4]==2)[0])
np.save('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster3_index',np.where(index[4]==3)[0])

"""##train each cluster seperately"""

cluster0_idx = np.load('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster0_index.npy')
cluster1_idx = np.load('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster1_index.npy')
cluster2_idx = np.load('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster2_index.npy')
cluster3_idx = np.load('/content/drive/My Drive/Colab Notebooks/LLD_32/cluster3_index.npy')

cluster_0 = icon_combined[cluster0_idx]
cluster_1 = icon_combined[cluster1_idx]
cluster_2 = icon_combined[cluster2_idx]
cluster_3 = icon_combined[cluster3_idx]

size_of_clusters = [len(cluster0_idx),len(cluster1_idx),len(cluster2_idx),len(cluster3_idx)]
plt.plot(size_of_clusters)
plt.xlabel("Each cluster ")
plt.ylabel("Number of images")
plt.title("Number of images in each cluster")

cluster_0_Loader = torch.utils.data.DataLoader(cluster_0, shuffle=True, batch_size=batch_size)
cluster_1_Loader = torch.utils.data.DataLoader(cluster_1, shuffle=True, batch_size=batch_size)
cluster_2_Loader = torch.utils.data.DataLoader(cluster_2, shuffle=True, batch_size=batch_size)
cluster_3_Loader = torch.utils.data.DataLoader(cluster_3, shuffle=True, batch_size=batch_size)

"""### ================= above is Kmeans part =================