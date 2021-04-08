import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np
import math
import imageEdit
from os.path import dirname, join
import time

imageSize = 64

main_path = 'C:\\Users\\akash\Documents\Programming\Machine Learning\Face Database' 

path = main_path + '\\celebA\\processed_celeba_small\\celeba' #directory w/ the images

MAX_SIZE = 9688

train_images = imageEdit.getImages(path,imageSize,'gray',MAX_SIZE)

train_images = train_images.reshape(train_images.shape[0], imageSize, imageSize, 1).astype('float16')
train_images = ((train_images - 127.5) / 127.5).astype('float16') 
# Normalize the images to [-1, 1]

print('Images Normalized')

vectorSize = 128

pca = PCA(n_components=vectorSize)
pca.fit(train_images.reshape(train_images.shape[0], -1).astype('float16'))

pca_variance = pca.explained_variance_ratio_
cumulative_variance = np.array([sum(pca_variance[:i+1]) for i in range(len(pca_variance))])
print(cumulative_variance)

print('Principal Component Analysis Complete')

variance = 100 #range from [-variance,variance]

for x in range(10):
	vector = (np.random.randn((vectorSize))*2*variance - variance)*pca_variance
	image = imageEdit.toImage(imageEdit.normalize(pca.inverse_transform(vector)),imageSize,'gray')
	#image = image.point(lambda i : (i ** 0.5) * 16)
	image.save('rand%d.png'%(x),format="PNG")