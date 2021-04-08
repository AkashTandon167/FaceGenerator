from os import listdir
from os.path import isfile, join, dirname
from PIL import Image #pillow party!!!
from PIL import ImageOps
import numpy as np
import random

imageSize = 64 #height and width of images

def normalize(array):
	min = np.amin(array)
	max = np.amax(array)
	array = array - min
	return array / (max - min) #rescales all elements in array to range [0,1]

def toImage(arr,imageSize,mode):
	if(mode == 'gray'):
		arr = np.array(normalize(arr)*255).reshape(imageSize,imageSize).astype(np.uint8)
		return Image.fromarray(arr).resize((512,512))
	elif(mode == 'rgb'):
		arr = np.array(normalize(arr)*255).reshape(imageSize,imageSize,3).astype(np.uint8)
		return Image.fromarray(arr).resize((512,512))
	else:
		return None
	
def toArray(image):
	return np.array(image).reshape(-1,1)
	
def getImage(index,path,images,imageSize,mode):
	
	img = Image.open(path + '\\' + str(images[index])).resize((128,128))
	width,height = img.size
	img = img.crop((20,40,width-20,height)).resize((imageSize,imageSize))
	#img = img.crop((28,28,width-28,height-28)).resize((imageSize,imageSize))
	if(mode == 'gray'):
		return img.resize((imageSize,imageSize)).convert('L')
	elif(mode == 'rgb'):
		return img.resize((imageSize,imageSize)).convert('RGB')
	else:
		return None
	
def crop(image,height_factor,width_factor): #crops image by two factors 0-1 (0.5 is half) and resizes it
	width,height = image.size
	height_margin = height*height_factor/2
	width_margin = width*width_factor/2
	return image.crop((int(width*width_margin),int(height*height_margin),
					   int(width*(1-width_margin)),int(height*(1-height_margin)))).resize((height,width))

crop = 3

def crop_variants(img,imageSize): #returns tuple of slightly cropped versions of an image
	width, height = img.size
	return (img.crop((crop,crop,width,height)).resize((imageSize,imageSize)),
			img.crop((0,0,width-crop,height-crop)).resize((imageSize,imageSize)),
			img.crop((0,crop,width-crop,height)).resize((imageSize,imageSize)),
			img.crop((crop,0,width,height-crop)).resize((imageSize,imageSize)))

		
def getImages(path,imageSize,mode,numImgs):
	images = [f for f in listdir(path) if isfile(join(path, f))] #list of files
		
	print('Images Loaded')

	imgs = [getImage(i,path,images,imageSize,mode) for i in range(min(len(images),numImgs))]
	#imgs += [ImageOps.mirror(i) for i in imgs]
	#imgs += [j for i in imgs for j in crop_variants(i,imageSize)]
	random.shuffle(imgs)
	#imgs[0].show()
	
	return np.array([toArray(i) for i in imgs])