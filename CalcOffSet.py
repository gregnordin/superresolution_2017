#This script reads in all images in the folder pointed to by
#'InputImagesFileLocation', crops them, and writes a list of their offsets
#to a file specified by "OffsetsOutputFile" variable in the same directly 
#as this script The cropped images are stored in the LowResImages/ directory
#in the directory of this script. This script generates the input needed by
#CalcHighRes.py

#Output - Low res images and offsets.txt file formated as a tuple list of offsets
#that can be pasted into CalcHighRes.py (see notes at beginning of CalcHighRes.py)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as scipy
import matplotlib.cm as cm
import glob
from scipy import ndimage
from random import randint

InputImagesFileLocation = '../../../../srprojf16/Fall 2016/Images_12.6.2016/*.tif'
OffsetsOutputFile = "offsets.txt"

image_list = []

target = open(OffsetsOutputFile, 'w')

i = 0;
for filename in glob.glob(InputImagesFileLocation): 
	i += 1
	
	testImage = scipy.imread(filename, 'L')	
	print(filename)
	image_list.append(testImage)

#We keep a index (which is the sequential index of the image being read)
index = 0

print("Done Loading Images")
croppedImage = np.zeros((30, 30), dtype=float)

yMinList = []
xMinList = []
offsets = []

histo = []

#For each lowres image that was read in
for testImage in image_list:
	originalImage = np.copy(testImage)
	#Crop imageto 30x30	
	for i in range(30):
		for j in range(30):
			croppedImage[i][j] = testImage[i + 110][j + 1230]

	testImage = croppedImage;
	
	totalBlackSpaceValue = 0

	#Find Average Black Space Value
	for i in range(0,3):
		for j in range(0,30):	
			totalBlackSpaceValue += testImage[i][j]
	totalBlackSpaceValue = totalBlackSpaceValue / 90

	imageToReconstruct = np.copy(testImage)

	#Subtract Blackspace value 
	testImage = testImage - totalBlackSpaceValue

	#Zero out remaining black space
	for i in range(testImage.shape[0]):
		for j in range(testImage.shape[1]):
			if testImage[i][j] < 17:
				testImage[i][j] = 0
	
	#Normalize Image
	testImage *= 255.0 / (testImage.max())

	#testImage is the image used to find the offsets while imageToReconstruct is the image that will be reconstructed
	#For this test, we are reconstructing the same part of the image that we used to generate the offsets
	imageToReconstruct *= 10.0 / (imageToReconstruct.max())

	#Crop the images again to 10x10
	secondCroppedImage = np.zeros((10, 10), dtype=float)

	imageToReconstructCropped = np.zeros((10, 10), dtype=float)
	
	#Crop both the image to be used to calculate the offsets and the image to reconstruct
	for i in range(10):
		for j in range(10):
			secondCroppedImage[i][j] = testImage[i + 8][j + 13]
			imageToReconstructCropped[i][j] = imageToReconstruct[i + 8][j + 13]
		

	testImage = secondCroppedImage 
	imageToReconstruct = imageToReconstructCropped

	#Find Sum of all pixels
	totalSoFar = 0
	for i in range(testImage.shape[0]):
		for j in range(testImage.shape[1]):
			totalSoFar += imageToReconstruct[i][j]

	#Calc Center of Mass
	sumX = 0
	sumY = 0
	bigSum = 0

	for i in range(testImage.shape[0]):
		for j in range(testImage.shape[1]):
			bigSum += testImage[i][j]
			sumY += testImage[i][j] * i
			sumX += testImage[i][j] * j

	yPoint = sumY/bigSum
	xPoint = sumX/bigSum

	#160 and 110 are the lowest y and x values of any offset
	yPoint = ((yPoint * 10))
	xPoint = ((xPoint * 10))
	
	yMinList.append(yPoint)
	xMinList.append(xPoint)

	#Adjust images that are more then 10 high res pixels off
	#For example, an offset of (45, 32) would translate to (4, 2) with the 
	#image being shifted up by 4 and left by 3 low-res pixels

	yPointInt = ((yPoint) - 37)
	xPointInt = ((xPoint - 36))

	#Check if we already have an image for this offset, if so, then skip this image
	offsetAlreadyUsed = False
	for i in range(len(offsets)):
		if int(offsets[i][0]) == int(yPointInt) and int(offsets[i][1]) == int(xPointInt):
			offsetAlreadyUsed = True
			break

	areaToReconstruct = np.zeros((10, 10), dtype=float)
 	for i in range(10):
		for j in range(10):
			areaToReconstruct[i][j] = originalImage[i + 600][j + 910]

	areaToReconstruct = imageToReconstruct

	#Write Offsets to offsets.txt
	target.write("(")
	target.write(str(yPointInt))
	target.write(",")
	target.write(str(xPointInt))
	target.write("),")
	target.write("\n")

	offsets.append((yPointInt ,xPointInt))

	#Write low res image
	folderName = "LowResImages/"
	np.savetxt( folderName + str(index), areaToReconstruct)
	index += 1

	#Code below will show each image with it's calculated center of mass marked
	#plt.figure()
	#plt.axhline(y= (yPoint) / 10.0 )
	#plt.axvline(x= (xPoint)/ 10.0 )
	#plt.imshow(testImage, cmap = cm.Greys_r,interpolation="nearest")
	#plt.show()

print 'Y min is ', min(yMinList), ' X min is ', min(xMinList)
target.close()

plt.figure()
plt.title("Total Image Value")
plt.xlabel('Value Bins')
plt.ylabel('Quantity')
plt.hist(histo, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,12,13,14,15, 16, 17, 18, 19, 20, 21, 22])
print("Done Calculating offsets and Cropping all Images")
plt.show()
exit()

		





