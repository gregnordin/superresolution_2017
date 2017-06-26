#Function to calculate high res image from low res input images. 
#Also has code to take a highres image and create offseted lowres images
#then run the super resolution algorithm on them (for testing).
#Last section is an example usage of running super resolution algorithm on 
#both input lowres images and generated lowres images.

#Note - use CalcOffset.py to generate the offsets needed by this script. 
#Paste them into the definition of the float_offsets variable in this file

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as scipy
import matplotlib.cm as cm
from matplotlib import pyplot as mp
import math
from random import randint
import random as rand
from mpl_toolkits.mplot3d import Axes3D
import cv2
from matplotlib import pyplot as plt

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*np.random.rand(n) + vmin



def define_pixel_relations(LRpix = 8, HRpix_per_LRpix = 5):
    """
    Return dict with low-res and hi-res image information

    LRpix - Size of low-res images in pixels, i.e., (LRpix x LRpix)
    HRpix - Size of hi-res images in pixels, i.e., (HRpix x HRpix)
    HRpix_per_LRpix - Number of hi-res pixels per low-res pixel,
                      i.e., one low-res pixel is
                      (HRpix_per_LRpix x HRpix_per_LRpix) hi-res pixels
    """
    return {"LRpix" : LRpix,
            "HRpix_per_LRpix" : HRpix_per_LRpix,
            "HRpix" : (LRpix  + 1) * HRpix_per_LRpix 
           }


#----------------------------------------------------------------------
# Image construction functions
#----------------------------------------------------------------------

'''
Shift image up and to the left by the given by shiftValueYX
Weights pixels for subpixel shifts
	imageWidth - width and height of image to shift
	imageToShift - the image that will be shifted
	shiftValueYX - high res pixels to shift image by

'''
def shift_image(imageWidth, imageToShift, shiftValueYX = (0,0)):

	imageHrShifted = np.zeros((imageWidth, imageWidth), dtype=float)

	fractionY = abs(shiftValueYX[0] - int(shiftValueYX[0]))
	fractionX = abs(shiftValueYX[1] - int(shiftValueYX[1]))

	bottomRightWeight = ((1 - fractionX) * (1 - fractionY))
	topRightWeight = ((fractionY) * (1 - fractionX))
	topLeftWeight = (fractionX * (fractionY))
	bottomLeftWeight = (fractionX * (1 -fractionY))

	for i in range(imageWidth):
		for j in range(imageWidth):

			#Bound Check
			if (i + 1 + int(shiftValueYX[0])) >= (imageWidth -1 ) or (j + 1 + int(shiftValueYX[1])) >= (imageWidth -1):
				continue

			valueBottomRight  = imageToShift[-1+i + 1 + int(shiftValueYX[0])][-1+j + 1 + int(shiftValueYX[1])] * bottomRightWeight
			valueTopRight = imageToShift[-1+i + int(shiftValueYX[0])][-1+j + 1 + int(shiftValueYX[1])] * topRightWeight
			valueTopLeft = imageToShift[-1+i + int(shiftValueYX[0])][-1+j + int(shiftValueYX[1])] * topLeftWeight
			valueBottomLeft  = imageToShift[-1+i + 1 + int(shiftValueYX[0])][-1+j + int(shiftValueYX[1])] * bottomLeftWeight

			imageHrShifted[i][j] = valueBottomRight + valueTopRight + valueTopLeft + valueBottomLeft
	
	return imageHrShifted

'''
Takes in a high res image and creates a low res image from it by averaging groups of
pixels. Also offsets the image
	pix - pixel relation object
	imageHR - High res image to downsample
	offset - Offset of output image
'''
def make_lowres_image(pix, imageHR, offset = (0,0)):
    imageLR = np.zeros((pix["LRpix"],pix["LRpix"]), dtype=float)
    # Set up x
    xHR_min = offset[0]
    xHR_max = pix["LRpix"]*pix["HRpix_per_LRpix"] + xHR_min
    xHR_interval = pix["HRpix_per_LRpix"]
    # Set up y
    yHR_min = offset[1]
    yHR_max = pix["LRpix"]*pix["HRpix_per_LRpix"] + yHR_min
    yHR_interval = pix["HRpix_per_LRpix"]
    # Loop over y (jj)
    for jj in range(int(round(yHR_min)), int(round(yHR_max)), int(round(yHR_interval))):
        jj_limit = min(jj+yHR_interval, pix["HRpix"])
        jLR = int((jj-yHR_min)/yHR_interval)
        # Loop over x (ii)
        for ii in range(int(round(xHR_min)), int(round(xHR_max)), int(round(xHR_interval))):
            ii_limit = min(ii+xHR_interval, pix["HRpix"])
            iLR = int((ii-xHR_min)/xHR_interval)
            imageLR[jLR,iLR] = np.average(imageHR[jj:jj_limit, ii:ii_limit])
    return imageLR
	
     
def make_highres_version_of_lowres(pix, lowres, offset, value=0):

    """
    Return a hi-res version of the input low-res image, `lowres`

    """
    highres = np.full((pix["HRpix"], pix["HRpix"]), value, dtype=float)
    for jj in range(pix["LRpix"]):
        jj_min = offset[1] + jj*pix["HRpix_per_LRpix"]
        jj_max = jj_min + pix["HRpix_per_LRpix"]
        for ii in range(pix["LRpix"]):
            ii_min = offset[0] + ii*pix["HRpix_per_LRpix"]
            ii_max = ii_min + pix["HRpix_per_LRpix"]
            highres[jj_min:jj_max, ii_min:ii_max] = lowres[jj,ii]
    return highres

def add_new_elements_to_dict(t):
    t["iter_lowres"] =  t["lowres"] 
    t["iter_sum_diff_highres"] = [ None ]
    t["estimated_image"] = t["guess"]
    t["IsConverged"] = False

def initialize_prior_to_making_estimates(t):
    t["iter_count"] = 0

def initialize_c(t, value):
    t["c"] = float(value)
    
def invert_image(img):
	invertedImage = np.copy(img)
	for i in range(len(img[0])): 
		for j in range(len(img[0])):
			invertedImage[i][j] = 255.0 - img[i][j]	
	return invertedImage

def isImageUnstable():
	if t["iter_count"] == 0:
		return False
	t["percentImageChangeIter"].append

def make_new_estimated_image(t, clip_range=False):

	print "Begin iteration ", t["iter_count"]
	t["iter_lowres"] = []

	# Make low res offsetted versions of estimated image
	t["iter_lowres"] = [make_lowres_image(t["pix"], t["estimated_image"], offset = o) for o in t["offsets"]] 

	# Diff the low res offsetted images from above with the original low res images
	temp_diff = []
	for a,b in zip(t["lowres"],t["iter_lowres"]):

		temp_diff.append(a-b)

	# Make temp_diff_highres
	temp_diff_highres = [make_highres_version_of_lowres(t["pix"], lowres, offset, value=0.0)
		         for lowres, offset in zip(temp_diff, t["offsets"])]

	# Make sum of differences
	temp_sum = np.copy(temp_diff_highres[0])
	for index in range(1,len(temp_diff_highres)):	
		temp_sum += temp_diff_highres[index]

	#Get absolute value of total value of temp_sum to check for over convergence
	totalPixelChange = sum(sum(abs(temp_sum/t["c"])))

	t["totalPixelChangeIter"].append(totalPixelChange)
	#Check if the image is converged based on percent change in diff image
	if t["iter_count"] != 0:
		prevPixelChange = t["totalPixelChangeIter"][t["iter_count"]-1]
		percentChange = 100 * (prevPixelChange - totalPixelChange) / totalPixelChange
		t["percentChangeIter"].append(percentChange)
		print "Percent Change is ", percentChange
		if percentChange > t["percentChangeIter"][max(t["iter_count"] - 2, 0) ]:
			#Image is converged
			print("Image has converged")
			t["imageHasConverged"] = True
		if percentChange < 0:
			print("Image is Unstable!")
			t["unstableImageDetected"] = True

	# Make new estimated image
	t["estimated_image"] = t["estimated_image"] + temp_sum / t["c"]
	if clip_range:
		np.clip(t["estimated_image"], 0, 255, out=t["estimated_image"])

	t["iter_count"] += 1


'''
Function for creating a dataset to run super resolution algorithm on
generated data (takes a high res image and creates low res images of it
	pix - pixel relation object
	offsets - List of tuples of offsets of the low res images
	originalImage - original highres image
'''

def make_test_image_dataset(pix, offsets, originalImage):
    lowres_images = [make_lowres_image(pix, originalImage, offset = o) for o in offsets]
    highres_images = [make_highres_version_of_lowres(pix, lowres, offset)
                      for lowres, offset in zip(lowres_images, offsets)]
    # Make average image of `highres_images`
    guess_image = np.zeros((originalImage.shape[0], originalImage.shape[1]), dtype=float)
    for image in highres_images:
        temp = np.copy(image)
        temp[np.isnan(temp)] = 0
        guess_image += temp
    guess_image /= len(highres_images)

    # End make average image
    return {"pix" : pix,
            "offsets" : offsets,
            "original" : originalImage,
            "lowres" : lowres_images,
            "highres" : highres_images,
            "guess" : guess_image,
            "totalPixelChangeIter" : [],
            "percentChangeIter" : [],
            "unstableImageDetected" : False,
            "imageHasConverged" : False,
           }

######## End test data functions #######

'''
Function for running creating a dataset to run super resolution algorithm on
real data (you provide the low res images)
	pix - pixel relation object
	offsets - List of tuples of offsets of the low res images
	lowres_images - list of lowres images
'''
def make_real_image_dataset(pix, offsets, lowres_images):
	highres_images = [make_highres_version_of_lowres(pix, lowres, offset)
		      for lowres, offset in zip(lowres_images, offsets)]

	# Make average image of `highres_images`
	guess_image = np.zeros((pix["HRpix"], pix["HRpix"]), dtype=float)
	for image in highres_images:
		temp = np.copy(image)
		temp[np.isnan(temp)] = 0
		guess_image += temp
	guess_image /= len(highres_images)
	# End make average image
	return {"pix" : pix,
		"offsets" : offsets,
		"lowres" : lowres_images,
		"highres" : highres_images,
		"guess" : guess_image,
		"totalPixelChangeIter" : [],
		"percentChangeIter" : [],
		"unstableImageDetected" : False,
		"imageHasConverged" : False,
		}


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


####################################################################
#Begin Example Usage
####################################################################

#C value determins how quickly the estimated (output) image converges.
CValue = 3000
iterations = 1
#Set to True to run algorithm on generated data
testRun = False
#Set to True to run algorithm on real data
realDataRun = True
#Set to True to save 3D Line scans of each iter estimate image
save3DScan = False

dlp_pix = define_pixel_relations(LRpix = 10, HRpix_per_LRpix = 10)

#These offset values were generated with CalcOffset.py
float_offsets = [
			(5.35233868214,8.03721406623),
			(5.76859504132,4.2916869227),
			(4.19322555812,8.13394919169),
			(3.99064391855,1.14285714286),
			(7.59568733154,4.44474393531),
			(3.76760620583,6.96157907562),
			(6.74421090132,7.7976487353),
			(0.383387378399,2.69044649539),
			(5.29444589764,5.94132008022),
			(5.82093394886,9.88245738636),
			(4.3323782235,5.76217765043),
			(6.55786014991,9.22024848547),
			(6.64350480808,8.49386555679),
			(7.55075554749,5.82738122249),
			(6.50303951368,4.8358662614),
			(4.3763359421,2.32071025835),
			(6.42136475053,3.92883687831),
			(3.89060308555,5.03085553997),
			(6.32764088802,5.24420590388),
			(2.84732824427,8.3320610687),
			(9.27903405383,4.79471724191),
			(4.69491525424,7.97919876733),
			(3.30761343245,8.07715970264),
			(2.51650894665,1.58915657672),
			(3.41873449132,10.5097704715),
			(7.31996571635,7.06883796463),
			(3.94424377142,2.59729554506),
			(1.196434813,4.0699056274),
			(5.63081658519,6.18269056672),
			(3.58881256133,7.0569185476),
			(1.30004438526,5.10075454949),
			(2.84886015366,6.07985221882),
			(7.31565326894,9.14366255762),
			(8.17991631799,6.57740585774),
			(0.596781477072,3.48273738659),
			(8.61912545975,5.65508786269),
			(5.45030425963,3.96348884381),
			(2.14274631497,2.93328161365),
			(3.64669843431,0.43294758339),
			(0.608283874227,4.15153428463),
			(1.07265388496,3.92431886983),
			(4.6763229046,4.0),
			(1.66395812173,0.168450660208),
			(3.03713790542,3.96286209458),
			(5.93117408907,7.5951417004),
			(2.06120776366,0.323808814216),
			(7.84423676012,3.61059190031),
			(4.42429739216,6.89476783905),
			(7.69455631477,6.32770713607),
			(2.41827583421,7.13242304274),
			(1.10768958499,2.14750906999),
			(7.59260217279,2.21520951888),
			(7.62266942883,8.13435927789),
			(8.09041415516,7.51991856435),
			(6.51774405544,2.42275062732),
			(4.18713300996,9.0),
			(5.87744227353,2.82770870337),
			(6.11392405063,6.81772151899),
	]


#List of ideal offsets that can be used for testing
test_offsets = []
for i in range(10):
	for j in range(10):
		test_offsets.append((i,j))

#Same as float_offsets but rounded to nearest whole number
round_offsets = []
for i in range(len(float_offsets)):
	round_offsets.append((int(round(float_offsets[i][0])),int(round(float_offsets[i][1]))))



################### Generated Data Test Setup #################################################
if testRun:
	testImage = np.zeros((300, 300), dtype=float)

	#Create a high res image to serve as original
	g = gaussian(np.linspace(-3, 3, 32), 0, 1)
	h = np.zeros((64), dtype=float)
	h[0:32] = 3
	h[17:32] = 3
	h[32:64] = 3
	h[17:27] = [7,40,100,200,400,600,400,100,20,7]

	#Pixels between generated the curves
	spaceBetween = 22
	#Space between generated curves and edge of image
	BoarderSize = 32

	#Multiple h by h and place the product in the high res image to be used as an original image for testing
	for i in range(128):
		for j in range(128):
			if (i > BoarderSize and i < (64 +BoarderSize) and  (j > BoarderSize) and j < (64 + BoarderSize)):	
				testImage[i][j] =  testImage[i][j] + h[i - BoarderSize] * h[j - BoarderSize]
			if (i > (BoarderSize + spaceBetween) and i < (64 + spaceBetween +BoarderSize) and  (j > BoarderSize) and j < (64+BoarderSize)):
				testImage[i][j] =  testImage[i][j] + h[i - BoarderSize - spaceBetween] * h[j - BoarderSize]
			if (i > BoarderSize and i < (64+BoarderSize) and  (j > BoarderSize + spaceBetween) and j < (64 + spaceBetween + BoarderSize)):	
				testImage[i][j] =  testImage[i][j] + h[i - BoarderSize] * h[j - BoarderSize - spaceBetween]
			if (i > (BoarderSize + spaceBetween) and i < (64 + spaceBetween + BoarderSize) and  (j > BoarderSize + spaceBetween) and j < (64 + spaceBetween +BoarderSize)):	
				testImage[i][j] =  testImage[i][j] + h[i - BoarderSize - spaceBetween] * h[j - BoarderSize - spaceBetween]
	
	#Normalize test image
	testImage = (testImage/testImage.max()) * 255

	#Crop the test image
	croppedTestImage = np.zeros((110, 110), dtype=float)
	for i in range(110):
		for j in range(110):
			croppedTestImage[i][j] = testImage[i + 10][j + 10]
	testImage = croppedTestImage

	#Generate Test Image set
	testDataSet = make_test_image_dataset(dlp_pix, test_offsets, testImage)

	add_new_elements_to_dict(testDataSet)
	initialize_prior_to_making_estimates(testDataSet)
	initialize_c(testDataSet, CValue)

	#Code to generate line scan for each iteration
	lines = []
	line_temp = np.array(testImage[[45]])
	line_temp = line_temp[0]
	lines.append(line_temp)

	print(line_temp)

	line_temp = np.array(testDataSet["guess"][[45]])
	line_temp = line_temp[0]
	lines.append(line_temp)

	#Folder to save line scan images in
	lineScanFoldername  = "NewLineScan/"

	#Show the original image
	plt.figure()
	plt.title("Original Generated Image")
	plt.imshow(testImage, cmap = cm.Greys_r,interpolation="nearest", extent=[0, 110, 110, 0])

	plt.show()
	f = np.fft.fft2(testImage)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = (20*np.log(np.abs(fshift)))
	
	#Save 3D Line Scan
	if save3DScan:
		np.savetxt( "New3DLineScan/guess", testDataSet["guess"])
		np.savetxt( "New3DLineScan/original", testImage)

	for index in range(iterations):
		make_new_estimated_image(testDataSet, clip_range=False)

		if testDataSet["unstableImageDetected"] or testDataSet["imageHasConverged"]:
			print "Image seems unstable"

		line_temp = np.array(testDataSet["estimated_image"][[45]])
		line_temp = line_temp[0]

		fig = plt.figure()
		plt.title("Red = Original Blue = guess Green = " + str(index) + "th iteration" + " CValue = " + str( CValue))
		plt.ylabel('Value')
		plt.xlabel('X Location')
		plt.plot(lines[0], 'r-', lines[1], 'b-', line_temp, 'g-')
		fig.savefig(lineScanFoldername + "{0:03}".format(index) + '.png')
		plt.close(fig)

		if save3DScan:
			np.savetxt( "New3DLineScan/" + str(index), testDataSet["estimated_image"])

	plt.figure()
	plt.title("Guess Generated Image")
	plt.imshow(testDataSet["guess"], cmap = cm.Greys_r,interpolation="nearest", extent=[0, 110, 0, 110])

	plt.figure()
	plt.title("Estimated image from Generated Image after " + str(testDataSet["iter_count"]) + " iterations")
	plt.imshow(testDataSet["estimated_image"], cmap = cm.Greys_r,interpolation="nearest")
	plt.show()


############# Real Data Setup and Test ####################################################
if realDataRun:
	#Code for plotting all the offsets on one cluster graph 
	#plt.figure()
	#plt.title("Offsets Used")
	#xs = [x[0] for x in float_offsets]
	#ys = [x[1] for x in float_offsets]
	#plt.scatter(xs, ys)
	#plt.show()

	folderName = "LowResImages/"
	lowres_images = []

	#Load Lowres images from disk
	for i in range(len(round_offsets)):
		lowres_images.append(np.loadtxt(folderName + str(i)))

	DLPDataSet = make_real_image_dataset(dlp_pix, round_offsets, lowres_images)

	#Show initial Guess
	plt.figure()
	plt.title("Initial Guess")
	plt.imshow(DLPDataSet["guess"], cmap = cm.Greys_r,interpolation="nearest")
	plt.show()

	f = np.fft.fft2(DLPDataSet["guess"])
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = (20*np.log(np.abs(fshift)))

	#Show frequency domain of guess image
	print(magnitude_spectrum)
	plt.subplot(121),plt.imshow(DLPDataSet["guess"], cmap = cm.Greys_r,interpolation="nearest", extent=[0, 110, 110, 0])
	plt.title('Initial Guess Realdata ')
	plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = cm.Greys_r,interpolation="nearest", extent=[0, 110, 110, 0])
	plt.title('Magnitude Spectrum')
	plt.show()

	# Initialize dict with test case
	add_new_elements_to_dict(DLPDataSet)
	initialize_prior_to_making_estimates(DLPDataSet)
	initialize_c(DLPDataSet, CValue)

	#Line scan code
	lines = []
	line_temp = np.array(DLPDataSet["guess"][[50]])
	line_temp = line_temp[0]
	lines.append(line_temp)
	lineScanFoldername  = "NewLineScan/"
	i = 0
	# Iterate to calculate super-resolution image
	for index in range(iterations):
		#Fun super resolution algorithm
		make_new_estimated_image(DLPDataSet, clip_range=False)
		scipy.imsave('../../../../ConvergingImages/' + "{0:03}".format(DLPDataSet["iter_count"]) + '.png', DLPDataSet["estimated_image"])
		i += 1

		line_temp = np.array(DLPDataSet["estimated_image"][[50]])
		line_temp = line_temp[0]
		lines.append(line_temp)

		fig = plt.figure()
		plt.title("Blue = guess Green = " + str(index) + "th iteration" + " CValue = " + str( CValue))
		plt.ylabel('Value')
		plt.xlabel('X Location')
		plt.plot(lines[0], 'b-', line_temp, 'g-')
		fig.savefig(lineScanFoldername + "{0:03}".format(index) + '.png')
		plt.close(fig)

		if DLPDataSet["unstableImageDetected"] or DLPDataSet["imageHasConverged"]:
			print "Image seems to be unstable, try a higher cValue"
			#break

	#Show estimated image 
	plt.figure()
	plt.title("Estimated image after " + str(DLPDataSet["iter_count"]) + " iterations")
	plt.imshow(DLPDataSet["estimated_image"], cmap = cm.Greys_r,interpolation="nearest")

	plt.figure()
	plt.title("Sample Low Res")
	plt.imshow(DLPDataSet["lowres"][0], cmap = cm.Greys_r, interpolation="nearest")

	plt.show()











1
