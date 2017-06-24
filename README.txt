Jun 2017

This repo contains 2 files:
CalcOffsets.py - Does two things (1) Calculates the center of mass of each low res image it is given and (2) crops and normalized each low res image.

CalcHighRes.py - Can run super resolution algorithm on given lowres images and/or run super resolution algorithm on a given highres image (which is offsets and downsamples multiple times to create low res image). 

SuperResolutionWritupAndSummary.html - Summary of the findings of running this code on a set of lowres images of a DLP array projection for a 3D printer. 

Instructions:
Start with CalcOffsets.py. You will need a set of lowres images which you will feed into CalcOffsets.py (see comments in file for details). CalcOffsets.py will find the center of mass of each image and return a formated tuple list of offsets as well as cropped and normalized copies of each of the low res images in the format needed by CalcHighRes.py. Copy the offsets into CalcHighRes.py (see notes in CalcHighRes.py for details) and ru CalcHighRes.py after succesfully running CalcOffsets.py. 

Notes: Use anaconda if you have problems with missing libraries. Avaliable here: https://www.continuum.io/downloads
