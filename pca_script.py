# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 22:20:26 2016

@author: SYARLAG1
"""

import cv2
import os
import numpy as np
import pandas as pd
import dicom
from PIL import Image

####################FUNCTIONS THAT WILL BE USED###################################################
def readDicom(fileLoc):
    return unrollMatrix(M)


def unrollMatrix(M):
    unrollM = np.zeros(shape=[1,M.shape[0]*M.shape[1]])
    startIndex = 0; endIndex = M.shape[1]
    for i in range(M.shape[0]):
        unrollM[0,startIndex:endIndex] = M[i]
        startIndex = endIndex
        endIndex =  endIndex + M.shape[1]
    return unrollM

from sklearn.preprocessing import StandardScaler

def reduceImageSizePCA(imageMatrix, varAccount = 0.9):
    normalizedMatrix = StandardScaler().fit_transform(imageMatrix)
    global covMatrix
    covMatrix = np.cov(normalizedMatrix.T)
    global U,S,V
    U,S,V = np.linalg.svd(covMatrix)
    diagonalElements = S
    countCols = 0
    for entry in diagonalElements:
        varAccounted = entry/S.sum()
        if varAccounted > varAccount:
            break
        countCols += 1
    return U[:,:countCols].T * normalizedMatrix

def padImage(imageMat, maxRows, maxCols):
    padImageMat = np.zeros(shape=[maxRows,maxCols])
    imageMatShape = imageMat.shape
    rowDiff = maxRows - imageMatShape[0]
    colDiff = maxCols - imageMatShape[1]

    rowOffSet = rowDiff//2 
    colOffSet = colDiff//2
    padImageMat[rowOffSet:imageMatShape[0]+rowOffSet,colOffSet:imageMatShape[1]+colOffSet] = imageMat
    
    return padImageMat           
            
            
            
##################SCRIPT FOR PCA################################################################
#rootdir = 'C:/Users/SYARLAG1/Desktop/resized_images-32x32'
#
#fullImageMat = np.zeros(shape=[1,1024])
#        
#for folder in os.listdir(rootdir):
#    newPathSubFolder1 = os.path.join(rootdir,folder)
#    for subFolder_level1 in os.listdir(newPathSubFolder1):
#        newPathsubfolder2 = os.path.join(newPathSubFolder1,subFolder_level1)
#        for subFolder_level2 in os.listdir(newPathsubfolder2):
#            newPathImageFolder = os.path.join(newPathsubfolder2, subFolder_level2)
#            os.chdir(newPathImageFolder)            
#            for imageFile in os.listdir(newPathImageFolder):
#                 imageArray = unrollMatrix(cv2.imread(imageFile,0))
#                 fullImageMat = np.vstack((fullImageMat,imageArray))
#
#fullImageMat = np.delete(fullImageMat, (0), axis=0) 
#
#os.chdir('C:/Users/SYARLAG1/Desktop/Lung-Nodule-Identification')    
#np.savetxt('./fullImageMatrix.csv',fullImageMat, delimiter=',')
#
#fullImageMat = np.genfromtxt('./fullImageMatrix.csv',delimiter=',')
#
#redImageMat = reduceImageSizePCA(fullImageMat, varAccount = 0.9) #applying PCA to this matrix
#
#
#os.chdir('C:\\Users\\SYARLAG1\\Desktop\\DePaul_Medix_Lab\\LIDC\\image-based_nodule_classification\\data\\LIDC\\resized_images-32x32\\LIDC-IDRI-0365\\1.3.6.1.4.1.14519.5.2.1.6279.6001.216207548522622026268886920069\\1.3.6.1.4.1.14519.5.2.1.6279.6001.802846969823720586279982179144')
#os.listdir('.\\')
#sampleImage = 'C:/Users/SYARLAG1/Desktop/resized_images-32x32\LIDC-IDRI-0001/1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178/1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192/000030-roi_30-rs_3-crop.tiff.32x32.tiff'
    
###############Script for PCA on DICOM files#####################################################
os.chdir('C:/Users/SYARLAG1/Desktop/Lung-Nodule-Identification')    

varLst = ['InstanceID','Labels']
instanceIDsLabels = pd.read_csv('./LIDC_REU2015.csv', usecols=varLst) 

# Reading in all the Dicom images
imgDict = {}
imgSizeDict = {}

for fileName in os.listdir('./nodules'):
    instanceID = int(fileName[:-4])
    if instanceID not in list(instanceIDsLabels.InstanceID): continue # we only pick images that are part of the 809
    fileLoc = './nodules/'+fileName 
    dicomRead =   dicom.read_file(fileLoc)  
    dicomMat = dicomRead.pixel_array
    imgDict[instanceID] = dicomMat
    imgSizeDict[instanceID] = dicomMat.shape

# Finding max row and max col
maxRow = 0 
maxCol = 0
for shape in imgSizeDict.values():
    if shape[0] > maxRow: maxRow = shape[0] #51
    if shape[1] > maxCol: maxCol = shape[1] #58


padImgDict = {}
for key in imgDict.keys():
    padImgDict[key] = padImage(imgDict[key], maxRow, maxCol)

# unrolling matrices and creating a single matrix
imgSeqLst = padImgDict.keys()
fullImageMat = np.zeros(shape=[1,maxRow*maxCol])

for imgMat in padImgDict.values():
    imgArr = unrollMatrix(imgMat)
    fullImageMat = np.vstack((fullImageMat, imgArr))
fullImageMat = fullImageMat[1:,:] # removing first row (was used to initialize)

np.savetxt('./fullImageMatrix.csv', fullImageMat, delimiter=',')

# Applying PCA on the new unrolled vectors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

normalizedMat = StandardScaler().fit_transform(fullImageMat)
pca = PCA(n_components=100)
redImgMat = pca.fit(normalizedMat)

redImgMat.explained_variance_ratio_[:10] 
## array([ 0.26953955,  0.10198192,  0.08870162,  0.06421865,  0.05518049, 
## 0.04118215,  0.03424778,  0.03166701,  0.02642079,  0.02131871])

# Visulizing the first 4 components
featureMeans = fullImageMat.mean(axis=0)

for index, arr in enumerate(redImgMat.components_[:4]):
    reshapedEigenImg = np.reshape(255*(arr - arr.min())/(arr.max()-arr.min()), newshape=(maxRow, maxCol))
    img = Image.fromarray(reshapedEigenImg)
    #img.save('PC%s.png'%(index+1))
    img.show()

## Visualizing the PCs

#temp = imgDict.values()[1]
#tempPad = padImage(temp, maxRow, maxCol)
#for i in range(unrollM.shape[1]): 
#    if unrollM[0,i]>0: 
#        print unrollM[0,i]
#############



















    
