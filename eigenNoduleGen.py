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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

####################FUNCTIONSD#################################################
# Unrolls a matrix into a single vector
def unrollMatrix(M):
    unrollM = np.zeros(shape=[1,M.shape[0]*M.shape[1]])
    startIndex = 0; endIndex = M.shape[1]
    for i in range(M.shape[0]):
        unrollM[0,startIndex:endIndex] = M[i]
        startIndex = endIndex
        endIndex =  endIndex + M.shape[1]
    return unrollM

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

# takes unpadded image and outputs a padded image based on the large col and row size
def padImage(imageMat, maxRows, maxCols):
    padImageMat = np.zeros(shape=[maxRows,maxCols])
    imageMatShape = imageMat.shape
    rowDiff = maxRows - imageMatShape[0]
    colDiff = maxCols - imageMatShape[1]

    rowOffSet = rowDiff//2 
    colOffSet = colDiff//2
    padImageMat[rowOffSet:imageMatShape[0]+rowOffSet,colOffSet:imageMatShape[1]+colOffSet] = imageMat
    
    return padImageMat           

# takes the img dict, list classes to be subsetted, and eigen count needed to return 
# shape of images, eigen vals,  a matrix of all eigen vecs
def createEigenNodules(imgLabelsTbl, classLst, eigenCount, screePlot = True, imgFolderLoc = './nodules'):

    instanceIDsLabels = imgLabelsTbl[imgLabelsTbl["Labels"].isin(classLst)]
    
    # Reading in necessary the Dicom images   
    imgDict = {}
    imgSizeDict = {}    
    
    for fileName in os.listdir(imgFolderLoc):
        instanceID = int(fileName[:-4])
        if instanceID not in list(instanceIDsLabels.InstanceID): continue # we only pick images that are part of the 809
        fileLoc = imgFolderLoc +'/'+fileName 
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

    # Pad Image
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
    
    # Applying PCA on the new unrolled vectors
    normalizedMat = StandardScaler().fit_transform(fullImageMat)
    pca = PCA()
    redImgMat = pca.fit(normalizedMat)
    
    # ScreePlot of the Eigen Values
    if screePlot:
        title = 'Scree Plot for ' + str(classLst)
        plt.figure(figsize=(8,5))
        eigVals = redImgMat.explained_variance_ratio_[:eigenCount*10] # Look at 10times the number of visuals needed 
        plt.plot(range(1,len(eigVals)+1), eigVals, 'ro-', linewidth=2)
        plt.title(title)
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')

    imgShape = (maxRow, maxCol)
    
    return imgShape, redImgMat.explained_variance_ratio_[:eigenCount], redImgMat.components_[:eigenCount]


# returns value in a different scale

def maxMinScale(x, inMin, inMax, outMin, outMax):
    return (outMax-outMin)*((x - inMin)/(inMax-inMin)) + outMin

# returns the image matrix. 
# Options:
# useGlobalBounds also splits positive and negative bounds before scaling them seperately
# normalize results in a normalize vector being used
# useLocalwithClip redoes global clip but with local bounds

def genEigenImg(eigenVec, imgShape, globalMax, globalMin, useLocalwithClip = True, useGlobalBounds = False, normalize = True, displayImg=True):    
    
    if normalize: arr = eigenVec/np.linalg.norm(eigenVec)
    else: arr = eigenVec
    if useLocalwithClip:
        scaledArr= [maxMinScale(x,0,arr.max(),127.5,255) if x >= 0 else\
                    maxMinScale(x,arr.min(),0, 0, 127.49) for x in arr]
        reshapedEigenImg = np.reshape(scaledArr, newshape=imgShape)       
    elif useGlobalBounds:
        scaledArr= [maxMinScale(x,0,globalMax,127.5,255) if x >= 0 else\
                    maxMinScale(x,globalMin,0, 0, 127.49) for x in arr]
        reshapedEigenImg = np.reshape(scaledArr, newshape=imgShape)         
    else:
        reshapedEigenImg = np.reshape(255*(arr - arr.min())/(arr.max()-arr.min()), newshape=imgShape)
    
    if displayImg: img = Image.fromarray(reshapedEigenImg); img.show()
    return reshapedEigenImg

###############################################################################

















    
