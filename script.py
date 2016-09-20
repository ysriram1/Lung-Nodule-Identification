# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 00:25:46 2016

@author: SYARLAG1
"""


import cv2
import os
import pandas as pd
from scipy.misc import imsave

os.chdir('C:/Users/SYARLAG1/Documents/Lung-Nodule-Identification/')    

from eigenNoduleGen import *

#from eigenNoduleGen import * 

varLst = ['InstanceID','Labels']
instanceIDsLabels = pd.read_csv('./LIDC_REU2015.csv', usecols=varLst) 

classLsts = [[1,2,3,4,5],[1],[2],[3],[4],[5]] 

eigenValDict = {}
eigenVecDict = {}
dimDict = {}
maxMinDict = {}
imgLst = []
imageMatDict = {}

##Generating the EigenNodules
for iClass, classLst in enumerate(classLsts):    
    
    imgShape, eigVals, eigVecs,fullImageMat = createEigenNodules(instanceIDsLabels, classLst, 10, screePlot = True, imgFolderLoc = './nodules')
    
    eigenValDict[tuple(classLst)] = eigVals    
    dimDict[tuple(classLst)] = imgShape    
    maxMinDict[tuple(classLst)] = (eigVecs.max(),eigVecs.min())
    eigenVecDict[tuple(classLst)] =  eigVecs
    imageMatDict[tuple(classLst)] = fullImageMat[:]    
    
    # Visualize the top 10 eigen Nodules
    for index, arr in enumerate(eigVecs):
        imgFileName = './EigenNodulesLocalScaledwZeroCenter/'+','.join(map(str, classLst))+'_PC'+str(index+1)+'.png'
        img = genEigenImg(arr, imgShape, eigVecs.max(), eigVecs.min(), useLocalwithClip=True, normalize=False, displayImg=False)
        #cv2.imwrite(imgFileName, img) # currently this is the only func that doesnt auto normalize
        
eigValDf = pd.DataFrame(eigenValDict)
eigValDf.to_csv('./eigenValues.csv')

## Gen mean images
for classLst in imageMatDict.keys():
    print imageMatDict[classLst].shape
    imgFileName = './meanImages/'+','.join(map(str, classLst))+'_PC'+str(index+1)+'.TIFF'
    img = np.reshape(np.mean(imageMatDict[classLst],0), newshape=dimDict[classLst])
    imsave(imgFileName, img)



##Saving TIFF images from negative arrays
for classLst in eigenVecDict.keys():
    for index, arr in enumerate(eigenVecDict[classLst]):
        imgFileName = './negativeImagesTIFF/'+','.join(map(str, classLst))+'_PC'+str(index+1)+'.TIFF'
        img = np.reshape(arr, newshape=dimDict[classLst])
        imsave(imgFileName, img)