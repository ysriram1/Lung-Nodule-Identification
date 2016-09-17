# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 00:25:46 2016

@author: SYARLAG1
"""
import os
os.chdir('C:/Users/SYARLAG1/Documents/Lung-Nodule-Identification')    

from eigenNoduleGen import * 

varLst = ['InstanceID','Labels']
instanceIDsLabels = pd.read_csv('./LIDC_REU2015.csv', usecols=varLst) 

classLsts = [[1,2,3,4,5],[1],[2],[3],[4],[5]] 

eigenValDict = {}
dimDict = {}
maxMinDict = {}

##Generating the EigenNodules
for classLst in classLsts:    
    
    imgShape, eigVals, eigVecs = createEigenNodules(instanceIDsLabels, classLst, 10, screePlot = True, imgFolderLoc = './nodules')
    
    eigenValDict[tuple(classLst)] = eigVals    
    dimDict[tuple(classLst)] = imgShape    
    maxMinDict[tuple(classLst)] = (eigVecs.max(),eigVecs.min())
    
    # Visual the top 10 eigen Nodules
    for index, arr in enumerate(eigVecs):
        imgFileName = './eigenNodulesGlobalScaled2/'+','.join(map(str, classLst))+'_PC'+str(index+1)+'.png'
        img = genEigenImg(arr, imgShape, eigVecs.max(), eigVecs.min(), useGlobalBounds = True)
        imsave(imgFileName, img)
        
eigValDf = pd.DataFrame(eigenValDict)
eigValDf.to_csv('./eigenValues.csv')