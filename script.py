# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 00:25:46 2016

@author: SYARLAG1
"""

from eigenNoduleGen import * 

os.chdir('C:/Users/SYARLAG1/Documents/Lung-Nodule-Identification')    

varLst = ['InstanceID','Labels']
instanceIDsLabels = pd.read_csv('./LIDC_REU2015.csv', usecols=varLst) 

classLsts = [[1,2,3,4,5],[1],[2],[3],[4],[5]] 

eigenValDict = {}

##Generating the EigenNodules
for classLst in classLsts:    
    imgShape, eigVals, eigVecs = createEigenNodules(instanceIDsLabels, classLst, 10, screePlot = True, imgFolderLoc = './nodules')
    
    eigenValDict[tuple(classLst)] = eigVals    
    
    # Visual the top 10 eigen Nodules
    for index, arr in enumerate(eigVecs):
        imgFileName = './eigenNodules/'+','.join(map(str, classLst))+'_PC'+str(index+1)+'.png'
        img = genEigenImg(arr, imgShape)
        imsave(imgFileName, img)
        
eigValDf = pd.DataFrame(eigenValDict)
eigValDf.to_csv('./eigenValues.csv')