#!/usr/bin/env python
import os,sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir) 

from corranalysis import getAbnormalCorrelations, getAbnormalCCVs, getAbnormalFeatures
from localization import DataLoader, Matrix

###############################################################################
# Main script
###############################################################################

# Names of input files
file1 = "./file1.dat"
file2 = "./file2.dat"

# Load files. The 'DataLoader.load' returns a 'Matrix' instance.
matrix1 = DataLoader.load(file1)
matrix2 = DataLoader.load(file2)

# Differentiate each matrix and eliminate the first 
# column (which contains ID's).
matrix1.diff()
matrix1.removeColumns([0])
matrix2.diff()
matrix2.removeColumns([0])

# Print matrices
print "\nMatrix 1:"
print matrix1
print "\nMatrix 2:"
print matrix2

# Get a "correlation matrix" for each matrix. A correlation matrix is a
# hyper-sphere, i.e., a matrix where each row is a CCV.
# We set a window size of 4 samples.
windowSize = 4
corrMatrix1 = matrix1.getCorrelationMatrix(windowSize)
corrMatrix2 = matrix2.getCorrelationMatrix(windowSize)

# Print correlation matrices
print "\nCorrelation Matrix 1:"
print corrMatrix1
print "\nCorrelation Matrix 2:"
print corrMatrix2
print "\nNotice that each matrix has only 4 rows, and each row is a CCV."

# Get Nearest-neighbor distance and abnormal correlations in the 
# top-K abnormal CCVs
K = 2
print "\nCharacteristics of the top", K, "abnormal CCVs:"
abnormalCCVs = getAbnormalCCVs(corrMatrix1, corrMatrix2, K)
for ccv in abnormalCCVs:
    print ccv

# Get top-D abnormal correlations (or features) in the abnormal CCVs
D = 3
abnormalFeatures = getAbnormalFeatures(abnormalCCVs, D)
print "\nList of abnormal correlations:"
for f in range(len(abnormalFeatures)):
    print "Abnormal features for CCV", f, ":", abnormalFeatures[f]
