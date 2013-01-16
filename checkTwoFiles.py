#!/usr/bin/env python

from localization import DataLoader, Column, Matrix
import sys

###############################################################################
# Main script
###############################################################################

fileA = sys.argv[1]
fileB = sys.argv[2]

fileAMatrix = DataLoader.load(fileA)
fileBMatrix = DataLoader.load(fileB)

print "File A:", "cols:", fileAMatrix.cols, "rows:", fileAMatrix.rows
print "File B:", "cols:", fileBMatrix.cols, "rows:", fileBMatrix.rows

for i in range(fileAMatrix.cols):
    if i > 0:
        print "Comparing col", i
        colA = fileAMatrix.getCol(i)
        colB = fileBMatrix.getCol(i)
    
        for j in range(colA.size()):
            diff = float(colA.at(j)) - float(colB.at(j))
            if diff > 0.001:
                print "\tDiff:", diff