from localization import DataLoader, dotProduct, euclideanDistance
from utilityFunctions import printMessage, HandleError
import sys
import math
import operator
    
class CorrelationVector(object):
    def __init__(self, diss, dimensionsList):
        self.diss = diss
        self.dimList = dimensionsList
        
    def __lt__(self, other):
        return self.diss < other.diss
    
    def __str__(self):
        ret = "[Nearest-neighbor distance: " + str(self.diss)
        ret = ret + ", Abnormal features: " + str(self.dimList) + "]"
        return ret

# Returns tuple (a,b)
# a: Euclidean distance between the two list
# b: ordered list of dimensions in listX (according to dissimilarity)
def calculateVectorDistance(listX, listY):
    dist = euclideanDistance(listX, listY)
    rank = {} # key: index, value: difference
    for i in range(len(listX)):
        diff = math.fabs(listX[i] - listY[i])
        rank[i] = diff
        
    # sort by values
    values = sorted(rank.items(), key=operator.itemgetter(1))
    values.reverse()
    l = [] # list of ordered dimensions
    for tup in values:
        l.append(tup[0])
        
    return (dist, l)

# This function returns a list of abnormal correlations (or correlation
# coefficients) in the following form:
# [[2,45,100], [100,3,9], [...], ...]
# Each element in the outer list corresponds to the most abnormal correlations
# (i.e., those that contribute the most to the distance) for a given CCV.
#
# What this example says is that, 2, 45, and 100 are the top-3 most abnormal 
# correlation coefficients for the first CCV (i.e., the most abnormal CCV).
#
# Subsequently, 100, 3 and 9, are the top-3 most abnormal 
# correlation coefficients for the second most abnormal CCV...and so on.
#
# Parameters:
# K: top-k the most distant correlation-coefficient vectors (CCVs)
# D: top-d the most abnormal correlations within each CCV
def getAbnormalCorrelations(corrMatrixA, corrMatrixB, K, D):
    abnormalCCVs = getAbnormalCCVs(corrMatrixA, corrMatrixB, K)
    ret = getAbnormalFeatures(abnormalCCVs, D)
    return ret

# Get characteristics top-K abnormal correlation coefficient vectors (CCVs).
# A CCV is represented by an instance of 'CorrelationVector'.
def getAbnormalCCVs(corrMatrixA, corrMatrixB, K):
    corrVecs = []
    for i in range(corrMatrixB.rows):
        rowB = corrMatrixB.getRow(i)
        vecs = []
        for j in range(corrMatrixA.rows):
            rowA = corrMatrixA.getRow(j)
            (dist, dimsList) = calculateVectorDistance(rowB, rowA)
            vecs.append(CorrelationVector(dist, dimsList))
        # Here we sort CCVs based on their dissimilarity
        # which is stored in the 'dist' variable of a CorrelationVector object
        vecs.sort()
        # Since we are using nearest-neighbor, we take the first element of
        # the 'vecs' array, i.e., the closest CCV to 'rowB'.
        corrVecs.append(vecs[0])
        
    # Now we sort and reverse 'corrVecs' to put the most abnormal CCVs at the
    # beginning and the most normal CCVs at the end. 
    corrVecs.sort()
    corrVecs.reverse()
    
    # Return only the top-K abnormal CCVs
    return corrVecs[0:K]

# Iterates over the abnormal CCVs and finds the top-D abnormal correlations.
# A CCV is an instance of 'CorrelationVector' which contains two variables --
# one of them containing a list of the most abnormal 
# correlations (i.e., dimList). The function iterates over 'dimList' (for each 
# CCV) and finds the top D elements.
def getAbnormalFeatures(abnormalCCVs, D):
    ret = []
    for i in range(len(abnormalCCVs)):
        tmp = []
        for j in range(D):
            tmp.append(abnormalCCVs[i].dimList[j])
        ret.append(tmp)
    return ret

def log(msg):
    print msg
    
def getFeaturesNames(fileName):
    file = open(fileName)
    lines = file.readlines()
    f = lines[0][0:-1]
    file.close()
    l = f.split(",")
    return l

def getMatricsFromCorrelationIndex(index, n):
    count = 0
    for i in range(n - 1):
        for j in range(n - i - 1):
            x = i
            y = x + j + 1
            if index == count:
                return (x, y)
            count = count + 1
    HandleError.exit('Could not find correlation index')
            
def printCorrelations(index, n, metrics):
    t = getMatricsFromCorrelationIndex(index, n)
    print 'Corr', index, metrics[t[0]], metrics[t[1]]
    
def findAbnormalMetrics(corrLists, metrics, n):
    metricsRank = {} # key: metric name, value: frequency of appearance
    for i in range(len(corrLists)):
        for c in corrLists[i]:
            t = getMatricsFromCorrelationIndex(c, n)
            name1 = metrics[t[0]]
            name2 = metrics[t[1]]
            
            # Add metrics to rank table
            if name1 not in metricsRank.keys():
                metricsRank[name1] = 1
            else:
                metricsRank[name1] = metricsRank[name1] + 1
                
            if name2 not in metricsRank.keys():
                metricsRank[name2] = 1
            else:
                metricsRank[name2] = metricsRank[name2] + 1
                
    l = sorted(metricsRank, key=metricsRank.get)
    l.reverse()
    return l

def printResults(metricsRank):
    # Print top N abnormal metrics
    l = sorted(metricsRank, key=metricsRank.get)
    l.reverse()
    
    print '\n========== Top-3 Abnormal Metrics =========='
    print 'Format: [Rank] [Metric]'
    i = 1
    for m in l:
        print '[' + str(i) + ']:', m 
        if i == 3:
            break
        i = i + 1
        
    print '\n========== Other Metrics =========='
    N = 3
    i = 1
    for m in l:
        if i > 3 and i <= 10:
            print '[' + str(i) + ']:', m
        if i == 10:
            break
        i = i + 1

#############################################################################
# Main script
#############################################################################

def metricsAnalysis(normalFile, abnormalFile):
    # Parameters
    winSize = [100, 125, 150, 175, 200]
    #winSize = range(200,400,20)
    K = [3] # top-k abnormal correlations
    D = [3] # top-d abnormal dimensions
    
    printMessage('Loading data files...')
    normM = DataLoader.load(normalFile)
    normM.diff()
    normM.removeColumns([0])
    n = normM.cols
    
    abnormM = DataLoader.load(abnormalFile)
    abnormM.diff()
    abnormM.removeColumns([0])
    
    # Get features names
    metrics = getFeaturesNames(normalFile)
    del(metrics[0]) # remove ID metric
    
    metricsRank = {}
    for w in winSize:
        printMessage('Calculating correlations for window-size: ' + str(w))
        normalCorrMatrix = normM.getCorrelationMatrix(w)
        abnormalCorrMatrix = abnormM.getCorrelationMatrix(w)
        
        for k in K:
            for d in D:
                printMessage('Finding abnormal correlations...')
                corrList = getAbnormalCorrelations(normalCorrMatrix, abnormalCorrMatrix, k, d)
                abnormalMetrics = findAbnormalMetrics(corrList, metrics, n)
                for m in abnormalMetrics:                    
                    if m not in metricsRank.keys():
                        metricsRank[m] = 1
                    else:
                        metricsRank[m] = metricsRank[m] + 1
    
    printResults(metricsRank)
                    
if __name__ == '__main__':
    normalFile = sys.argv[1]
    abnormalFile = sys.argv[2]
    metricsAnalysis(normalFile, abnormalFile)
else:
    print "Correlation analysis module loaded."