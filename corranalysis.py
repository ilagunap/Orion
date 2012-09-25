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

# For each row in matrix B: calculates nearest neighbor distances
# with respect to matrix A.
# K: top-k most distance correlation vectors to return
# D: top-d most abnormal correlations within each returned correlation vector
#
# Returns:
# List of k correlation vectors and most abnormal correlations
# [[2,45,100], [100,3,9], [...], ...]
def getAbnormalCorrelations(corrMatrixA, corrMatrixB, K, D):
    corrVecs = []
    for i in range(corrMatrixB.rows):
        rowB = corrMatrixB.getRow(i)
        vecs = []
        for j in range(corrMatrixA.rows):
            rowA = corrMatrixA.getRow(j)
            (dist, dimsList) = calculateVectorDistance(rowB, rowA)
            vecs.append(CorrelationVector(dist, dimsList))
        vecs.sort()
        corrVecs.append(vecs[0])
    corrVecs.sort()
    corrVecs.reverse()
    
    ret = []
    for i in range(K):
        tmp = []
        for j in range(D):
            tmp.append(corrVecs[i].dimList[j])
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