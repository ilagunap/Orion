from utilityFunctions import HandleError, printMessage
import sys
import numpy
import math
import operator
import os
import time

class Column(object):
    def __init__(self, list):
        self.l = list
    
    def max(self):
        return max(self.l)
    
    def at(self, i):
        return self.l[i]
    
    def append(self, col):
        self.l.extend(col.l)
    
    def __str__(self):
        return str(self.l)
    
    def size(self):
        return len(self.l)
    
    def __eq__(self, other):
        for i in range(len(self.l)):
            if self.l[i] != other.l[i]:
                return False
        return True
        
'A numerical column has the following aggregates:'
'sum, average, standard-dev, min, max, count-zeros, '
'count-positives, count-negatives'
class NumColumn(Column):
    def __init__(self, list):
        Column.__init__(self, list)
    
    def diff(self):
        newList = []
        for i in range(len(self.l)):
            if (i > 0):
                newList.append(self.l[i] - self.l[i-1])

        self.l = newList
        
    def scalarProduct(self, col):
        newList = []
        for i in range(len(self.l)):
            newList.append(self.l[i] * col.at(i))
            
        return NumColumn(newList)
    
    def scalarDivision(self, col):
        newList = []
        for i in range(len(self.l)):
            newList.append(self.l[i] / float(col.at(i)))
            
        return NumColumn(newList)
    
    def substract(self, col):
        newList = []
        for i in range(len(self.l)):
            newList.append(self.l[i] - col.at(i))
            
        return NumColumn(newList)
    
    def dotProduct(self, col):
        ret = 0
        for i in range(len(self.l)):
            ret = ret + (self.l[i] * col.at(i))
        return ret
    
    def getAggregates(self):
        return NumColumn.calculateAggregates(self.l)
    
    @staticmethod
    def calculateAggregates(list):
        sum = float(numpy.sum(list))
        avg = float(numpy.average(list))
        std = float(numpy.std(list))
        min_v = float(min(list))
        max_v = float(max(list))
        countzeros = 0.0
        countpositives = 0.0
        countnegatives = 0.0
        
        for i in range(len(list)):
            if list[i] == 0:
                countzeros = countzeros + 1.0
            if list[i] > 0:
                countpositives = countpositives + 1.0
            if list[i] < 0:
                countnegatives = countnegatives + 1.0
        
        return [sum,avg,std,min_v,max_v,countzeros,countpositives,countnegatives]
    
    def getSubColumn(self, x, y):
        ret = self.l[x:y+1]
        return NumColumn(ret)
    
    def getCorrlationsColumn(self, colX, winSize):
        rows = len(self.l)
        numWins = int(math.floor(float(rows)/winSize))
        ret = []
        for i in range(numWins):
            x = i*winSize
            y = x + winSize - 1
            listX = self.l[x:y+1]
            listY = colX.l[x:y+1]
            corr = calculateCorrelation(listX, listY)
            ret.append(corr)
        return NumColumn(ret)
        
class StringColumn(Column):
    def __init__(self, list):
        Column.__init__(self, list)
        
    def size(self):
        return len(self.l)
    
    def diff(self):
        if (len(self.l) > 0):
            del(self.l[-1:])
            
    def getSubColumn(self, x, y):
        ret = self.l[x:y+1]
        return StringColumn(ret)
    
    def getUniqueObservations(self):
        ret = set([])
        for e in self.l:
            if e not in ret:
                ret.add(e)
        return ret
    
    # Returns one observation before and one after a given observation
    def getPreviousAndBeforeObservations(self, middleObs):
        ret = []
        for i in range(len(self.l)):
            if (self.l[i] == middleObs):
                seq = [middleObs]
                if i > 0:
                    seq.insert(0,self.l[i-1]) # obs before
                if i < (len(self.l)-1):
                    seq.append(self.l[i+1])
                    
                #flag = False
                #for e in ret:
                #    if str(e) == str(seq):
                #        flag = True
                #if flag == False:
                #    if len(seq) > 2:
                #        ret.append(seq)
                
                if seq not in ret:
                    if len(seq) > 2:
                        ret.append(seq)
                
        return ret
    
class Matrix:
    def __init__(self, rows, cols):
        self.listOfCols = []
        self.cols = cols
        self.rows = rows
        
    def addColumn(self, col):
        # If column is a 'Column' class
        if isinstance(col, Column):
            self.listOfCols.append(col)
            return
            
        # If it's a list
        if (len(col) != self.rows):
            HandleError.exit("Invalid size of column")
            
        if (len(self.listOfCols)+1 > self.cols):
            HandleError.exit("Too many columns")
        
        if (len(self.listOfCols) == 0):
            c = StringColumn(col)
        else:
            c = NumColumn(col)
        self.listOfCols.append(c)
        
        
    def max(self, col):
        return self.listOfCols[col].max()
        
    def getRow(self, index):
        ret = []
        for c in self.listOfCols:
            ret.append(c.at(index))
        return ret
    
    def getCol(self, index):
        #return self.listOfCols[index].getList()
        return self.listOfCols[index]
        
    def diff(self):
        for c in self.listOfCols:
            c.diff()
        self.rows = self.listOfCols[0].size()
        
    def removeColumns(self, list=[]):
        if len(list) == 0:
            return
        for i in list:
            if i > self.cols-1:
                HandleError.exit("Incorrect column index: " + str(i))
                
        newList = []
        for i in range(self.cols):
            if i not in list:
                newList.append(self.listOfCols[i])
        self.listOfCols = newList
        self.cols = len(self.listOfCols)
    
    def removeColumnsButKeep(self, list=[]):
        if len(list) == 0:
            return
        for i in list:
            if i > self.cols-1:
                HandleError.exit("Incorrect column index: " + str(i))
                
        newList = []
        for i in range(self.cols):
            if i in list:
                newList.append(self.listOfCols[i])
        self.listOfCols = newList
        self.cols = len(self.listOfCols)
        
    def __str__(self):
        ret = ""
        for i in range(self.rows):
            r = ""
            for j in range(self.cols):
                r = r + str(self.listOfCols[j].at(i))
                if (j < self.cols-1):
                    r = r + ", "
                    
            ret = ret + r
            if (i < self.rows-1):
                 ret = ret + "\n"
                 
        return ret
    
    # Returns a hyper-sphere of CCVs
    def getCorrelationMatrix(self, winSize):
        tmpCol = self.listOfCols[0]
        rows = tmpCol.getCorrlationsColumn(self.listOfCols[1], winSize).size()
        n = self.cols
        cols = n * (n-1) / 2
        newMatrix = Matrix(rows, cols)
        for i in range(len(self.listOfCols) - 1):
            for j in range(len(self.listOfCols) - i - 1):
                x = i
                y = x + j + 1
                colX = self.listOfCols[x]
                colY = self.listOfCols[y]
                corrCol = colX.getCorrlationsColumn(colY, winSize)
                newMatrix.addColumn(corrCol)
        return newMatrix
    
    def addMatrix(self, m):
        for i in range(len(self.listOfCols)):
            self.listOfCols[i].append(m.listOfCols[i])
        self.rows = self.rows + m.rows
    
class DataLoader:
    @staticmethod
    def load(fileName):
        data = []
        #print "Loading file in memory..."
        file = open(str(fileName), 'r')
        for l in file.readlines():
            r = l[:-1].split(',')
            data.append(r)
        file.close()
        #print "done!"
        
        #print "Creating data matrix..."
        cols = len(data[0])
        m = Matrix(len(data)-1, cols)
        for c in range(cols):
            col = []
            for r in range(len(data))[1:]:
                if (c > 0):
                    tmp = DataLoader.getValueFromString(data[r][c])
                else:
                    tmp = data[r][c]
                col.append(tmp)
            m.addColumn(col)
        #print "done!"
        
        return m
    
    @staticmethod
    def getValueFromString(valStr):
        if valStr == 'NaN':
            ret = 0
        else:
            if '.' in valStr:
                ret = float(valStr)
            else:
                ret = int(valStr)
        return ret
                    
    
class Histogram:
    'Compute frequencies of occurrence of events in column 1'
    'INPUT: 2-column matrix' 
    @staticmethod
    def histo(matrix):
        if matrix.__class__ != Matrix:
            HandleError.exit("In Histogram.histo: incorrect type: " + 
                             str(type(matrix)))
        if matrix.cols > 2:
            HandleError.exit("In Histogram.histo: matrix cols > 2")
            
        events = {}
        for r in range(matrix.rows):
            row = matrix.getRow(r)
            if row[0] in events.keys():
                events[row[0]] = events[row[0]] + row[1]
            else:
                events[row[0]] = row[1]
                
        m = Matrix(len(events.keys()), 2)
        m.addColumn(events.keys())
        m.addColumn(events.values())
        
        return m
    
    @staticmethod
    def histoWeighted(matrix):
        if matrix.__class__ != Matrix:
            HandleError.exit("In Histogram.histo: incorrect type: " + 
                             str(type(matrix)))
        if matrix.cols > 2:
            HandleError.exit("In Histogram.histo: matrix cols > 2")
            
        events = {}
        counts = {}
        for r in range(matrix.rows):
            row = matrix.getRow(r)
            if row[0] in events.keys():
                events[row[0]] = events[row[0]] + row[1]
                counts[row[0]] = counts[row[0]] + 1 
            else:
                events[row[0]] = row[1]
                counts[row[0]] = 1
                
        m = Matrix(len(events.keys()), 2)
        m.addColumn(events.keys())
        
        n1 = NumColumn(events.values())
        n2 = NumColumn(counts.values())
        m.addColumn(n1.scalarDivision(n2))
        
        return m
    
    'Normalize second column based on maximum value'
    @staticmethod
    def normalize(matrix):
        if matrix.__class__ != Matrix:
            HandleError.exit("In Histogram.histo: incorrect type: " + 
                             str(type(matrix)))
        if matrix.cols > 2:
            HandleError.exit("In Histogram.histo: matrix cols > 2")
            
        maximum = float(matrix.max(1))
        v = []
        for r in range(matrix.rows):
            row = matrix.getRow(r)
            v.append(row[1]/maximum*100)
        
        ret = Matrix(matrix.rows, 2)
        ret.addColumn(matrix.getCol(0))
        ret.addColumn(v)
        return ret
        
class Window(object):
    def __init__(self, col1, col2):
        if isinstance(col1, StringColumn) and isinstance(col2, NumColumn):
            if col1.size() != col2.size():
                HandleError.exit("In window: cols of different sizes")
            self.colA = col1
            self.colB = col2
        else:
            if len(col1) != len(col2):
                HandleError.exit("In window: cols of different sizes")
            self.colA = StringColumn(col1)
            self.colB = NumColumn(col2)
        self.aggr = self.colB.getAggregates()
        
    def size(self):
        return self.colA.size()
    
    def euclideanDistance(self, win2):
        if not isinstance(win2, Window):
            HandleError.exit("In euclideanDistance: argument is not a window")
            
        d = self.colB.substract(win2.colB)
        p = d.dotProduct(d)
        return math.sqrt(p)
    
    def aggregatesDistance(self, win2):
        if not isinstance(win2, Window):
            HandleError.exit("In aggregateDistance: argument is not a window")
            
        return Window.listDistance(win2.aggr, self.aggr)
    
    @staticmethod
    def listDistance(l1, l2):
        l = []
        for i in range(len(l1)):
            l.append(l1[i]-l2[i])
            
        ret = 0
        for i in range(len(l)):
            ret = ret + math.pow(l[i], 2)
        
        return math.sqrt(ret)
                
    def __eq__(self, other):
        if (self.colA == other.colA) and (self.colB == other.colB):
            return True
        else:
            return False
        
    def __hash__(self):
        return hash(str(self.colA) + str(self.colB))
    
    def __str__(self):
        ret = ""
        for i in range(self.colA.size()):
            ret = ret + str(self.colA.at(i)) + "|" + str(self.colB.at(i)) + "\n"
        return ret
    
    def normalizeAggregates(self, averageList, stdList):
        s = len(self.aggr)
        if (len(averageList) != s or len(stdList) != s):
            HandleError.exit("In normalizeAggregates: incorrect list sizes")
            
        for i in range(len(self.aggr)):
            if (stdList[i] != 0.0):
                self.aggr[i] = (self.aggr[i] - averageList[i]) / stdList[i]
            else:
                self.aggr[i] = 0
                
    def getUniqueObservations(self):
        return self.colA.getUniqueObservations()
    
    def getPreviousAndBeforeObservations(self, middleObs):
        seqs = self.colA.getPreviousAndBeforeObservations(middleObs)
        return seqs
        
##############################################################################
# Helper functions
##############################################################################

# Returns a set of windows
def createHyperSphere(matrix, winSize, winType='SLIDING'):
    c1 = matrix.getCol(0)
    c2 = matrix.getCol(1)
    rows = matrix.rows
    
    s = set([])
    if winType is 'REGULAR':
        for i in range(rows-winSize-1):
            x = i
            y = i + winSize - 1
            subCol1 = c1.getSubColumn(x, y)
            subCol2 = c2.getSubColumn(x, y)
            w = Window(subCol1, subCol2)
            s.add(w)
    elif winType is 'SLIDING':
        numWins = int(math.floor(float(rows)/winSize))
        for i in range(numWins):
            x = i*winSize
            y = x + winSize - 1
            subCol1 = c1.getSubColumn(x, y)
            subCol2 = c2.getSubColumn(x, y)
            w = Window(subCol1, subCol2)
            s.add(w)
    else:
        HandleError.exit("In createHyperSphere: incorrect window type")
        
    return s

# Nearest neighbor algorithm
# Returns a list of tuples (window, score)
def findOutlierWindows(normalSphere, abnormalMatrix, winSize, type, normal):
    abnormalSphere = createHyperSphere(abnormalMatrix, winSize)
    
    # Normalize spheres
    if normal:
        #log("Normalizing...")
        normalSphere = normalizeAggregates(normalSphere)
        abnormalSphere = normalizeAggregates(abnormalSphere)
    
    outliers = {} # Hash: key: window, value: NN-distance
    c = 0
    per = 5.0 # initial percentage
    for w in abnormalSphere:
        done = float(c)/len(abnormalSphere)*100
        if (done > per):
            #log("Done: " + str(round(done,1)) + "%")
            per = per + 5.0
        c = c + 1
        
        d = []
        # Iterate over normal sphere
        for x in normalSphere:
            if type is 'EUCLID':
                d.append(w.euclideanDistance(x))
            elif type is 'AGGR':
                d.append(w.aggregatesDistance(x))
        minDistance = min(d)
        outliers[w] = minDistance
        
    # Sort hash based on values
    l = sorted(outliers.items(), key=operator.itemgetter(1))
    l.reverse()
    return l # returns list of tuples

def addLists(l1, l2):
    ret = []
    for i in range(len(l1)):
        ret.append(l1[i] + l2[i])
    return ret
        
def multiplyList(c, l):
    ret = []
    for i in l:
        ret.append(c*i)
    return ret

def divideList(d, l):
    ret = []
    for i in l:
        ret.append(i/d)
    return ret

# multiply each element by p
def powerList(l, p):
    ret = []
    for i in l:
        ret.append(math.pow(i, p))
    return ret

# l1 - l2
def substractList(l1, l2):
    ret = []
    for i in range(len(l1)):
        ret.append(l1[i] - l2[i])
    return ret

def normalizeAggregates(sphere):
    # Get sum
    aggr = []
    for w in sphere:
        if (len(aggr) == 0):
            aggr = w.aggr
        else:
            aggr = addLists(aggr, w.aggr)
            
    # Get mean
    mu = divideList(len(sphere), aggr)
    
    # Get std
    sigma = []
    for w in sphere:
        l = substractList(w.aggr, mu)
        l = powerList(l, 2.0)
        if (len(sigma) == 0):
            sigma = l
        else:
            sigma = addLists(sigma, l)
    sigma = divideList(len(sphere), sigma)
    sigma = powerList(sigma, 0.5) # root square
    
    # Normalize each window
    s = set([])
    for w in sphere:
        w.normalizeAggregates(mu, sigma)
        s.add(w)
    return s
    

def log(msg):
    #fd = open('./bugfinder.log', 'a')
    #fd.write(str(msg) + "\n")
    #fd.close()
    print msg

def cleanLog():
    fd = open('./bugfinder.log', 'w')
    fd.close()
    
def saveWindow(w, id):
    dir = "output_wins"
    if not os.path.exists(dir):
        os.makedirs(dir)
    fileName = "win_" + str(id)
    file = open(os.path.join(dir, fileName),'w')
    file.write(str(w))
    file.close()
    
# Calculate Pearson's correlation coefficient
# based on two list of numeric values
def calculateCorrelation(listX, listY):
    if len(listX) != len(listY):
        HandleError.exit("In calculateCorrelation: lists of different sizes")
    
    avg_x = numpy.average(listX)
    avg_y = numpy.average(listY)
    std_x = numpy.std(listX)
    std_y = numpy.std(listY)
    
    ret = 0
    for i in range(len(listX)):
        tmp1 = (listX[i] - avg_x) / std_x
        tmp2 = (listY[i] - avg_y) / std_y
        ret = ret + (tmp1 * tmp2)
    ret = ret / (len(listX) - 1)
    if math.isnan(ret):
        ret = 0
    return ret

def dotProduct(listX, listY):
    ret = 0
    for i in range(len(listX)):
        ret = ret + (listX[i] * listY[i])
    return ret
    
def euclideanDistance(listX, listY):
    d = []
    for i in range(len(listX)):
        d.append(listX[i] - listY[i])
    dis = dotProduct(d, d)
    return math.sqrt(dis)
    
##############################################################################
# Public Interface
##############################################################################

def findAnomalousWindows(normalFileName, abnormalFileName, metric, manyWins):
    if manyWins == True:
        winSize = range(100,250,10)
    else:
        winSize = [150, 200, 250]
    
    distanceType = ['AGGR']
    normalize = [True]
    
    m = DataLoader.load(normalFileName)
    m.diff()
    m.removeColumnsButKeep([0,int(metric)])
    
    abnormalMatrix = DataLoader.load(abnormalFileName)
    abnormalMatrix.diff()
    abnormalMatrix.removeColumnsButKeep([0, int(metric)])
    
    printMessage("Finding outliers...")
    ret = []
    for size in winSize:
        sphere = createHyperSphere(m, size)
        for t in distanceType:
            for n in normalize:
                outliers = findOutlierWindows(sphere, abnormalMatrix, size, t, n)                
                # Return top-5 anomalous windows
                for i in range(5):
                    if len(outliers) >= (i+1):
                        ret.append(outliers[i][0])
    return ret

# Find the most frequent points in the windows list.
# method: CLASSNAME_ONLY, CLASSNAME_AND_METHOD
def findAnomalousPoints(windowsList, method):
    occurrenceNumber = {}
    subNameOccurr = {}
    for win in windowsList:
        obsSet = win.getUniqueObservations()
        for o in obsSet:
            # Parse observation
            if method == 'CLASSNAME_ONLY':
                
                # Only split by '-' in Java applications.
                if '-' in o:
                    name = o.split('-')[1].split('$')[0]
                    tmp = o.split('-')[1].split('$')[1:]
                    subName = "$".join(tmp)
                else:
                    name = o
                    subName = o
                      
                if name not in subNameOccurr.keys():
                    tmp = {}
                    tmp[subName] = 1
                    subNameOccurr[name] = tmp
                else:
                    if subName not in subNameOccurr[name].keys():
                        subNameOccurr[name][subName] = 1
                    else:
                        subNameOccurr[name][subName] = subNameOccurr[name][subName] + 1
                
            elif method == 'CLASSNAME_AND_METHOD':
                if '-' in o:
                    name = o.split('-')[1]
                else:
                    name = o
            else:
                HandleError.exit('in findAnomalousPoints: unknown method')
            
            if name not in occurrenceNumber.keys():
                occurrenceNumber[name] = 1
            else:
                occurrenceNumber[name] = occurrenceNumber[name] + 1
    
    l = sorted(occurrenceNumber, key=occurrenceNumber.get)
    l.reverse()
    ret = []
    for e in l:
        ret.append((e, occurrenceNumber[e]))
    
    #print "org/apache/hadoop/dfs/DFSClient", subNameOccurr['org/apache/hadoop/dfs/DFSClient']
    #print "org/apache/hadoop/hbase/regionserver/HRegion", subNameOccurr['org/apache/hadoop/hbase/regionserver/HRegion']
    return ret

def printCulpritSubWindows(abnormalCodeRegions, abnormalWindows):
    
    # Hash of key: code_region, value: list of sets of strings
    sequences = {}
    
    for i in range(4):
        region = abnormalCodeRegions[i][0]
        for w in abnormalWindows:
            seqs = w.getPreviousAndBeforeObservations(region)
            
            if region not in sequences.keys():
                sequences[region] = [seqs]
            else:
                for seq in seqs:
                    if seq not in sequences[region]:
                        sequences[region].append(seq)
    
    print "\n++++++++++ Abnormal Sequences +++++++"
    for region in sequences.keys():
        print "For region:", region
        for seq in sequences[region]:
            for reg in seq:
                print "\t", reg
            print ""
            
def localizationAnalysis(normalFile, abnormalFile, metric, manyWins):    
    windows = findAnomalousWindows(normalFile, abnormalFile, metric, manyWins)
    obs = findAnomalousPoints(windows, 'CLASSNAME_ONLY')
    #obs = findAnomalousPoints(windows, 'CLASSNAME_AND_METHOD')
    
    print '\n========== Top-3 Abnormal Code Regions =========='
    print '[1]:'
    #print '\t[' + str(obs[0][1]) + ']', obs[0][0]
    prev_dist = obs[0][1]
    
    level = 1
    for o in obs:
        if prev_dist == o[1]:
            print '\t[' + str(o[1]) + ']', o[0]
        else:
            level = level + 1
            if level > 3:
                break
            
            print '[' + str(level) + ']:'
            print '\t[' + str(o[1]) + ']', o[0]
            
        prev_dist = o[1]
     
    # For IBM case   
    #printCulpritSubWindows(obs, windows)

##############################################################################
# Main script
##############################################################################
def main():
    cleanLog()
    winSize = [50]
    #winSize = [4]
    #distanceType = ['EUCLID','AGGR']
    distanceType = ['AGGR']
    normalize = [True, False]

    log("Loading files...")
    fileName = sys.argv[1]
    m = DataLoader.load(fileName)
    m.diff()
    m.removeColumnsButKeep([0,11])

    fileName = sys.argv[2]
    abnormalMatrix = DataLoader.load(fileName)
    abnormalMatrix.diff()
    abnormalMatrix.removeColumnsButKeep([0,11])

    log("Finding outliers...")
    for size in winSize:
        for t in distanceType:
            sphere = createHyperSphere(m, size)
            for n in normalize:
                outliers = findOutlierWindows(sphere, abnormalMatrix, size, t, n)
                log("***** Saving windows: " + t + " win-size: " + 
                    str(size) + " norm: " + str(n))
                id = str(size) + "_" + t + "_norm_" + str(n) 
                saveWindow(outliers[0][0], id + str(1))
                saveWindow(outliers[1][0], id + str(2))
                saveWindow(outliers[2][0], id + str(3))
                saveWindow(outliers[3][0], id + str(4))
                saveWindow(outliers[4][0], id + str(5))
                
if __name__ == '__main__':
    main()
else:
    print "Localization module loaded."
        