#!/usr/bin/env python

from localization import findAnomalousWindows, findAnomalousPoints, printCulpritSubWindows, localizationAnalysis
from utilityFunctions import HandleError, printMessage
from corranalysis import metricsAnalysis
import sys
import optparse
import time

#############################################################################
# Helper functions
#############################################################################

def parseOptions():
    usage = "usage: %prog [options]"
    parser = optparse.OptionParser(usage)
    
    parser.add_option("--select-metrics", dest="SELECT_METRICS",
                      action="store_true", default=False, 
                      help="Select abnormal metrics.")
    
    parser.add_option("--select-regions", dest="SELECT_REGIONS",
                      action="store_true", default=False, 
                      help="Select abnormal code regions.")
    
    parser.add_option("-m", "--metric", dest="METRIC", 
                      help="Name for the abnormal metric.")
    
    parser.add_option("-a", "--abnormal-traces", dest="AFILE", 
                      help="Abnormal traces file.")
    parser.add_option("-n", "--normal-traces", dest="NFILE", 
                      help="Normal traces file.")
    
    parser.add_option("-w", "--many-windows", dest="MANY_WINDOWS",
                      action="store_true", default=False, 
                      help="Use a large number of windows in the analysis.\nUseful when data sets are small.")
    
    (options, args) = parser.parse_args()
    #if len(options) == 0:
    #    parser.print_help()
    #    exit()
        
    return (options, args)

def getAbnormalFile(options):
    if options.AFILE is None:
        HandleError.exit('No abnormal-traces file given.\nUse -h option for help.')
    else:
        return options.AFILE

def getNormalFile(options):
    if options.NFILE is None:
        HandleError.exit('No normal-traces file given.\nUse -h option for help.')
    else:
        return options.NFILE
    
def useManyWindows(options):
    return options.MANY_WINDOWS
    
def getMetric(options):
    metric_id = ''
    if options.METRIC is None:
        HandleError.exit('No abnormal metric is given.\nUse -h option for help.')
    else:
        metric_name = options.METRIC
        
        # Find metric number in the file
        normalFile = options.NFILE
        file = open(normalFile, 'r')
        metrics = file.readline()[:-1].split(',')
        del(metrics[0])
        file.close()
        
        metric_found = False
        for i in range(len(metrics)):
            if metric_name == metrics[i]:
                metric_found = True
                metric_id = i + 1
                break
        
        if metric_found == False:
            HandleError.exit('Unknown metric name.')
            
    return metric_id
    
# Returns two operation-mode types: SELECT_METRICS and SELECT_REGIONS
def getMode(options):
    select_metrics = options.SELECT_METRICS
    select_regions = options.SELECT_REGIONS
    
    # Can only specify one operational mode
    if select_metrics is True and select_regions is True:
        HandleError.exit('Cannot use these options together:\n --select-metrics & --select-regions. \nUse -h option for help.')
        
    if select_metrics is False and select_regions is False:
        HandleError.exit('Please use one of these options:\n--select-metrics OR --select-regions. \nUse -h option for help.')
        
    if select_metrics is True:
        mode = 'SELECT_METRICS'
    elif select_regions is True:
        mode = 'SELECT_REGIONS'
        
    return mode
    
#############################################################################
# Main script
#############################################################################

# Parse options
(options, args) = parseOptions()
mode = getMode(options)
abnormalFile = getAbnormalFile(options)
normalFile = getNormalFile(options)
manyWins = useManyWindows(options)

printMessage('Normal File: ' + normalFile)
printMessage('Abnormal File: ' + abnormalFile)

if mode == 'SELECT_METRICS':
    metricsAnalysis(normalFile, abnormalFile)
elif mode == 'SELECT_REGIONS':
    metric = getMetric(options)
    localizationAnalysis(normalFile, abnormalFile, metric, manyWins)

