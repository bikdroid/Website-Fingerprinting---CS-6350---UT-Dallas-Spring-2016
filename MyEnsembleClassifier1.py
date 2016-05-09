# This is a Python framework to compliment "Peek-a-Boo, I Still See You: Why Efficient Traffic Analysis Countermeasures Fail".
# Copyright (C) 2012  Kevin P. Dyer (kpdyer.com)
# See LICENSE for more details.

import sys
import config
import time
import os
import random
import getopt
import string
import csv
import itertools
import numpy as np


# custom
from Datastore import Datastore
from Webpage import Webpage

# countermeasures
from PadToMTU import PadToMTU
from PadRFCFixed import PadRFCFixed
from PadRFCRand import PadRFCRand
from PadRand import PadRand
from PadRoundExponential import PadRoundExponential
from PadRoundLinear import PadRoundLinear
from MiceElephants import MiceElephants
from DirectTargetSampling import DirectTargetSampling
from Folklore import Folklore
from WrightStyleMorphing import WrightStyleMorphing

# classifiers
from LiberatoreClassifier import LiberatoreClassifier
from WrightClassifier import WrightClassifier
from BandwidthClassifier import BandwidthClassifier
from HerrmannClassifier import HerrmannClassifier
from TimeClassifier import TimeClassifier
from PanchenkoClassifier import PanchenkoClassifier
from VNGPlusPlusClassifier import VNGPlusPlusClassifier
from VNGClassifier import VNGClassifier
from JaccardClassifier import JaccardClassifier
from ESORICSClassifier import ESORICSClassifier

def intToCountermeasure(n):
    countermeasure = None
    if n == config.PAD_TO_MTU:
        countermeasure = PadToMTU
    elif n == config.RFC_COMPLIANT_FIXED_PAD:
        countermeasure = PadRFCFixed
    elif n == config.RFC_COMPLIANT_RANDOM_PAD:
        countermeasure = PadRFCRand
    elif n == config.RANDOM_PAD:
        countermeasure = PadRand
    elif n == config.PAD_ROUND_EXPONENTIAL:
        countermeasure = PadRoundExponential
    elif n == config.PAD_ROUND_LINEAR:
        countermeasure = PadRoundLinear
    elif n == config.MICE_ELEPHANTS:
        countermeasure = MiceElephants
    elif n == config.DIRECT_TARGET_SAMPLING:
        countermeasure = DirectTargetSampling
    elif n == config.WRIGHT_STYLE_MORPHING:
        countermeasure = WrightStyleMorphing
    elif n > 10:
        countermeasure = Folklore

        # FIXED_PACKET_LEN: 1000,1250,1500
        if n in [11,12,13,14]:
            Folklore.FIXED_PACKET_LEN    = 1000
        elif n in [15,16,17,18]:
            Folklore.FIXED_PACKET_LEN    = 1250
        elif n in [19,20,21,22]:
            Folklore.FIXED_PACKET_LEN    = 1500

        if n in [11,12,13,17,18,19]:
            Folklore.TIMER_CLOCK_SPEED   = 20
        elif n in [14,15,16,20,21,22]:
            Folklore.TIMER_CLOCK_SPEED   = 40

        if n in [11,14,17,20]:
            Folklore.MILLISECONDS_TO_RUN = 0
        elif n in [12,15,18,21]:
            Folklore.MILLISECONDS_TO_RUN = 5000
        elif n in [13,16,19,22]:
            Folklore.MILLISECONDS_TO_RUN = 10000

        if n==23:
            Folklore.MILLISECONDS_TO_RUN = 0
            Folklore.FIXED_PACKET_LEN    = 1250
            Folklore.TIMER_CLOCK_SPEED   = 40
        elif n==24:
            Folklore.MILLISECONDS_TO_RUN = 0
            Folklore.FIXED_PACKET_LEN    = 1500
            Folklore.TIMER_CLOCK_SPEED   = 20
        elif n==25:
            Folklore.MILLISECONDS_TO_RUN = 5000
            Folklore.FIXED_PACKET_LEN    = 1000
            Folklore.TIMER_CLOCK_SPEED   = 40
        elif n==26:
            Folklore.MILLISECONDS_TO_RUN = 5000
            Folklore.FIXED_PACKET_LEN    = 1500
            Folklore.TIMER_CLOCK_SPEED   = 20
        elif n==27:
            Folklore.MILLISECONDS_TO_RUN = 10000
            Folklore.FIXED_PACKET_LEN    = 1000
            Folklore.TIMER_CLOCK_SPEED   = 40
        elif n==28:
            Folklore.MILLISECONDS_TO_RUN = 10000
            Folklore.FIXED_PACKET_LEN    = 1250
            Folklore.TIMER_CLOCK_SPEED   = 20


    return countermeasure

def intToClassifier(n):
    classifier = None
    if n == config.LIBERATORE_CLASSIFIER:
        classifier = LiberatoreClassifier
    elif n == config.WRIGHT_CLASSIFIER:
        classifier = WrightClassifier
    elif n == config.BANDWIDTH_CLASSIFIER:
        classifier = BandwidthClassifier
    elif n == config.HERRMANN_CLASSIFIER:
        classifier = HerrmannClassifier
    elif n == config.TIME_CLASSIFIER:
        classifier = TimeClassifier
    elif n == config.PANCHENKO_CLASSIFIER:
        classifier = PanchenkoClassifier
    elif n == config.VNG_PLUS_PLUS_CLASSIFIER:
        classifier = VNGPlusPlusClassifier
    elif n == config.VNG_CLASSIFIER:
        classifier = VNGClassifier
    elif n == config.JACCARD_CLASSIFIER:
        classifier = JaccardClassifier
    elif n == config.ESORICS_CLASSIFIER:
        classifier = ESORICSClassifier

    return classifier

def usage():
    print """
    -N [int] : use [int] websites from the dataset
               from which we will use to sample a privacy
               set k in each experiment (default 775)

    -k [int] : the size of the privacy set (default 2)

    -d [int]: dataset to use
        0: Liberatore and Levine Dataset (OpenSSH)
        1: Herrmann et al. Dataset (OpenSSH)
        2: Herrmann et al. Dataset (Tor)
        (default 1)

    -C [int] : classifier to run
        0: Liberatore Classifer
        1: Wright et al. Classifier
        2: Jaccard Classifier
        3: Panchenko et al. Classifier
        5: Lu et al. Edit Distance Classifier
        6: Herrmann et al. Classifier
        4: Dyer et al. Bandwidth (BW) Classifier
        10: Dyer et al. Time Classifier
        14: Dyer et al. Variable n-gram (VNG) Classifier
        15: Dyer et al. VNG++ Classifier
        (default 0)

    -c [int]: countermeasure to use
        0: None
        1: Pad to MTU
        2: Session Random 255
        3: Packet Random 255
        4: Pad Random MTU
        5: Exponential Pad
        6: Linear Pad
        7: Mice-Elephants Pad
        8: Direct Target Sampling
        9: Traffic Morphing
        (default 0)

    -t [int]: number of trials to run per experiment (default 1)

    -t [int]: number of training traces to use per experiment (default 16)

    -T [int]: number of testing traces to use per experiment (default 4)

    -R [int] : threshold for moving the seed
    """

def run():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "t:T:R:N:k:c:C:d:n:r:h")
    except getopt.GetoptError, err:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    char_set = string.ascii_lowercase + string.digits
    runID = ''.join(random.sample(char_set,8))

    for o, a in opts:
        if o in ("-k"):
            config.BUCKET_SIZE = int(a)
        elif o in ("-C"):
            config.CLASSIFIER = int(a)
        elif o in ("-d"):
            config.DATA_SOURCE = int(a)
        elif o in ("-c"):
            config.COUNTERMEASURE = int(a)
        elif o in ("-N"):
            config.TOP_N = int(a)
        elif o in ("-t"):
            config.NUM_TRAINING_TRACES = int(a)
        elif o in ("-T"):
            config.NUM_TESTING_TRACES = int(a)
        elif o in ("-n"):
            config.NUM_TRIALS = int(a)
        elif o in ("-r"):
            runID = str(a)
        elif o in ("-R"):
            config.THRESHOLD = int(a)
        else:
            usage()
            sys.exit(2)

    from datetime import datetime
    mytime = datetime.now().time()

    outputFilenameArray = ['results',
                           'k'+str(config.BUCKET_SIZE),
                           'c'+str(config.COUNTERMEASURE),
                           'd'+str(config.DATA_SOURCE),
                           'C'+str(config.CLASSIFIER),
                           'N'+str(config.TOP_N),
                           't'+str(config.NUM_TRAINING_TRACES),
                           'T'+str(config.NUM_TESTING_TRACES),
                           'full_cycle_' + str(mytime)
                          ]
    outputFilename = os.path.join(config.OUTPUT_DIR,'.'.join(outputFilenameArray))

    if not os.path.exists(config.CACHE_DIR):
        os.mkdir(config.CACHE_DIR)

    if not os.path.exists(outputFilename+'.output'):
        banner = ['accuracy','overhead','timeElapsedTotal','timeElapsedClassifier']
        f = open( outputFilename+'.output', 'w' )
        f.write(','.join(banner))
        f.close()
    if not os.path.exists(outputFilename+'.debug'):
        f = open( outputFilename+'.debug', 'w' )
        f.close()

    if config.DATA_SOURCE == 0:
        startIndex = config.NUM_TRAINING_TRACES
        endIndex   = len(config.DATA_SET)-config.NUM_TESTING_TRACES
    elif config.DATA_SOURCE == 1:
        maxTracesPerWebsiteH = 160
        startIndex = config.NUM_TRAINING_TRACES
        endIndex   = maxTracesPerWebsiteH-config.NUM_TESTING_TRACES
    elif config.DATA_SOURCE == 2:
        maxTracesPerWebsiteH = 18
        startIndex = config.NUM_TRAINING_TRACES
        endIndex   = maxTracesPerWebsiteH-config.NUM_TESTING_TRACES

    seed = config.STARTSEED
    threshold = config.THRESHOLD
    testEnds = False
    accuracyList = []
    errorList  = []

    #seed = 16                 #can be set to 16
    #threshold = 90
    #testEnds = False


    # CSV outputs and others

    csvOutputFileName = ['results',
                         'k' + str(config.BUCKET_SIZE),
                         'c' + str(config.COUNTERMEASURE),
                         'd' + str(config.DATA_SOURCE),
                         'C' + str(config.CLASSIFIER),
                         'N' + str(config.TOP_N),
                         't' + str(config.NUM_TRAINING_TRACES),
                         'T' + str(config.NUM_TESTING_TRACES),
                         'full_cycle_' + str(mytime)
                         ]
    csvFullOutputFileName = ['results',
                         'k' + str(config.BUCKET_SIZE),
                         'c' + str(config.COUNTERMEASURE),
                         'd' + str(config.DATA_SOURCE),
                         'C' + str(config.CLASSIFIER),
                         'N' + str(config.TOP_N),
                         't' + str(config.NUM_TRAINING_TRACES),
                         'T' + str(config.NUM_TESTING_TRACES),
                         ]



    csvOutputFile = os.path.join(config.OUTPUT_DIR, '.'.join(csvOutputFileName))
    csv_file = open(csvOutputFile + '.csv', 'wt')
    csv_full_output = os.path.join(config.OUTPUT_DIR, '.'.join(csvFullOutputFileName))
    csv_Model_output_file = open(csv_full_output + '.csv', 'a')
    infowriter = csv.writer(csv_Model_output_file)


    fullrunOut = open(outputFilename + '_Full.debug','a')

    foutput = open(outputFilename + '.output', 'a')

    print "CSV Seed-Accuracy output file ( Use tail -f filename, to watch seed updates ) : outputs/" + str(csv_file.name)
    print "The overall accuracy is printed at the bottom of system outputs below. "
    print "Starting seeding ... "
    
    # outputs

    while (seed<endIndex ):
        seedlevel2=seed
        accuracy = 100
        fullrunOut = open(outputFilename + '_Full.debug', 'a')

        foutput = open(outputFilename + '.output', 'a')

        while (accuracy>threshold & testEnds==False ):
            startStart = time.time()

            webpageIds = range(0, config.TOP_N - 1)
            random.shuffle( webpageIds )
            webpageIds = webpageIds[0:config.BUCKET_SIZE]


            preCountermeasureOverhead = 0
            postCountermeasureOverhead = 0

            classifier     = intToClassifier(config.CLASSIFIER)
            countermeasure = intToCountermeasure(config.COUNTERMEASURE)

            trainingSetLIb = []
            testingSetLib  = []
            trainingSetVPP = []
            testingSetVPP  = []
            trainingSetJacc = []
            testingSetJacc = []

            targetWebpage = None


            for webpageId in webpageIds:
                if config.DATA_SOURCE == 0:
                    webpageTrain = Datastore.getWebpagesLL( [webpageId], seed-config.NUM_TRAINING_TRACES, seed )
                    webpageTest  = Datastore.getWebpagesLL( [webpageId], seedlevel2, seedlevel2+config.NUM_TESTING_TRACES )
                elif config.DATA_SOURCE == 1 or config.DATA_SOURCE == 2:
                    webpageTrain = Datastore.getWebpagesHerrmann( [webpageId], seed-config.NUM_TRAINING_TRACES, seed )
                    webpageTest  = Datastore.getWebpagesHerrmann( [webpageId], seedlevel2, seedlevel2+config.NUM_TESTING_TRACES )

                webpageTrain = webpageTrain[0]
                webpageTest = webpageTest[0]

                if targetWebpage == None:
                    targetWebpage = webpageTrain

                preCountermeasureOverhead  += webpageTrain.getBandwidth()
                preCountermeasureOverhead  += webpageTest.getBandwidth()

                metadata = None
                if config.COUNTERMEASURE in [config.DIRECT_TARGET_SAMPLING, config.WRIGHT_STYLE_MORPHING]:
                    metadata = countermeasure.buildMetadata( webpageTrain,  targetWebpage )

                i = 0
                for w in [webpageTrain, webpageTest]:
                    for trace in w.getTraces():
                        if countermeasure:
                            if config.COUNTERMEASURE in [config.DIRECT_TARGET_SAMPLING, config.WRIGHT_STYLE_MORPHING]:
                                if w.getId()!=targetWebpage.getId():
                                    traceWithCountermeasure = countermeasure.applyCountermeasure( trace,  metadata )
                                else:
                                    traceWithCountermeasure = trace
                            else:
                                traceWithCountermeasure = countermeasure.applyCountermeasure( trace )
                        else:
                            traceWithCountermeasure = trace

                        postCountermeasureOverhead += traceWithCountermeasure.getBandwidth()
                        instanceLib = LiberatoreClassifier.traceToInstance( traceWithCountermeasure )

                        if instanceLib:
                            if i==0:
                                trainingSetLIb.append( instanceLib )
                            elif i==1:
                                testingSetLib.append( instanceLib )

                        instanceVPP = VNGPlusPlusClassifier.traceToInstance(traceWithCountermeasure)

                        if instanceVPP:
                            if i == 0:
                                trainingSetVPP.append(instanceVPP)
                            elif i == 1:
                                testingSetVPP.append(instanceVPP)

                        instanceJacc = JaccardClassifier.traceToInstance(traceWithCountermeasure)

                        if instanceJacc:
                            if i == 0:
                                trainingSetJacc.append(instanceJacc)
                            elif i == 1:
                                testingSetJacc.append(instanceJacc)

                    i+=1

            ###################

            startClass = time.time()

            # Obtaining Accuracies and website predictions for all traces from the 3 different classifiers.

            [accuracyLib,debugInfoLib] = LiberatoreClassifier.classify( runID, trainingSetLIb, testingSetLib )
            [accuracyVPP,debugInfoVPP] = VNGPlusPlusClassifier.classify( runID, trainingSetVPP, testingSetVPP )
            [accuracyJacc,debugInfoJacc] = JaccardClassifier.classify(runID,trainingSetJacc,testingSetJacc)


            ###
            # best classifier using overall accuracy
            max =0
            maxC = accuracyLib
            if accuracyLib>max:
                max=accuracyLib
                maxC = 'a'
            if accuracyVPP>max:
                max=accuracyVPP
                maxC= 'b'
            if accuracyJacc>max:
                max=accuracyJacc
                maxC= 'c'


            from collections import defaultdict
            libDICT = defaultdict(list)
            vppDICT = defaultdict(list)
            jaccDICT = defaultdict(list)
            liblist = [];
            vpplist = [];
            jacclist = [];
            for entry in debugInfoLib:
                l = (entry[0],entry[1])
                liblist.append(l)
            for k,v in liblist:
                libDICT[k].append(v)
               # add a key-value pair

            for entry in debugInfoVPP:
                l = (entry[0],entry[1])
                vpplist.append(l)
            for k,v in vpplist:
                vppDICT[k].append(v)

            for entry in debugInfoJacc:
                l = (entry[0], entry[1])
                jacclist.append(l)
            for k, v in vpplist:
                jaccDICT[k].append(v)

            bestlist = [];

            totalClass =0;
            totalPrediction=0

            # Voting system implemented as of now  
            for key in libDICT:
                for k in range(0,config.NUM_TESTING_TRACES-1,1):
                    a=str(libDICT[key][k])
                    b=str(vppDICT[key][k])
                    c=str(jaccDICT[key][k])
                    totalClass += 1.0
                    if a.__eq__(b):
                        l = (str(key),a)
                        bestlist.append(l)
                    elif a.__eq__(c):
                        l = (str(key),a)
                        bestlist.append(l)
                    elif b.__eq__(c):
                        l=(str(key),b)
                        bestlist.append(l)
                    else:
                        # most accurate one
                        if maxC == 'a':
                            l = (str(key),a)
                        elif maxC =='b' :
                            l = (str(key),b)
                        elif maxC == 'c':
                            l = (str (key),c)

                        bestlist.append(l)

                    if l[0]==l[1]:
                     totalPrediction += 1.0

            accuracy=totalPrediction/totalClass * 100.0

            end = time.time()

            overhead = str(postCountermeasureOverhead)+'/'+str(preCountermeasureOverhead)

            output = [accuracy,overhead]

            output.append( '%.2f' % (end-startStart) )
            output.append( '%.2f' % (end-startClass) )

            summary = ', '.join(itertools.imap(str, output))


            outputFilenameArray = ['results',
                                   'k' + str(config.BUCKET_SIZE),
                                   'c' + str(config.COUNTERMEASURE),
                                   'd' + str(config.DATA_SOURCE),
                                   'C' + str(config.CLASSIFIER),
                                   'N' + str(config.TOP_N),
                                   't' + str(config.NUM_TRAINING_TRACES),
                                   'T' + str(config.NUM_TESTING_TRACES),
                                   'Seed' + seed.__str__(),
                                   ]

            try:
                csv_file = open(csvOutputFile + '.csv', 'a')
                writer = csv.writer(csv_file)
                if (accuracy < threshold):
                    list2=(seedlevel2.__str__(),accuracy.__str__(),1,config.DATA_SET[seedlevel2])
                    writer.writerow(list2)
                else:
                    #writer.writerow(seedlevel2.__str__() + ',' + accuracy.__str__() + ',sub-seed')
                    list2 = (seedlevel2.__str__(), accuracy.__str__(), 0,config.DATA_SET[seedlevel2])
                    writer.writerow(list2)


            finally:
                print "test-seed accuracy data written for "+seedlevel2.__str__()
                csv_file.close()
#

                # end of csv writing

            accuracyList.append(accuracy)
            errorList.append(100 - accuracy)
            seedlevel2=seedlevel2+config.NUM_TESTING_TRACES
            if(seedlevel2+config.NUM_TESTING_TRACES>endIndex):
                testEnds=True
                break
            if(accuracy<threshold):
                break

        seed=seedlevel2 # Seed is changed each time the loop breaks due to depression in Accuracy below Threshold.

        print 'New seed ( Re-training ).. '+seed.__str__()
        if (testEnds == True):
            break

    # Mean Accuracy and MSE calculations.
    meanAcc = np.mean(accuracyList)
    meanErr = np.mean(errorList)
    Msum = 0
    totalAcc = 0
    for acc in accuracyList:
        Msum = Msum + (acc)
        totalAcc += 1.0

    print "Total accuracy=" + (Msum / totalAcc).__str__()

    Msum = 0
    totalErr = 0
    for err in errorList:
        Msum = Msum + (err - meanErr) * (err - meanErr)
        totalErr += 1.0
        Msum = Msum / 100.0
    MsumRatio = Msum/totalErr
    print "MSE=" + (MsumRatio).__str__()



    try:
        infowriter = csv.writer(csv_Model_output_file)

            #writer.writerow(seedlevel2.__str__() + ',' + accuracy.__str__() + ',sub-seed')
        list2=(str(config.BUCKET_SIZE),totalAcc.__str__(),str(MsumRatio))
        infowriter.writerow(list2)
    finally:
        print "Seed-Accuracy CSV file updated"





    csv_file.close()
#    fdbgJacc.close()
#    fdbgLib.close()
#    fdbgVPP.close()
    csv_Model_output_file.close()




if __name__ == '__main__':
    run()
