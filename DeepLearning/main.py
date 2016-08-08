import cPickle as pickle
import os
from random import Random

import numpy as np
from DeepLearning import model
import DataHandler.MidiDataHandler as MDH
from DeepLearning.DeepLearningHandler import generatMusicFunction, trainDataPart

def calculateSimilarity(testedFile,sourcePath,chuckSize,randomSamples = None):
    print 'start calculating similarity may take long time'
    from scipy.spatial import distance
    midiHandler = MDH.MidiDataHandler()
    testedFile = midiHandler.midiFileToDataMatrix(testedFile)
    sorceSongsDict = midiHandler.loadMidiData(sourcePath)
    testedChunkCount = len(testedFile) - chuckSize
    if testedChunkCount <= 0:
        raise IOError('input size is bigger then chunk size')
    if randomSamples is not None:
        testedChunkCount = np.random.randint(0,testedChunkCount,min(randomSamples,len(testedFile)) ).tolist()
    else :
        testedChunkCount = range(testedChunkCount)

    similarity = np.ones((len(testedChunkCount),1))

    for i , testedChunkStrat in enumerate(testedChunkCount):
        Aflat =  np.array(testedFile[testedChunkStrat : min( testedChunkStrat+chuckSize,len(testedFile))]).flatten()
        minDist = 9999
        for song in sorceSongsDict.values():
            songChunkCount = len(song) - chuckSize
            tempRes = np.ones((songChunkCount,1))
            for songChunkStart in range(songChunkCount):
                Bflat =np.array(song[songChunkStart:min(songChunkStart+chuckSize,len(song))]).flatten()
                tempRes[songChunkStart] = distance.cosine( Aflat, Bflat )
            minDist = min(minDist,np.min(tempRes))
        similarity[i] = minDist
    return (1-np.average(similarity)) *  100

if __name__ == '__main__':
    midiHandler = MDH.MidiDataHandler()
    filePath = '.\\midi\\'
    filedest = '.\\flattnMidi\\'
    modelDest = '.\\output\\'
    genFile = "composition8"
    if os.listdir(filedest) == []:
        data = midiHandler.flattenDirectory(filePath,filedest)
    songsDic = midiHandler.loadMidiData(filedest)
    m = model.Model([300, 300], [100, 50] ,MDH.NotesLowBound ,MDH.NotesUpperBound, dropout=0.5)
    trainDataPart(m, songsDic, 10)
    pickle.dump(m.learned_config, open(modelDest+"final_learned_config.p", "wb"))
    generatMusicFunction(m,songsDic,10,name=genFile)
    print 'calculation similarity started '
    sim = calculateSimilarity(modelDest+genFile+'.mid',filedest,64,randomSamples = 50)
    with open(modelDest+"logFile.txt", "w") as text_file:
        text_file.write('calculation similarity started ')
        text_file.write('similrity : ' + str(sim) + " %")
    print 'similrity : ' + str(sim) + " %"