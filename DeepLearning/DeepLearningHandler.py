import cPickle as pickle
import datetime
import signal
import numpy as np
import theano, theano.tensor as T
from DataHandler import MidiDataHandler as MDH

class connectionTransformation(theano.Op):
    """
    theano op class to implement the connection of nodes the neuron connections of the rnn
    and the neurons net itself
    """
    __props__ = ()

    def make_node(self, state, time):
        """
        make node ...
        :param state:
        :param time:
        :return:
        """
        state = T.as_tensor_variable(state)
        time = T.as_tensor_variable(time)
        return theano.Apply(self, [state, time], [T.bmatrix()])

    def perform(self, node, inputs_storage, output_storage,params = None):
        """
        Required: Calculate the function on the inputs and put the variables in
        the output storage. Return None.
        """
        state, time = inputs_storage
        beat = MDH.buildBeat(time)
        context = MDH.buildContext(state)
        notes = []
        for note in range(len(state)):
            notes.append(MDH.noteInputForm(note, state, context, beat))
        output_storage[0][0] = np.array(notes, dtype='int8')

def getTrainedPhaseData(modelTemp, dataPieces, batches, name="trainedData"):
    """
    collect and save the temp trained data for further use also good if the machine crashes so u can continue from the
     last checkPoint
    :param modelTemp: the model learned so far
    :param dataPieces: the data set to learn upon
    :param batches: number of batches to run with
    :param name: name of the output p file ... p for properties
    :return: none ... it pickle(serialize) the result
    """
    learnedData = []
    for i in range(batches):
        ipt, opt = MDH.getPieceBatch(dataPieces)
        rnnFlowConnections = modelTemp.updateDatafun(ipt, opt)
        learnedData.append((ipt, opt, rnnFlowConnections))
    pickle.dump(learnedData, open('output/' + name + '.p', 'wb'))

def generatMusicFunction(modelTemp, pcs, times, keepDataTempLearning=False, name="final",destName = 'output\\'):
    """
    as the name sugest this function used to compose music from the trained model
    :param modelTemp:  the trained model to learn from
    :param pcs:  the data pices to generet music with
    :param times: the data that give the times to the model building phase
    :param keepDataTempLearning: is to keep data trough the learning ..make it havier to learn
    :param name: the name of the file to generate
    :return:
    """
    md = MDH.MidiDataHandler()
    xIpt, xOpt = map(lambda x: np.array(x, dtype='int8'), MDH.splitToSegments(pcs))
    all_outputs = [xOpt[0]]
    if keepDataTempLearning:
        allDataUpdate = []
    modelTemp.initSlowLearning(xIpt[0])
    cons = 1
    for time in range(MDH.batchLength * times):
        resdata = modelTemp.slowFunction(cons)
        nnotes = np.sum(resdata[-1][:, 0])
        if nnotes < 2:
            if cons > 1:
                cons = 1
            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3
        all_outputs.append(resdata[-1])
        if keepDataTempLearning:
            allDataUpdate.append(resdata)
    md.DataMatrixToMidiFile(np.array(all_outputs), destName + name)
    if keepDataTempLearning:
        pickle.dump(allDataUpdate, open(destName + name + '.p', 'wb'))

def trainDataPart(modelTemp,pieces,epochs,start=0,destLoctation = 'output\\' ):
    """
    initiate the training sequence a pseudo epoch like process of learning in the deep learning
    this iterative process initiate the rnn learning procedure of foreword propagation and back propagation in which the
     weights between the layers of the deep learning being updated
    :param modelTemp: the model to train Continue train the model allowes us to continue training an existent model
    so we can create even smarter net in the future.
    :param pieces: the data to run trough the net with
    :param epochs: the number of epochs / iterations to train upone
    :param start: if we continue to train from a given point (not really important past logical continuation of the process)
    :return:
    """


    mh = MDH.MidiDataHandler()
    stopflag = [False]
    def signalWorker(signame, sf):
        stopflag[0] = True
    old_handler = signal.signal(signal.SIGINT, signalWorker)
    prevTime = datetime.datetime.now()
    with open(destLoctation+"logFile.txt", "w") as text_file:
        for i in range(start,start+epochs):
            currTime = datetime.datetime.now()
            total_time=(currTime -prevTime)
            prevTime = currTime
            if stopflag[0]:
                break
            error = modelTemp.updateFunction(*MDH.getPieceBatch(pieces))
            # if i % 10 == 0:
            print "epoch {}, time  {}, error  {}".format(i,total_time.total_seconds(),error)
            text_file.write("epoch {}, time {}, error {}".format(i,total_time.total_seconds(),error))
            if i % 50 == 0 or (i % 20 == 0 and i < 100):
                xIpt, xOpt = map(np.array, MDH.splitToSegments(pieces))
                mh.DataMatrixToMidiFile(np.concatenate((np.expand_dims(xOpt[0], 0), modelTemp.predict_fun(MDH.batchLength, 1, xIpt[0])), axis=0),'output/sample{}'.format(i))
                pickle.dump(modelTemp.learned_config,open(destLoctation+'params{}.p'.format(i), 'wb'))
    signal.signal(signal.SIGINT, old_handler)

