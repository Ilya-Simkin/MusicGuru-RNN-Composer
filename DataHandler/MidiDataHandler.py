import os
import itertools
import midi
import numpy as np
from DataHandler.MidiFileTools import MidiFileTools
import os, random

__author__ = 'ilya simkin'

batch_width = 10 # number of sequences in a batch
batchLength = 16*4 # length of each sequence
breakSize = 64 # interval between possible start locations
NotesLowBound = 24 # do not change the pitch of the first note
NotesUpperBound = 104 # do not change, the pitch of the max note

class MidiDataHandler(object):

    def loadMidiData(self,dirpath):
        '''
        loads a midi files from directory check them for being midi files and transform them to data to use in learn process

        :param dirpath:  directory where the midi files at
        :return: midi music data set in matrix of data of time stamp checked music
        '''
        isDebug = False
        songsDict = {}
        lis = os.listdir(dirpath)
        for fname in lis:
            if fname[-4:] not in ('.mid','.MID'):
                continue
            name = fname[:-4]
            if isDebug:
                print "starting {}".format(name)
            outMatrix = self.midiFileToDataMatrix(os.path.join(dirpath, fname))
            if len(outMatrix) < batchLength:
                continue
            songsDict[name] = outMatrix
            if isDebug:
                print "Loaded {}".format(name)
        return songsDict

    def flattenDirectory(self,sourceDirectory,DestDirectory):
        '''
        take a normale multi tempo multi track midi data diractory an makes a flatten one track one tempo midi file from
        each midi file for further use
        :param sourceDirectory: source dir
        :param DestDirectory: destination dir for flatten midi files in piano flatten format
        :return:return the data set made
        '''
        isDebug = False
        files = os.listdir(sourceDirectory)
        assert len(files) > 0, 'No data!'
        dataSet = []
        for f in files:
            try:
                if isDebug:
                    print('File {}'.format(str(f)))
                ptol = MidiFileTools(sourceDirectory+f)
                newpet = ptol.flatMidiFile()
                ptol.saveMidiToDisc(DestDirectory+'Flatten'+f,newpet)
                dataSet.append(newpet)
            except IOError:
                print('someThing went very wrongly')
                pass
        return dataSet

    def process_dataset(self,path, outname=None, min_length=500):
        """loads a midi files and make a data set out of the files so we can learn from it with theano
        saves the data for further use and return it as well
        """
        isDebug = False
        files = os.listdir(path)
        assert len(files) > 0, 'No data!'
        dataSet = []
        for f in files:
            try:
                if isDebug:
                    print('File {}'.format(str(f)))
                matrix = self.midiFileToDataMatrix(path + f)
                if isDebug:
                    self.DataMatrixToMidiFile(matrix,name='resTored'+str(f))
                statemat = np.array(matrix)
                piano_roll = self.pianNotesFromData(statemat).astype(np.float32)
                if piano_roll.shape > min_length:
                    dataSet.append(piano_roll)
                else:
                    print('Song too short... skipping this one.')
                    break

            except IOError:
                print('someThing went very wrongly')
                pass
        if outname is not None:
            np.savez(outname, dataSet)
            print('File saved as {}.npz'.format(outname))
        return dataSet

    def midiFileToDataMatrix(self,midifile):
        '''
        reads a single midi file and transform it to data matrix
        it also skips notes that are out of bound of the min and max pitch level
        :param midifile: path to midi file
        :return: matrix of assigned notes in a given time frame
        '''
        isDebug = False
        pattern = midi.read_midifile(midifile)
        timeleft = [0]
        posns = [0]
        statematrix = []
        span = NotesUpperBound - NotesLowBound
        time = 0
        state = [[0,0] for x in range(span)]
        statematrix.append(state)
        while True:
            if time % (pattern.resolution / 8) == (pattern.resolution / 16):
                # Crossed a note boundary. Create a new state, defaulting to holding notes
                oldstate = state
                state = [[oldstate[x][0],0] for x in range(span)]
                statematrix.append(state)
            for i in range(1):
                while timeleft[i] == 0:
                    track = pattern[i]
                    pos = posns[i]
                    evt = track[pos]
                    if isinstance(evt, midi.NoteEvent):
                        if (evt.pitch < NotesLowBound) or (evt.pitch >= NotesUpperBound):
                            if isDebug:
                                 print "Note {} at time {} out of bounds (ignoring)".format(evt.pitch, time)
                        else:
                            if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
                                state[evt.pitch - NotesLowBound] = [0, 0]
                            else:
                                state[evt.pitch - NotesLowBound] = [1, 1]
                    elif isinstance(evt, midi.TimeSignatureEvent):
                        if evt.numerator not in (2,3, 4,6):
                            # We don't want to worry about non-4 time signatures. Bail early!
                            if isDebug:
                                print midifile
                                print "Found time signature event {}. Bailing!".format(evt)
                            return statematrix
                    try:
                        timeleft[i] = track[pos + 1].tick
                        posns[i] += 1
                    except IndexError:
                        timeleft[i] = None

                if timeleft[i] is not None:
                    timeleft[i] -= 1

            if all(t is None for t in timeleft):
                break

            time += 1
        # if isDebug:
        #     for i in statematrix:
        #         e = []
        #         for ind,j in enumerate(i) :
        #             if j[0] !=0 or j[1] != 0 :
        #                 e.append(ind)
        #         print  e
        return statematrix

    def DataMatrixToMidiFile(self,statematrix, name="example"):
        '''
        take a data matrix of note pressed in a given time frame and create a legal midi file which is saved to a given path
        :param statematrix: data matrix
        :param name: file name or path to save the midi file
        :return:the midi file pattern to work with
        '''
        isDebug = False
        statematrix = np.asarray(statematrix)
        pattern = midi.Pattern()
        pattern.resolution = 224
        track = midi.Track()
        pattern.append(track)
        span = NotesUpperBound - NotesLowBound
        tickscale = 28
        lastcmdtime = 0
        prevstate = [[0,0] for x in range(span)]
        for time, state in enumerate(statematrix + [prevstate[:]]):
            offNotes = []
            onNotes = []
            for i in range(span):
                n = state[i]
                p = prevstate[i]
                if p[0] == 1:
                    if n[0] == 0:
                        offNotes.append(i)
                    elif n[1] == 1:
                        offNotes.append(i)
                        onNotes.append(i)
                elif n[0] == 1:
                    onNotes.append(i)
            for note in offNotes:
                track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note + NotesLowBound))
                lastcmdtime = time
            for note in onNotes:
                track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=72, pitch=note + NotesLowBound))
                lastcmdtime = time
            prevstate = state
        eot = midi.EndOfTrackEvent(tick=1)
        track.append(eot)
        midi.write_midifile("{}.mid".format(name), pattern)
        return pattern

    def pianNotesFromData(self,statemat):
        '''
        take a data matrix frame and check the piano rolls in it such as if a key was just pressed and so on
        :param statemat:
        :return:
        '''
        song_length = statemat[:, :, 0].shape[0]
        nnotes = statemat[:, :, 0].shape[1]
        strikes = statemat[:, :, 1].astype(np.int64)
        holds = statemat[:, :, 0].astype(np.int64)
        pianoroll = np.zeros((2 * song_length, nnotes), dtype=np.int64)
        for n in range(nnotes):
            for t in range(2 * song_length - 2):
                t_orig = t // 2
                next_strike = strikes[t_orig + 1, n]
                current_hold = holds[t_orig, n]
                if t % 2 == 0 and current_hold == 1:
                    pianoroll[t, n] = 1
                if t % 2 == 1 and current_hold == 1 and next_strike == 0:
                    pianoroll[t, n] = 1
        return pianoroll

    def dataFromPianoNotes(self,pianoroll):
        '''
        return a data frame from a piano roll so it can be translated to data set or writen to midi file
        :param pianoroll: information about the given time frame of the notes pressed and so on
        :return:
        '''
        assert pianoroll.shape[0] % 2 == 0, 'Error: not an even length piano roll!'
        song_length = pianoroll.shape[0]
        nnotes = pianoroll.shape[1]
        strikes = np.zeros((song_length / 2, nnotes), dtype=np.int64)
        holds = np.zeros((song_length / 2, nnotes), dtype=np.int64)
        for n in range(nnotes):
            # first note:
            value = pianoroll[0, n]
            if value == 1:
                holds[0, n] = 1
                strikes[0, n] = 1
            for t in range(1, song_length / 2):
                change = pianoroll[2 * t, n] - pianoroll[2 * t - 1, n]
                value = pianoroll[2 * t, n]
                if change == 0 and value == 1:
                    holds[t, n] = 1
                if change == 1 and value == 1:
                    holds[t, n] = 1
                    strikes[t, n] = 1
        return np.concatenate((holds[:, :, None], strikes[:, :, None]), axis=-1).astype(np.int64)

def splitToSegments(pieces):
    '''
    a function the get an random piece of music data so that the rnn can learn from it
    it create a set of the configurable length to imitate a logical piece of music data each time
    (or a set of logical pieces)
    :param pieces: the data to get the samples from
    :return: set of samples or segments of data of the music
    '''
    isDebug = False
    randPieces = random.choice(pieces.values())
    length = len(randPieces)
    start = random.randrange(0,length-batchLength,breakSize)
    if isDebug:
        print "Range is 0 - "+ str( length - batchLength )+ " " + str( breakSize ) + " -> " + str( start )
    segOut = randPieces[start:start+batchLength]
    segIn = noteStateMatrixToTheanoInput(segOut)
    return segIn, segOut

def getPieceBatch(pieces):
    '''
    just like the previase function but create a batch of data of samples ... also explained above
    :param pieces:
    :return:
    '''
    isDebug = False
    res = []
    for i in range(batch_width):
        #choose a random song out of the midi files
        randSongD = random.choice(pieces.values())
        length = len(randSongD)
        #choose a random line to start from
        start = random.randrange(0,length-batchLength,breakSize)
        if isDebug:
            print "Range is 0 - "+ str( length - batchLength )+ " " + str( breakSize ) + " -> " + str( start )
        segOut = randSongD[start:start+batchLength]
        segIn = noteStateMatrixToTheanoInput(segOut) # prepering data to be input to theano
        res.append((segIn, segOut))
    i,o = zip(*res)
    return np.array(i), np.array(o)

def buildContext(state):
    '''
    add a context of what notes are being played in a given point of time based on the fact that there are 12 notes
    in a piano octave including the "black keys"
    :param state:
    :return:
    '''
    context = [0]*12
    for note, notestate in enumerate(state):
        if notestate[0] == 1:
            noteNameClass = (note +NotesLowBound) % 12  # set which note this is like 'do' 're' 'mi' and sharps
            context[noteNameClass] += 1
    return context

def buildBeat(time):
    '''
    calculate the beat of the data chunk as 4 ticks of the metronom ...used as an extra feature to the theano learning process
    :param time: the time frame of the given note (how much from the beginning it time frame is )
    :return: a pseudo beat feature
    '''
    res = []
    for x in (time%2, (time//2)%2, (time//4)%2, (time//8)%2 ):
        res.append(2*x-1)
    return res

def noteInputForm(note, state, context, beat):
    '''
    combine the data to a single data frame from all the extracted features
    pluse think of the notes as 2 grams to add the calculation of connections between followed notes data
    :param note:
    :param state:
    :param context:
    :param beat:
    :return:
    '''
    part_position = [note]
    noteNameClass = (note + NotesLowBound) % 12
    part_noteNameClass = [int(i == noteNameClass) for i in range(12)]
    # Concatenate the note states for the previous vicinity
    tempList = []
    for i in range(-12, 13):
        if len(state) >  note + i  :
            tempList.append(state[note + i])
        else:
            tempList.append([0,0])
    pp = itertools.chain.from_iterable(tuple(tempList))
    part_prev_vicinity = list(pp)
    part_context = context[noteNameClass:] + context[:noteNameClass]
    return part_position + part_noteNameClass + part_prev_vicinity + part_context + beat + [0]

def noteStateMatrixToTheanoInput(statematrix):
    '''
    concat the data to get input to theano learning ... the final form of data that will eventually go to the rnn
    :param statematrix:
    :return:
    '''
    res = []
    for time,state in enumerate(statematrix): #iterate over the chunk of data
        beat = buildBeat(time)
        context = buildContext(state)
        notes = []
        for note in range(len(state)):
            thType = noteInputForm(note, state, context, beat)
            notes.append(thType)
        res.append(notes)
    return res



if __name__ == '__main__':
    # midiHandler = MidiDataHandler()
    # filePath = 'C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\DeepLearning\\midi\\'
    # filedest = 'C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\DeepLearning\\midi\\'
    # fileS = 'C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\DataHandler\\tests\\'
    # data = midiHandler.flattenDirectory(filePath,filedest)
    # print 'done'
    # filePath = 'C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\sad\\fogdew.mid'
    # data_path  = 'C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\DataHandler\\tests\\
    # ptol = MidiFileTools('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\Rigions\\Irish\\sad\\johnny.mid')
    # newpet = ptol.flatMidiFile()
    # ptol.saveMidiToDisc('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\flatten\\johnnyFlat.mid',newpet)
    # 
    # dataset_temp  = midiHandler.process_dataset(fileS,outname='foggyTest')
    # # for i in dataset_temp[0]:
    # #     print np.where(i==1)
    print 'sdasd'
