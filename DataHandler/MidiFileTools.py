import os
import midi
import sys
import numpy as np

class MidiFileTools(object):
    ''' a class that manipulate midi data '''
    def __init__(self,path):
        '''init a midi data  tool on a given midi file '''
        self.rawPattern = None
        self.timeSignatureNumerator = 4
        self.timeSignatureDenominator = 4
        self.metronome = 24
        self.musicSequence = []
        self.channelPointers = dict()
        self.otherEvents = []
        self.rawPattern =  midi.read_midifile(path)
        self.resolution  = self.rawPattern.resolution
        toAdd = 16 - self.resolution % 16
        self.resolution  += toAdd
        for i,traks in enumerate(self.rawPattern):
            self.musicSequence.append(midi.Track())
            for event in traks: # go over all the tracks and collect the midi events
                if isinstance(event,midi.TimeSignatureEvent):
                    self.musicSequence[i].append(event)
                    self.timeSignatureNumerator = event.get_numerator()
                    self.timeSignatureDenominator = event.get_denominator()
                    self.metronome = event.get_metronome
                elif isinstance(event,midi.SetTempoEvent):
                    # self.tempo = event.get_bpm()
                    self.musicSequence[i].append(event)
                elif isinstance(event,midi.ControlChangeEvent):
                    self.musicSequence[i].append(event)
                elif isinstance(event,midi.NoteEvent):
                    self.musicSequence[i].append(event)
                    self._addToChannel(event.channel,i,event)

                elif isinstance(event,midi.PitchWheelEvent):
                    self.musicSequence[i].append(event)
                elif isinstance(event,midi.EndOfTrackEvent):
                     self.musicSequence[i].append(event)
                else:
                    self.otherEvents.append(event)

    def _addToChannel(self,channel,track,event):
        '''resolution of channel and track dictionary to event that created in that track channel pair
        '''
        if not self.channelPointers.has_key((channel,track)):
            self.channelPointers[(channel,track)] = []
        self.channelPointers[(channel,track)].append(event)

    def getChannelsList(self):
        '''get all the channel types'''
        return self.channelPointers.keys()

    def muteOtherChanelsForEver(self,mainChannel):
        '''select one channel in the midi file and remove all the other from the file ... this can be used if we know
        what channel in the midi file we want to isolate ... can be used to better hear the main melody
        '''
        for (chanle,track),chanSeq in self.channelPointers.iteritems():
            if chanle != mainChannel:
                for i,event in enumerate(chanSeq):

                    if isinstance(event,midi.NoteEvent):
                        chanSeq[i].velocity = 0
        # newSeq=[]
        # for event in self.musicSequence:
        #     if not isinstance(event,midi.NoteEvent):
        #         if

    def muteOtherChanels(self,mainChannel):
        ''' a softer version of previous function it isolate the requested channel without removing or merging the other
        good for continues work with midi file
        '''
        res = []
        for trak in self.musicSequence:
            i = 0
            ticks = 0
            newSequence = midi.Track()
            while i < len(trak):
                if isinstance(trak[i],midi.NoteEvent):
                    if trak[i].channel  != mainChannel:
                        ticks +=  trak[i].tick
                    else:
                        if isinstance(trak[i],midi.NoteOnEvent):
                            temp = midi.NoteOnEvent()
                        else:
                            temp = midi.NoteOffEvent()
                        temp.data = trak[i].data
                        temp.set_pitch(trak[i].get_pitch())
                        temp.set_velocity(trak[i].get_velocity())
                        temp.channel = trak[i].channel
                        temp.tick = ticks +trak[i].tick
                        newSequence.append(temp)
                        ticks = 0
                elif isinstance(self.musicSequence[i],midi.ControlChangeEvent):
                    ticks += self.musicSequence[i].tick
                elif isinstance(self.musicSequence[i],midi.EndOfTrackEvent):
                         # temp = midi.EndOfTrackEvent()
                    newSequence.append(self.musicSequence[i])
                elif isinstance(self.musicSequence[i],midi.TimeSignatureEvent):
                    newSequence.append(self.musicSequence[i])
                elif isinstance(self.musicSequence[i],midi.SetTempoEvent):
                    newSequence.append(self.musicSequence[i])
                else:
                    print "ooops"
                i = 1+i
            res.append(newSequence)
        return res

    def mergeChanels(self,chanelsList):
        '''merge channels to one
        '''
        chanSorted = [j for i,j in chanelsList]
        chanSorted.sort()
        firstChanle = chanelsList[0]
        for key ,chanSeq in self.channelPointers.iteritems():
                for i,event in enumerate(chanSeq):
                    if event.channel and event.channel not in chanSorted:
                        chanSeq[i].channel = firstChanle

    def flatMidiFile(self):
        """
        the must function of the package it recalculate all the tempo and time of events in all the data of the midi
        it flat the different tracks of the midi and create a simple piano midi from the file to learn music from in better way
        also can be used to create note sheet from midi files ... further project ideas
        :return:
        """
        self.rawPattern.make_ticks_abs()
        tempoTrak = []
        for tI,track in enumerate(self.rawPattern):
            for eI, event in enumerate(track):
                if isinstance(event,midi.SetTempoEvent):
                    tempoTrak.append((event.tick,event))
        tempoTrak.sort()
        tempoTrak = [j for i,j in tempoTrak]
        intervalse = []
        lastTempoStartTime = 0
        for i in range(1,len(tempoTrak)):
            intervalse.append((tempoTrak[i-1].tick,tempoTrak[i].tick,self._getTickTime(tempoTrak[i-1]),lastTempoStartTime))
            lastTempoStartTime = ( tempoTrak[i].tick - tempoTrak[i-1].tick ) * self._getTickTime(tempoTrak[i-1]) + lastTempoStartTime
        if len(tempoTrak) > 0:
            intervalse.append((tempoTrak[len(tempoTrak)-1].tick,sys.maxint,self._getTickTime(tempoTrak[len(tempoTrak)-1]),lastTempoStartTime))
        else:
            tempTempo = midi.SetTempoEvent()
            tempTempo.set_bpm(120)
            intervalse.append((0,sys.maxint,self._getTickTime(tempTempo),0))
        combinedTrack = []
        for tI,track in enumerate(self.rawPattern):
            for eI,event in enumerate(track):
                tickTime = 0
                lastTempoStartTime = 0
                lastTempoTick = 0
                for startT,endT,tickInSec,AbsStart in intervalse:
                    if event.tick >= startT and event.tick < endT :
                        tickTime = tickInSec
                        lastTempoStartTime = AbsStart
                        lastTempoTick = startT
                        break
                absTime = ( event.tick - lastTempoTick) * tickTime +  lastTempoStartTime
                combinedTrack.append((absTime ,event))
        combinedTrack.sort()
        combinedTrack = self._filterTrack(combinedTrack)
        #update the Ticks by absolute times
        absoluteTempo = midi.SetTempoEvent()
        absoluteTempo.set_bpm(120)
        TickInTime = self._getTickTime(absoluteTempo)
        prevAbsTime = 0
        for absTime,event in combinedTrack:
            ticks = int((absTime- prevAbsTime) / TickInTime)
            prevAbsTime = absTime
            event.tick = ticks
            if isinstance(event,midi.SetTempoEvent):
                event.set_bpm(120)
        newpatt = midi.Pattern()
        newpatt.resolution = self.resolution
        resTrack = midi.Track()
        absoluteTempo = midi.SetTempoEvent(tick=0)
        absoluteTempo.set_bpm(120)
        resTrack.append(absoluteTempo)
        for abT,ev in combinedTrack:
            ev.channel = 0
            resTrack.append(ev)
        resTrack = self._fixUnreleasedKeys(resTrack)
        resTrack.append(midi.EndOfTrackEvent(tick=1))
        newpatt.append(resTrack)
        # newpatt.make_ticks_rel()
        return newpatt

    def saveMidiToDisc(self,pathToSave,musicSeq = None):
        """
        as it sounds it save a midi file from midi data to disc
        :param pathToSave:
        :param musicSeq:
        :return:
        """
        pattern = midi.Pattern()
        pattern.resolution = self.resolution
        if musicSeq is None:
            for trak in self.musicSequence:
                pattern.append(trak)
        else:
            for trak in musicSeq:
                pattern.append(trak)
        midi.write_midifile(pathToSave, pattern)

    def _fixUnreleasedKeys(self,trackToFix):
        """
        another help functions to fix annoying midi non consistency where the note on even is never released by note off
         or by note on of the same pitch with velocity 0 ...
         based on the idea that there is no longer sound then full note (4 quarter notes in the real piano sounding ...
         well or simplified not to have )
        :param trackToFix:
        :return:
        """
        isDebug = False
        assignedKeys = dict()
        trackToFix.make_ticks_abs()
        newTrack = midi.Track()
        newTrack.tick_relative = False
        for i ,event in enumerate(trackToFix):
            # if i != len(newTrack) :
            #     print 'pause'
            # if i == 53 :
            #     print 'pasue'
            if isinstance(event,midi.NoteEvent):
                if isinstance(event, midi.NoteOffEvent) or event.velocity == 0 :
                    if assignedKeys.has_key(event.pitch):
                        value = assignedKeys.pop(event.pitch)
                        if event.tick - value.tick > self.resolution:
                            event.tick = value.tick + self.resolution -1
                            for i in range(len(newTrack)-1,-1,-1):
                                if newTrack[i].tick < event.tick:
                                    newTrack.insert(i+1,event)
                                    break
                        else:
                             newTrack.append(event)
                    else:
                        newTrack.append(event)
                        if isDebug:
                            print "useless Note"
                elif assignedKeys.has_key(event.pitch):
                    value = assignedKeys.pop(event.pitch)
                    if event.tick - value.tick > self.resolution:
                        turnOfEv = midi.NoteOffEvent()
                        turnOfEv.set_pitch(event.pitch)
                        turnOfEv.tick = value.tick + self.resolution -1
                        for i in range(len(newTrack)-1,-1,-1):
                            if newTrack[i].tick < turnOfEv.tick:
                                newTrack.insert(i+1,turnOfEv)
                                break
                    newTrack.append(event)
                    assignedKeys[event.pitch] = event
                else:
                    assignedKeys[event.pitch] = event
                    newTrack.append(event)
            else:
                newTrack.append(event)
        newTrack.tick_relative = False
        newTrack.make_ticks_rel()
        # for ii,i in enumerate(newTrack):
        #     if i.tick < 0 :
        #         print 'oope'
        return newTrack

    def _filterTrack(self,trackToFilter):
        """
        help function to filter undesired tracks of midi file
        :param trackToFilter:
        :return:
        """
        setTypes = { midi.NoteOnEvent,midi.NoteOnEvent, midi.TimeSignatureEvent}
        filteredTrack = []
        # lastGoodTick = 0
        for t,ev in trackToFilter:
            if type(ev) in setTypes:
                # ev.tick += acumTick
                filteredTrack.append((t,ev))
                ev.tick = 0
            # else:
            #     acumTick +=  ev.tick
        return filteredTrack

    def _getTickTime(self,lastTempo):
        """
        return the relative tick time in fixed tempo ...used to fix ticks in flatten midi file creation
        :param lastTempo:
        :return:
        """
        prevBpm = lastTempo.get_bpm()
        mspb = 60 * 1000000.0 / prevBpm
        tickTime = (mspb / self.resolution) / 1000000.0
        return tickTime


if __name__ == '__main__':
    #
    # pat = MidiFileTools('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\sad\\ASax4BozzaAria.mid')
    # # chanls = pat.getChannelsList()
    # # for chan in chanls:
    # #     neMus = pat.muteOtherChanels(chan)
    # #     pat.saveMidiToDisc('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\test'+str(chan)+'.mid',neMus)
    # chanls = pat.getChannelsList()
    # # for cha in chanls:
    # #     if cha == 4:
    # #         print 'fdfd'
    # #     pat = MidiFile('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\sad\\house_of_the_rising_sun2.mid')
    # #     pat.muteOtherChanelsForEver(cha)
    # #     pat.saveMidiToDisc('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\test'+str(cha)+'.mid')
    #
    # newpet = pat.flatMidiFile()
    #
    #
    # # no = midi.NoteOffEvent()
    # # no.get
    # # pat.mergeChanels(chanls)
    # pat.saveMidiToDisc('C:\\Users\\darkSide\\PycharmProjects\\MusicGuru\\MidiDB\\musicCollection\\testMerge16.mid',newpet)
    print 'dont run me '