# MusicGuru-RNN-Composer
An assignment in a course to develop an RNN with Theano to make the computer learn to compose music in midi format.

### An homework assignment in a course to develop a Recurrent neural network that can create new data after training.
 We trained a Long Short-Term Memory (LSTM) Recurrent Neural Network on different Data sets of midi files. Afterward we sampled from the created model to generate new musical pieces.
 This report will explain the work that was done by use. To create an RNN using Theano to compose music.

#### the Code itself is in the files in this directory and we will explain further what is what in there

### The authors of this work are : 
* Ilya Simkin, id : 305828188
* Or May-Paz, id : 301804134

# Introduction
Recurrent Neural Networks (RNNs) are popular models that have shown great promise in many NLP tasks. but despite their recent popularity in the field of text analysis they can be used in many other fields like the use google dose with them in the field of computer vision.
in this work we are trying to train the computer to compose music which is as well a trainable thing just as painting or writing correctly.

This post assumes a familiarity with machine learning and neural networks. For a good overview of RNN's, I highly recommend reading Andrej Karpathy's excellent blog post [Here](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) for an in-depth explanation.

# Data Handeling
As mentioned before our project is aiming to work with Musical data so first of all we needed to find a proper digital music format to work with. Obviously we stopped our choice on midi files due the fact that they the best representation of digital music, other choices were the more popular and commonly used mp3 files but those files weight much more and have much more analogical structure. We wanted to work with something closer to sheet music even thought of creating a computer vision solution to scan and translate real sheet music of notes but it would add much unneeded complexity. 

### So a little about midi files: 

If you give a look in [Wikipedia](https://en.wikipedia.org/wiki/MIDI): 
MIDI (/ˈmɪdi/; short for Musical Instrument Digital Interface) is a technical standard that describes a protocol, digital interface and connectors and allows a wide variety of electronic musical instruments, computers and other related devices to connect and communicate with one another…

Basically a normal midi file is a set of event codded in binary with commands that create a set of notes. 
there are many type of Events part of them have no connection to the music in the file or even has the lyrics of the song but there is notes events as well.

![Alt text](https://raw.githubusercontent.com/Ilya-Simkin/MusicGuru-RNN-Composer/master/images/midiEvents.JPG "Midi event flow example")
(midi event flow example)

•	Each note event has a Velocity which is the analogical form of how hard you would hit a piano key (the faster it will be louder)
•	As well each note Event has the pitch value. It is translated to how low or high is the sound, such that lower pitch means lower note in terms of sound frequency.
![Alt text](https://raw.githubusercontent.com/Ilya-Simkin/MusicGuru-RNN-Composer/master/images/pianopitchMidi.jpg "Midi Pitch To piano Note")(Midi Pitch To piano Note)

Due to the fact that we want to learn something that close to one instrument music we had to transform all the data to a pitch and velocity common for piano playing .

### Another important information we needed to think about is the time context;
Each event in the midi file has a tick count which is the delay between the execution of the event and the previous event. But the tricky thing is that tick is not an Absolut time unit it is relative to 2 things:
•	The tempo which is the "how fast ticks should pass"
•	And the resolution which is how much tempo units there is in a second.
This was a major thing because that meant that one midi file can have many different areas in which the "time of the note" is different from each other.

So one of the hardest thing we had to do is to resample all the midi file to be in one tempo and resolution and recalculate all the time stamps of the event so the music won't change.

### The last bad thing we had to work around in our music data set is multi track and multi-channel:
Midi files may have many different instruments which play on different channels, even worst then that is the fact that music in midi file may come in different tracks the equivalent of few files played together.
We had to write an algorithm that will flatten the file in a way that it will be played in a way equivalent to one piano.  (Maybe one will need 5 hands to play it in the right way, but that’s not our problem ;-)  ) .
All the data handling was written in python and contained in the object:  MidiFileTools under DataHandler/MidiFileTools this class is the handler of one midi file, while depended on great package called "python-midi" which was a great help in reading and parsing a binary midi file into a set of events.

## Music Language Modeling

After we done to working with the midi file we need to transfer it to some thing that theano can work with.
So the next step is music Language modeling, it is the problem of modeling symbolic sequences of polyphonic music in a completely general piano roll representation. Piano roll representation is a key distinction here, meaning we're going to use the symbolic note sequences as represented by sheet music, as opposed to more complex, acoustically rich audio signals. MIDI files are perfect for this, as they encode all the note information exactly to how it would be displayed on a piano roll.

The most straightforward way to learn this way is to discretize a piece of music into uniform time steps. There are 88 possible pitches in a MIDI file that represent a piano, so every time step is encoded into an 88-dimensional binary vector as shown below. A value of 1 at index i indicates that pitch i is playing at a given time step. also for each point there is an indication if the Note was just pressed or is continuously pressed in the given time step from before. so we basically have 2 binary numbers in each note cell.

![Alt text](https://raw.githubusercontent.com/Ilya-Simkin/MusicGuru-RNN-Composer/master/images/pianoToMatrix.jpg "piano Note to matrix ")(piano Note to matrix)

We chose to set the time frame intervals in a way that will give us the option to sample notes in length as small as "one eighth" notes which is not the shortest actually used in playing piano but will represent melodies good enough.

To make things make more sense we used an idea of a similar works we saw online, the extra data that is added to each time frame matrix row will give context to the played notes and make them combine melodies that will be nicer for human ears :
 
•	Position: The MIDI note value of the current note. Used to get a vague idea of how high or low a given note is, to allow for differences (like the concept that lower notes are typically chords, upper notes are typically melody).
•	Pitch class : extra 12 features added that will be 1 at the position of the current note, starting at 'Do' for 0 and increasing by 1 per half-step, and 0 for all the others. Used to allow selection of more common chords (i.e. it's more common to have a C major chord than an E-flat major chord)
•	Previous Vicinity: extra features that gives context for surrounding notes in the last time steps, one octave in each direction. The value at index 2(i+12) is 1 if the note at offset i from current note was played last time step, and 0 if it was not. The value at 2(i+12) + 1 is 1 if that note was articulated last time step, and 0 if it was not. (So if you play a note and hold it, first time step has 1 in both, second has it only in first. If you repeat a note, second will have 1 both times.)
•	Previous Context : Value at index i will be the number of times any note x where (x-i-pitch class) mod 12 was played last time step. Thus if current note is "Do" and there were 2 "Mi" at the  last time step, the value at index 4 (since "Mi"  is 4 half steps above "Do" ) would be 2.
•	Beat : Essentially a binary representation of position within the measure, assuming 4/4 time (due to the fact that we pretty much flatten the music to be 4/4 in the previse steps of midi flattening  ). With each row being one of the beat inputs, and each column being a time step, it basically just repeats the following a constant petter.

A trained model outputs the conditional distribution of notes at a time step, given the all the time steps that have occurred before it. One problem with this naive formulation is that the amount of potential note configurations is too high (2N for N possible notes) to take the softmax classification approach normally language modeling.
Instead, we found works that used something called a sigmoid cross-entropy loss function to predict the probability of whether each note class is active or not separately.


