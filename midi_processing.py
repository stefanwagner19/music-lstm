import pretty_midi
import numpy as np
import math


def process_midi(path):

	# opens midi file and returns notes
	# returns numpy array of notes

	# load midi file and get beats
	file = pretty_midi.PrettyMIDI(path)

	beat = file.get_beats()[1]

	# create placeholder array
	con = np.array([])

	# run through all notes played
	for j in range(len(file.instruments)):

		# ignores drums
		if file.instruments[j].is_drum == False:

			# temporary array for notes per instrument
			temp = np.zeros((len(file.instruments[j].notes), 3))

			for i, note in enumerate(file.instruments[j].notes):

				pitch = note.pitch
				start = note.start
				end = note.end
				
				# store note info temporarily
				temp[i] = [pitch, start, end]

			# save all note info
			con = np.append(con, temp)

	# reshape and sort based on note start time
	con = con.reshape(-1,3)
	n = con[con[:,1].argsort()]

	# for j in n:
	# 	print(j)

	# turn index 1 to time until next note played and index 2 to note duration
	for i, note in enumerate(n):

		duration = (note[2] - note[1])/beat*16

		# indicator for end of song
		if i+1 == len(n):
			n[i,1] = 420

		# assign time until next note
		else:
			pause_to_next = n[i+1,1] - note[1]
			n[i,1] = pause_to_next/beat*16

		n[i,2] = duration

	for i in range(len(n)):

		n[i][2] = round_duration(n[i][2])

		if n[i][1] == 420:
			n[i][1] = n[i,2]

		else:
			n[i][1] = round_pause(n[i][1])

	# for j in n:
	# 	print(j)
	q = np.where(n[:,1]==0)

	anquor = -2
	prev = -2
	
	for i in q[0]:
		if i-1 != prev:
			if anquor == -2:
				anquor = i
			else:
				y = n[anquor:prev+2]
				z = y[y[:, 0].argsort()]
				y[:, (0, 2)] = z[:, (0, 2)]
				n[anquor:prev+2, (0,2)] = y[:,(0,2)]
				anquor = i
		prev = i

	y = n[anquor:prev+2]
	z = y[y[:, 0].argsort()]
	y[:, (0, 2)] = z[:, (0, 2)]
	n[anquor:prev+2, (0,2)] = y[:,(0,2)]


	return n.astype(float)


def round_pause(x):

	# used for rounding the pauses to appropriate values
	# returns customly rounded value

	if x <= 0.5:
		return 0
	elif x <= 1.5:
		return 1
	elif x <= 16:
		return round(x/2)*2
	elif x <= 32:
		return round(x/4)*4
	elif x < 128:
		return round(x/16)*16
	else:
		return 128


def round_duration(x):

	# used for rounding the durations appropriately
	# returns customly rounded value

	if x <= 16:
		return 16
	elif x < 128:
		return pow(2 ,round(math.log(x)/math.log(2), 0))
	else:
		return 128


def notes_to_text(notes):

	# converts notes to text in tokenized form
	# returns list of strings

	array = []
	num_notes = len(notes)

	beat_count = 32

	for i, note in enumerate(notes):

		n = "n" + str(int(note[0]))
		d = "d" + str(int(note[2]))

		array.extend((n, d))


		p = "p" + str(int(note[1]))
		array.append(p)
		beat_count -= int(note[1])
		if beat_count <= 0:
			beat_count += 32
			array.append(".")

	return array


def text_to_notes(text):
	# converts tokenized text to individual note arrays taking into account possible errors
	# returns numpy array of notes

	# shows what type of note is to be expected next
	# -1 indicates start of sequence
	indicator = -1

	# stores all notes
	notes = []

	while (len(text) != 0) and (text[0] != "END"):
		# expecting pitch
		if indicator <= 0:

			# add note to other notes
			if indicator != -1:
				notes.append(note)

			# create new note array
			note = []

			# if it isn't a note remove it
			if text[0][0] != "n":
				text.pop(0)
		
			else:
				note.append(int(text[0][1:]))
				text.pop(0)
				indicator = 1

		# expecting duration
		elif indicator == 1:

			if text[0][0] != "d":
				note.append(16)

			else:
				note.append(int(text[0][1:]))
				text.pop(0)

			indicator = 2

		# expecting pause or next note
		elif indicator == 2:

			if text[0][0] != "p":
				indicator = 0

			else:
				note.append(int(text[0][1:]))
				text.pop(0)

			indicator = 0 

	if len(note) >= 2:
		notes.append(note)

	n = []

	# add pause attribute to all notes and flip duration and pause values
	for i in notes:
		if len(i) != 0:
			n.append(i)

	for note in n:

		if len(note) == 2:
			note.append(0)

		note[1], note[2] = note[2], note[1]

	return np.array(n).astype(float)


def write_midi(n, beat=0.5):

	# writes midi oject for later file writing
	# returns PrettyMIDI object

	pm = pretty_midi.PrettyMIDI()
	instrument = pretty_midi.Instrument(program=0)

	start = 0.0

	for note in n:

		end = note[2]*beat/16 + start

		pm_note = pretty_midi.Note(
			pitch=int(note[0]),
			start=start,
			end=end,
			velocity=100
		)
		start = start + note[1]*beat/16

		instrument.notes.append(pm_note)

	pm.instruments.append(instrument)
	return pm


def create_token_dict():

	# creates a dictionary containing all possible tokens and their values
	# returns token dictionary
	dic = {".":0}
	count = 1

	# create note entries
	for i in range(21, 109):
		key = "n" + str(i)
		dic[key] = count
		count += 1
	
	# create pause entries
	dic["p1"] = count
	count += 1

	for i in range(0, 17):
		if i % 2 == 0:
			key = "p" + str(i)
			dic[key] = count
			count += 1

	for i in range(20, 33):
		if i % 4 == 0:
			key = "p" + str(i)
			dic[key] = count
			count += 1

	for i in range(48, 129):
		if i % 16 == 0:
			key = "p" + str(i)
			dic[key] = count
			count += 1

	#create duration entries
	x = 16
	while x <= 128:
		key = "d" + str(x)
		dic[key] = count
		x *= 2
		count += 1

	# dic["END"] = count

	return dic


def create_reverse_dict():
	
	token_dict = create_token_dict()
	dic = {}

	for i, key in enumerate(token_dict.keys()):
		dic[i] = key

	return dic


def tokenize(arr):

	dic = create_token_dict()
	x = np.zeros(len(arr))

	for i in range(len(arr)):
		x[i] = dic[arr[i]]

	return x.astype(int)


def untokenize(arr):

	arr = arr.astype(int)

	dic = create_reverse_dict()
	x = np.empty(len(arr)).astype(str)

	for i in range(len(arr)):
		x[i] = dic[arr[i]]

	return x