import torch as T 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os

from generator import Generator 
from discriminator import Discriminator 
import midi_processing as mpr 


def generate(gen, dis, length, path, name):
	gen.eval()
	dis.eval()

	mid = mpr.process_midi(path)

	text = mpr.notes_to_text(mid)
	final = mpr.tokenize(text[:80])

	possible_next_notes = []
	maxim = 0
	max_index = 0

	while len(final) < length:

		possible_next_notes = []
		maxim = 0
		max_index = 0

		for i in range(6):

			gen_inp = T.from_numpy(final).to("cuda:0", dtype=T.long)

			next_note = 1

			while next_note != 0:
				next_note = gen(gen_inp[-60:].unsqueeze(0)).squeeze()
				next_note = gen.choose_max(next_note).cpu()
				gen_inp = gen_inp.cpu()
				gen_inp = np.append(gen_inp, next_note)
				gen_inp = T.from_numpy(gen_inp).cuda()

			if len(gen_inp) < 120:
				rating = dis(gen_inp.unsqueeze(0))
			else:
				rating = dis(gen_inp[-120:].unsqueeze(0))

			if rating > maxim:
				max_index = i
				maxim = rating

			possible_next_notes.append(gen_inp.squeeze())

		final = possible_next_notes[max_index].detach().cpu().clone().numpy().astype(np.int)

	converted = mpr.text_to_notes(mpr.untokenize(final).tolist())
	file = mpr.write_midi(converted, 0.5)
	file.write(f"FinalResults\\{name}.mid")


dict_size = 113
DEVICE = T.device("cuda:0" if T.cuda.is_available() else "cuda:1")

dis_emb_size = 4
dis_hidden_size = 128
dis_n_layers = 2
dis_dense_size = 64
dis_dropout = 0.1
dis_model_path = ".\\models\\dis_models\\lstm_epoch_150.pth"

gen_emb_size = 4
gen_hidden_size = 128
gen_n_layers = 2
gen_dense_size = 64
gen_dropout = 0.3
gen_model_path = ".\\models\\gen_models\\lstm_epoch_150.pth"

generator = Generator(dict_size=dict_size, emb_size=gen_emb_size, hidden_size=gen_hidden_size, n_layers=gen_n_layers, \
		dense_size=gen_dense_size, dropout=gen_dropout, device=DEVICE)

try:
	checkpoint = T.load(gen_model_path)
	generator.load_state_dict(checkpoint['model_state_dict'])
	print("Generator Load Successful\n")

except:
	print("Generator Failed Loading\n")


discriminator = Discriminator(dict_size=dict_size, emb_size=dis_emb_size, hidden_size=dis_hidden_size, n_layers=dis_n_layers, \
		dense_size=dis_dense_size, dropout=dis_dropout, device=DEVICE)

try:
	checkpoint = T.load(dis_model_path)
	discriminator.load_state_dict(checkpoint['model_state_dict'])
	print("Discriminator Load Successful\n")

except:
	print("Discriminator Failed Loading\n")

