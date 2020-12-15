import torch as T 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os

import midi_processing as mpr

class Generator(nn.Module):
	def __init__(self, dict_size, emb_size, hidden_size, n_layers, dense_size, dropout, device):
		super(Generator, self).__init__()

		self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
		self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, \
			batch_first=True, bidirectional=True, dropout=dropout)
		self.gru = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=n_layers, \
			batch_first=True, dropout=dropout)
		self.dense_1 = nn.Linear(hidden_size, dense_size)
		self.dense_2 = nn.Linear(dense_size, dict_size)
		self.relu = nn.LeakyReLU()

		self.dict_size = dict_size
		self.emb_size = emb_size
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.dense_size = dense_size
		self.dropout = dropout
		self.device = device
		
		self.to(device)

	def forward(self, x):
		out = self.embedding(x)
		out, hidden = self.lstm(out)
		out = self.relu(out)
		out, hidden = self.gru(out)
		out = self.relu(out)
		out = self.dense_1(out[:,-1])
		out = self.relu(out)
		out = nn.Dropout(self.dropout)(out)
		out = self.dense_2(out)

		return out

	@staticmethod
	def choose_max(arr):
		sort = arr.argsort()[-5:]
		index = np.random.choice(5, 1, p=[0.025/3, 0.025/3, 0.025/3, 0.075, 0.9])[0]
		return sort[index]


class Data(Dataset):

	def __init__(self, path, seq_ln, max_len, device):

		for _, _, files in os.walk(path):
			num_samples = 0

			for f in files:
				n = np.load(os.path.join(path, f))
				if len(n) > max_len:
					n = n[:max_len]
				if len(n) > seq_ln:
					num_samples += len(n)-seq_ln

			inp_seq = np.zeros((num_samples, seq_ln))
			target_seq = np.zeros((num_samples, 1))

			count = 0

			for i, f in enumerate(files):
				n = np.load(os.path.join(path, f))
				if len(n) > max_len:
					n = n[:max_len]
				if len(n) > seq_ln:
					for j in range(len(n)-seq_ln):
						inp_seq[count] = n[j:j+seq_ln]
						target_seq[count] = n[j+seq_ln]
						count += 1
				
		self.len = num_samples
		self.inp_seq = T.from_numpy(inp_seq).to(device, dtype=T.long)
		self.target_seq = T.from_numpy(target_seq).to(device, dtype=T.long)
		self.dtype = device

		print(self.len)

	def __getitem__(self, index):
		return (self.inp_seq[index], self.target_seq[index])

	def __len__(self):
		return self.len


def training(model, optimizer, dataset, epochs=100, batch_size=8):

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	loss = nn.CrossEntropyLoss()

	print("Starting Training")

	for i in range(epochs):

		print(f"Epoch: {i+1}")

		dataiter = iter(dataloader)

		for j, (data, true) in enumerate(dataiter):
			optimizer.zero_grad()
			output = model(data)
			batch_loss = loss(output, true.view(-1))
			batch_loss.backward()
			optimizer.step()

		print("Loss: {:.5f}".format(batch_loss.item()))
		print("-------------")

		checkpoint_gen = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict()
		}
		T.save(checkpoint_gen, f".\\models\\genV1\\lstm_epoch_{i}.pth")

		generate(model, "D:\\Projects\\MusicGAN\\datasets\\Lakh\\clean_midi.tar\\clean_midi\\_Weird Al_ Yankovic\\Amish Paradise.mid", f"epoch_{i}", 400)


def generate(model, path, f_name, max_len=1000):
	len_input = 60

	mid = mpr.process_midi(path)

	text = mpr.notes_to_text(mid)
	inp = T.from_numpy(mpr.tokenize(text[:len_input])).type(T.cuda.LongTensor)

	final = inp.cpu().clone().numpy().astype(np.int)

	model.eval()

	for i in range(max_len):

		out = model(inp[None,:])
		
		maxed_out = np.zeros(out.shape[0])

		for j in range(len(maxed_out)):
			maxed_out[j] = model.choose_max(out[j])

		prev = inp.detach().cpu().clone().numpy().astype(np.int)

		inp = np.append(prev[1:], maxed_out[-1])

		final = np.append(final, maxed_out[-1])

		inp = T.from_numpy(inp).type(T.cuda.LongTensor)

	model.train()

	print(mpr.untokenize(final).tolist())
	converted = mpr.text_to_notes(mpr.untokenize(final).tolist())
	file = mpr.write_midi(converted, 0.5)
	file.write(f"generated\\{f_name}.mid")


if __name__ == "__main__":
	dict_size = 113
	emb_size = 4
	hidden_size = 128
	n_layers = 2
	dense_size = 64
	dropout = 0.3
	lr = 0.0001
	device = T.device("cuda:0")

	PATH = r".\datasets\videogameNPY"
	MODEL_PATH = ".\\models\\genV1\\lstm_epoch_.pth"
	FILE_LEN = 400
	SEQ_LEN = 60

	data = Data(PATH, SEQ_LEN, FILE_LEN, device)

	model = Generator(dict_size=dict_size, emb_size=emb_size, hidden_size=hidden_size, n_layers=n_layers, \
		dense_size=dense_size, dropout=dropout, device=device)

	optimizer = T.optim.Adam(model.parameters(), lr=lr, betas=(0, 0.99))

	try:
		checkpoint = T.load(MODEL_PATH)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		print("Load Successful\n")

	except:
		print("Failed Loading\n")

	training(model, optimizer, data, epochs=500, batch_size=64)