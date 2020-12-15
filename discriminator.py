import torch as T 
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import os 

import midi_processing as mpr 

class Discriminator(nn.Module):
	def __init__(self, dict_size, emb_size, hidden_size, n_layers, dense_size, dropout, device):
		super(Discriminator, self).__init__()

		self.embedding = nn.Embedding(num_embeddings=dict_size, embedding_dim=emb_size)
		self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=n_layers, \
			batch_first=True, bidirectional=True, dropout=dropout)
		self.gru = nn.GRU(input_size=hidden_size*2, hidden_size=hidden_size, num_layers=n_layers, \
			batch_first=True, dropout=dropout)
		self.dense_1 = nn.Linear(hidden_size, dense_size)
		self.dense_2 = nn.Linear(dense_size, 1)
		self.relu = nn.LeakyReLU()
		self.sigmoid = nn.Sigmoid()

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
		out = self.sigmoid(out)

		return out


class Data(Dataset):

	def __init__(self, path, device):

		num_real = 0
		num_fake = 0

		for folder_path, _, files in os.walk(path):
			for f in files:
				if folder_path[-4:] == "real":
					num_real += 1
				else:
					num_fake += 1

		print(num_real)
		print(num_fake)

		real = np.zeros((num_real, 120))
		fake = np.zeros((num_fake, 120))

		real_count = 0
		fake_count = 0

		for folder_path, _, files in os.walk(path):
			for f in files:
				if folder_path[-4:] == "real":
					real[real_count] = np.load(os.path.join(folder_path, f))
					real_count += 1
				else:
					fake[fake_count] = np.load(os.path.join(folder_path, f))
					fake_count += 1

		real = real.astype(np.uint8)
		fake = fake.astype(np.uint8)

		data = np.zeros((real_count+fake_count, 120))

		data[:len(real)] = real
		data[len(real):] = fake 

		labels = np.zeros((real_count+fake_count, 1))

		labels[:len(real)] = 1

		self.len = real_count + fake_count
		self.data = T.from_numpy(data).to(device, dtype=T.long)
		self.labels = T.from_numpy(labels).to(device, dtype=T.float)

	def __len__(self):
		return self.len 

	def __getitem__(self, index):
		return self.data[index], self.labels[index]


def training(model, optimizer, dataset, epochs=100, batch_size=8):

	dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
	loss = nn.BCELoss()

	print("Starting Training")

	for i in range(epochs):

		print(f"Epoch: {i+1}")

		dataiter = iter(dataloader)

		for j, (data, true) in enumerate(dataiter):
			optimizer.zero_grad()
			output = model(data)
			batch_loss = loss(output, true)
			batch_loss.backward()
			optimizer.step()

		print("Loss: {:.5f}".format(batch_loss.item()))
		print("-------------")

		checkpoint_gen = {
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict()
		}
		T.save(checkpoint_gen, f".\\models\\disV1\\lstm_epoch_{i}.pth")



if __name__ == "__main__":
	dict_size = 113
	emb_size = 4
	hidden_size = 128
	n_layers = 2
	dense_size = 64
	dropout = 0.1
	lr = 0.0001
	device = T.device("cuda:0")

	PATH = r".\datasets\dis_videogameNPY"
	MODEL_PATH = ".\\models\\disV1\\lstm_epoch_.pth"

	data = Data(PATH, device)

	model = Discriminator(dict_size=dict_size, emb_size=emb_size, hidden_size=hidden_size, n_layers=n_layers, \
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