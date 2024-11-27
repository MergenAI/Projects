import pandas as pd
import torch as t
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn


class WeatherRegression(nn.Module):
    def __init__(self, input_size, latent_size, output_size=1):
        super(WeatherRegression,self).__init__()
        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 64)

        self.mu = nn.Linear(64, latent_size)
        self.logvar = nn.Linear(64, latent_size)

        self.l4 = nn.Linear(20, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)

    def encoder(self, x):
        x = t.relu(self.l1(x))
        x = t.relu(self.l2(x))
        x = self.l3(x)

        mu = self.mu(x)
        logvar = self.logvar(x)

        return mu, logvar

    def decoder(self, z):
        h1 = t.relu(self.l4(z))
        h1 = t.relu(self.l5(h1))
        h1 = (self.l6(h1))
        return h1

    def reparametrization(self, mu, logvar):
        std = t.exp(.5 * logvar)
        eps = t.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparametrization(mu, logvar)
        return self.decoder(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    reconstruction_loss = nn.functional.mse_loss(recon_x, x, reduction="sum")
    kl_loss = t.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (-.5)
    return reconstruction_loss + kl_loss


df = pd.read_csv(
    "C:\Program Files (x86)\Araştırma\Araştırma\Dersler\\24SS\SEP\ifn-el-0\Code\DataAnalysis\Datasets\\daily_dataframe_ndr_with_one_shift_dropna_week.csv")

print(df.shape)
y = df.loc[:, "tavg"]
x = df.drop(["tavg", "Unnamed: 0", "time"], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_tensor = t.Tensor(x_train)
x_test_tensor = t.Tensor(x_test)
y_train_tensor = t.Tensor(y_train.to_numpy()).view(-1, 1)
y_test_tensor = t.Tensor(y_test.to_numpy()).view(-1, 1)

train_dataset = t.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
train_loader = t.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

input_size = len(pd.DataFrame(x_train).columns)

# Parameters
input_dim = input_size
latent_dim = 20  # Size of latent space
learning_rate = 1e-3
batch_size = 64
epochs = 10

device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = WeatherRegression(input_size=input_dim, latent_size=latent_dim).to(device)
optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    loss = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        recon, mu,logvar = model(x_batch)

        batch_loss = loss_function(recon,y_batch,mu,logvar)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item()
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss / len(train_loader)}')


