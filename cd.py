# Perceptron
import numpy as np

def perceptron(x1, x2, w1, w2, b):
    net = w1 * x1 + w2 * x2 + b
    return 1 if net > 0 else 0

inputs = [(0, 0), (0, 1), (1, 0), (1, 1)]

print("AND Gate (w1=1, w2=1, b=-1.5):")
for x1, x2 in inputs: print(f"{x1} AND {x2} = {perceptron(x1, x2, 1, 1, -1.5)}")

print("\nOR Gate (w1=1, w2=1, b=-0.5):")
for x1, x2 in inputs: print(f"{x1} OR {x2} = {perceptron(x1, x2, 1, 1, -0.5)}")

print("\nNAND Gate (w1=-1, w2=-1, b=1.5):")
for x1, x2 in inputs: print(f"{x1} NAND {x2} = {perceptron(x1, x2, -1, -1, 1.5)}")

print("\nNOR Gate (w1=-1, w2=-1, b=0.5):")
for x1, x2 in inputs: print(f"{x1} NOR {x2} = {perceptron(x1, x2, -1, -1, 0.5)}")

def static_sum(x1, x2):
    or_out = perceptron(x1, x2, 1, 1, -0.5)
    nand_out = perceptron(x1, x2, -1, -1, 1.5)
    return perceptron(or_out, nand_out, 1, 1, -1.5)

print("\nStatic Sum (Sum bit = XOR via OR, NAND, AND combination):")
for x1, x2 in inputs: print(f"{x1} + {x2} (Sum) = {static_sum(x1, x2)}")

# Backpropogation

import numpy as np
from sklearn.neural_network import MLPClassifier

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

def sigmoid(x): return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x): return x * (1 - x)

def backprop_custom(init_type="random"):
    np.random.seed(42)
    input_size, hidden_size, output_size = 2, 2, 1
    
    if init_type == "random":
        W1 = np.random.randn(input_size, hidden_size)
        W2 = np.random.randn(hidden_size, output_size)
    elif init_type == "xavier":
        W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1 / input_size)
        W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1 / hidden_size)
        
    b1 = np.zeros((1, hidden_size))
    b2 = np.zeros((1, output_size))
    
    lr = 0.1
    for epoch in range(10000):
        # Forward pass
        hidden_in = np.dot(X, W1) + b1
        hidden_out = sigmoid(hidden_in)
        out_in = np.dot(hidden_out, W2) + b2
        predicted_out = sigmoid(out_in)
        
        # Backpropagation
        error = y.reshape(-1, 1) - predicted_out
        d_out = error * sigmoid_derivative(predicted_out)
        
        error_hidden = d_out.dot(W2.T)
        d_hidden = error_hidden * sigmoid_derivative(hidden_out)
        
        W2 += hidden_out.T.dot(d_out) * lr
        b2 += np.sum(d_out, axis=0, keepdims=True) * lr
        W1 += X.T.dot(d_hidden) * lr
        b1 += np.sum(d_hidden, axis=0, keepdims=True) * lr
        
    print(f"[{init_type.capitalize()} Init] Predictions:\n{predicted_out}")

print("--- Custom Backprop (Random Init) ---")
backprop_custom("random")

print("\n--- Custom Backprop (Xavier Init) ---")
backprop_custom("xavier")

print("\n--- MLPClassifier (sklearn) ---")
mlp = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', max_iter=10000, random_state=42)
mlp.fit(X, y)
print(f"MLPClassifier Predictions: {mlp.predict(X)}")

# CNN

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

print("--- Variation 1 & 2: Convolution on Given Matrix / Image ---")
matrix = np.array([
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0],
    [10, 10, 10, 0, 0, 0]
], dtype=np.float32)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

output_dim = matrix.shape[0] - kernel.shape[0] + 1
output_matrix = np.zeros((output_dim, output_dim))

for i in range(output_dim):
    for j in range(output_dim):
        output_matrix[i, j] = np.sum(matrix[i:i+3, j:j+3] * kernel)

print("Original Matrix:\n", matrix)
print("Convolution Result:\n", output_matrix)

print("\n--- Variation 3: CNN on MNIST Dataset ---")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train[:1000], y_train[:1000], epochs=2, batch_size=32, validation_data=(x_test[:200], y_test[:200]))

# RNN

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_features = 10000
maxlen = 50

print("Loading IMDB dataset...")
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words=max_features)
input_train = pad_sequences(input_train, maxlen=maxlen)
input_test = pad_sequences(input_test, maxlen=maxlen)

model = Sequential([
    Embedding(max_features, 32),
    SimpleRNN(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train[:2000], y_train[:2000], epochs=3, batch_size=128, validation_split=0.2)

# SOM

import numpy as np

# A simple Self-Organizing Map (SOFM) for clustering RGB colors
data = np.array([
    [1.0, 0.0, 0.0], # Red
    [0.0, 1.0, 0.0], # Green
    [0.0, 0.0, 1.0], # Blue
    [1.0, 1.0, 0.0]  # Yellow
])

grid_size = (2, 2)
weights = np.random.rand(grid_size[0], grid_size[1], 3)
epochs = 100
lr = 0.1

for epoch in range(epochs):
    for x in data:
        # Find Best Matching Unit (BMU)
        distances = np.linalg.norm(weights - x, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Update weights (only BMU updated for simplicity)
        weights[bmu_idx] += lr * (x - weights[bmu_idx])

print("Trained SOM Weights (Grid representing colors):")
print(weights)

# PCA Autoencoder

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# PCA on Iris Dataset
print("--- PCA on Iris Dataset ---")
iris = load_iris()
X = iris.data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Original shape: {X.shape}, PCA shape: {X_pca.shape}")

# Autoencoder on given dataset (using Iris for simplicity to match PCA)
print("\n--- Autoencoder on Iris Dataset ---")
X_scaled = X / np.max(X, axis=0)

input_layer = Input(shape=(4,))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(4, activation='sigmoid')(encoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=8, verbose=0)
reconstructed = autoencoder.predict(X_scaled)
print(f"Reconstruction MSE: {np.mean(np.square(X_scaled - reconstructed)):.4f}")

# VAE
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # mu
        self.fc22 = nn.Linear(400, 20) # logvar
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

model.train()
for batch_idx, (data, _) in enumerate(train_loader):
    data = data.to(device)
    optimizer.zero_grad()
    recon_batch, mu, logvar = model(data)
    loss = loss_function(recon_batch, data, mu, logvar)
    loss.backward()
    optimizer.step()
    if batch_idx == 100: 
        print(f"Batch {batch_idx}, Loss: {loss.item() / len(data):.4f}")
        break

# GAN

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.ToTensor()
train_loader = DataLoader(datasets.MNIST(root='./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)

print("--- Variation 1: GAN on MNIST ---")
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 784), nn.Tanh())
    def forward(self, x): return self.main(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(nn.Linear(784, 256), nn.LeakyReLU(0.2), nn.Linear(256, 1), nn.Sigmoid())
    def forward(self, x): return self.main(x)

netG, netD = Generator().to(device), Discriminator().to(device)
criterion = nn.BCELoss()
optD, optG = optim.Adam(netD.parameters(), lr=0.0002), optim.Adam(netG.parameters(), lr=0.0002)

for i, (data, _) in enumerate(train_loader):
    real_data = data.view(-1, 784).to(device)
    batch_size = real_data.size(0)
    
    # Train D
    netD.zero_grad()
    label_real = torch.ones(batch_size, 1).to(device)
    output_real = netD(real_data)
    errD_real = criterion(output_real, label_real)
    
    noise = torch.randn(batch_size, 100).to(device)
    fake_data = netG(noise)
    label_fake = torch.zeros(batch_size, 1).to(device)
    output_fake = netD(fake_data.detach())
    errD_fake = criterion(output_fake, label_fake)
    
    errD = errD_real + errD_fake
    errD.backward()
    optD.step()
    
    # Train G
    netG.zero_grad()
    label_real_g = torch.ones(batch_size, 1).to(device)
    output_g = netD(fake_data)
    errG = criterion(output_g, label_real_g)
    errG.backward()
    optG.step()
    
    if i == 50:
        print(f"GAN Step {i} | Loss_D: {errD.item():.4f} | Loss_G: {errG.item():.4f}")
        break

print("\n--- Variation 2: Diffusion Model (Adding Noise) ---")
# Simple forward diffusion demonstration
def forward_diffusion(x0, t, beta):
    noise = torch.randn_like(x0)
    alpha = 1 - beta
    alpha_cumprod = torch.cumprod(alpha, dim=0)
    mean = torch.sqrt(alpha_cumprod[t]) * x0
    variance = torch.sqrt(1 - alpha_cumprod[t]) * noise
    return mean + variance

# Demo adding noise
sample_image = next(iter(train_loader))[0][0].view(-1, 784)
betas = torch.linspace(0.0001, 0.02, 100) # 100 timesteps
noisy_image = forward_diffusion(sample_image, t=50, beta=betas)
print(f"Original image mean: {sample_image.mean().item():.4f}, std: {sample_image.std().item():.4f}")
print(f"Noisy image (t=50) mean: {noisy_image.mean().item():.4f}, std: {noisy_image.std().item():.4f}")
     
