import json
from nltk_utils import tokenize, stem, bag_of_words 
import numpy as np  

import torch  
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

# Load intents file
with open('intents.json','r') as f:
    intents = json.load(f)

# Prepare the training data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    name = intent['name']
    tags.append(name)
    for pattern in intent['trainingPhrases']:  # Fixed to match your input structure
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append([w, name])
        
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))  # Ensure unique words
tags = sorted(set(tags))

# Create training data
x_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # Pass `all_words` to bag_of_words
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)
    
    label = tags.index(tag)  # Fix here - 'name' should be replaced with 'tag'
    y_train.append(label)
x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)  # Make sure this matches the length of the vocabulary
learning_rate = 0.001
num_epochs = 1000


# Dataset class - This should be outside the hyperparameter block
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# Prepare DataLoader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        # Backward and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print epoch and loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
        
# Final loss printout after training is complete
print(f'Final Loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE) 

print(f'training complete. file saved to {FILE}')