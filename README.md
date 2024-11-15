Contextual Chatbot using PyTorch
This is a simple Contextual Chatbot implementation using PyTorch. The bot is designed to help users by providing responses based on the input, with the main focus on real estate queries. The bot can be customized with additional intents, patterns, and responses to meet your specific use case.

The architecture follows a Feed-Forward Neural Network with two hidden layers, making it beginner-friendly while offering an efficient solution for contextual chatbots.

Demo
You can watch the tutorial on how this chatbot works:
Watch the Tutorial

Features
Customizable: Modify the intents.json file to update the chatbot’s behavior.
Contextual: The bot can understand different intents and respond based on user queries.
Simple Architecture: Based on a feed-forward neural network with 2 hidden layers.
Real Estate Use Case: The bot can handle questions related to properties, real estate listings, pricing, and more.
Requirements
Python 3.x
PyTorch
NLTK
Other dependencies listed below
Installation
Follow these steps to set up the chatbot on your machine.

1. Create a Virtual Environment
Create a virtual environment for your project to manage dependencies.

bash
Copy code
mkdir myproject
cd myproject
python3 -m venv venv
2. Activate the Environment
Mac/Linux:

bash
Copy code
source venv/bin/activate
Windows:

bash
Copy code
venv\Scripts\activate
3. Install PyTorch and Dependencies
You need to install PyTorch and NLTK.

bash
Copy code
pip install torch
pip install nltk
For PyTorch installation, refer to the official website for the specific version compatible with your setup.

4. Download NLTK Data
Run this once in Python to download the necessary data for tokenization:

python
Copy code
import nltk
nltk.download('punkt')
Usage
1. Train the Model
Run the train.py script to start training the chatbot. It will generate the data.pth file used by the chatbot during the interaction phase.

bash
Copy code
python train.py
2. Run the Chatbot
Once the model is trained, run the chat.py script to start chatting with the bot.

bash
Copy code
python chat.py
Customization
Modify intents.json