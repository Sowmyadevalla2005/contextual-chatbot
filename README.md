# Contextual Chatbot using PyTorch  

A simple chatbot for real estate queries, built with PyTorch and NLTK. Customize responses by editing `intents.json`.  

## Features  
- 🏠 **Real Estate Focus**: Handles property listings, pricing, and FAQs  
- 🧠 **Neural Network**: 2-layer feed-forward model for contextual responses  
- ⚙️ **Easy Customization**: Modify `intents.json` to add new queries  

## Installation  
```bash
git clone https://github.com/yourusername/real-estate-chatbot.git
cd real-estate-chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch nltk
python -c "import nltk; nltk.download('punkt')"

Usage
Train the model:

bash
python train.py  # Generates `data.pth`
Chat with the bot:

bash
python chat.py
Customization
Edit intents.json to add/update responses:

json
{
  "intents": [
    {
      "tag": "pricing",
      "patterns": ["What's the price?", "How much does it cost?"],
      "responses": ["Prices start at $300k."]
    }
  ]
}
Demo
Watch Tutorial

Requirements
Python 3.x

PyTorch

NLTK
