# 🏠 Real Estate Chatbot with PyTorch

A production-ready chatbot for real estate inquiries, featuring contextual understanding and easy customization through JSON configuration.

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?logo=python&logoColor=ffdd54)
![NLTK](https://img.shields.io/badge/NLTK-%2300A67E.svg?logo=nltk&logoColor=white)

## 🌟 Key Features
- **Property Specialist**: Pre-trained for listings, pricing, and location queries
- **Smart Dialogue**: 2-layer neural network maintains conversation context
- **Zero-Code Customization**: Modify behavior via simple JSON edits
- **Training Dashboard**: Real-time accuracy metrics during model training

## 🚀 Quick Start
```bash
# Clone and setup
git clone https://github.com/yourusername/real-estate-chatbot.git
cd real-estate-chatbot
python -m venv venv && source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"

# Train and run
python train.py && python chat.py

## 🛠️ Configuration
intents.json Structure:
json
{
  "intents": [
    {
      "tag": "availability",
      "patterns": ["Any units available?", "What's in stock?"],
      "responses": ["We have 5 properties currently available.", "Inventory updates daily."],
      "context": ["property-inquiry"]
    }
  ]
}
## 📦 Project Structure
real-estate-chatbot/
├── train.py         # Model training script
├── chat.py          # Interactive chat interface
├── model.py         # Neural network definition
├── nlp_utils.py     # Text processing functions
├── intents.json     # Conversation patterns
└── data.pth         # Trained model weights
## 📋 Requirements
Python 3.8+

PyTorch 1.12+

NLTK 3.7+
