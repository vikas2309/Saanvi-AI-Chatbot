import numpy as np
import random
import json
import nltk
import torch
import torch.nn as nn
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request, jsonify
stemmer = PorterStemmer()

app = Flask(__name__, static_folder='static', template_folder='templates')

bot_name = "Saanvi"

bot_output = "Hi I am Saanvi built by Vikas Reddy Ginuga. I am here to help you with queries related to KMIT College.! (type 'quit' to exit)"


# Neural Network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # No activation and no softmax at the end
        return out

# Tokenization function
def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word, punctuation character, or number.
    """
    return nltk.word_tokenize(sentence)

# Stemming function
def stem(word):
    """
    Stemming = finding the root form of the word.
    """
    return stemmer.stem(word.lower())

# Bag of words function
def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise.
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag

def chatbot_response(bot_input):
    """
    Generate a response from the chatbot based on the input sentence.

    Args:
    bot_input (str): The input sentence from the user.

    Returns:
    str: The chatbot's response.
    """
    
    if bot_input is None:
        return bot_output
    
    sentence = bot_input.lower()
    if sentence in ["quit", "bye", "thanks"]:
        return "Goodbye!"

    # Tokenize and process the input sentence
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Get the model's output
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate the probability of the predicted class
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    # Generate response based on the highest probability
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

# Vectorizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('C:/Users/vikas/kmitbot/intents.json', 'r') as json_data:
    intents = json.load(json_data)
    

FILE = "C:/Users/vikas/kmitbot/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


#Routes
@app.get('/')
async def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET"])
async def predict():
    # Get the user message from the request.
    userText = request.args.get('msg')  
    # Generate a response using the chatbot.
    response = chatbot_response(userText)
    # Return the response as JSON. 
    return jsonify({'response': response})  
    

# Run the Flask app in debug mode.
if __name__ == "__main__":
    app.run(debug=True) 