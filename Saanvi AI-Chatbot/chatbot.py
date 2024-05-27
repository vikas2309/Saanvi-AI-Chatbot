import numpy as np  # For numerical operations.
import random  # For generating random choices.
import json  # For working with JSON data.
import nltk  # For natural language processing.
from nltk.stem import WordNetLemmatizer  # For text lemmatization.
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore

# Download the Punkt tokenizer.
nltk.download('punkt') 
# Download the WordNet corpus. 
nltk.download('wordnet')  


# Preprocess data
def preprocess_data(intents):
    # Extract words, classes, and documents from the intents.
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            # Tokenize each pattern.
            word_list = nltk.word_tokenize(pattern)  
            # Add tokens to the words list.
            words.extend(word_list)  
            # Append tokens and tag to documents.
            documents.append((word_list, intent['tag']))  
            if intent['tag'] not in classes:
                # Add tag to classes if not already present.
                classes.append(intent['tag'])  

    # Lemmatize and lower the words.
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    # Remove duplicates and sort the words.
    words = sorted(list(set(words)))  
    # Remove duplicates and sort the classes.
    classes = sorted(list(set(classes)))  
    
    return words, classes, documents


# Create training data
def create_training_data(words, classes, documents):
    # Create training data from words, classes, and documents.
    training = []
    # Create an empty output array for classes.
    output_empty = [0] * len(classes)  

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        # Lemmatize and lower the words.
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            # Create a bag of words.
            bag.append(1) if word in word_patterns else bag.append(0)  

        output_row = list(output_empty)
        # Set the corresponding class index to 1.
        output_row[classes.index(doc[1])] = 1
        # Append bag of words and output row to training data.
        training.append([bag, output_row])  

    # Ensure all rows have the same length
    for i in range(len(training)):
        if len(training[i][0]) != len(words) or len(training[i][1]) != len(classes):
            # Print inconsistent training data.
            print(f"Inconsistent training data at index {i}: {training[i]}")  

    # Shuffle the training data.
    random.shuffle(training)  
    # Convert training data to a NumPy array.
    training = np.array(training, dtype=object)  
    # Extract features.
    train_x = np.array(list(training[:, 0]), dtype=np.float32)  
    # Extract labels.
    train_y = np.array(list(training[:, 1]), dtype=np.float32)  

    return train_x, train_y


# Method to build chatbot responses
def clean_up_sentence(sentence):
    # Tokenize and lemmatize a sentence.
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


# Method to convert a sentence into a bag-of-words representation.
def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    # Print the found word.
                    print(f"found in bag: {w}")  
    return np.array(bag)


# Method to predict the class of a given sentence.
def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    # Set the error threshold.
    ERROR_THRESHOLD = 0.25  
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort results by probability.
    results.sort(key=lambda x: x[1], reverse=True)  
    return_list = []
    for r in results:
        # Append intent and probability to the return list.
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})  
    return return_list


# Method to get a response based on the predicted intent.
def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            # Choose a random response for the intent.
            result = random.choice(i['responses'])
            break
    return result


# Method to generate a chatbot response for the given text.
def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res


# Initialize the WordNet lemmatizer.
lemmatizer = WordNetLemmatizer()  

# Load the intents from a JSON file.
with open('intents.json', 'r') as file:
    intents = json.load(file)  

# Preprocess the intents data.
words, classes, documents = preprocess_data(intents)  

# Create training data.
train_x, train_y = create_training_data(words, classes, documents)  

# Build and train the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(64, activation='relu'))  
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  

# Train the model.
model.fit(train_x, train_y, epochs=80, batch_size=5, verbose=1)  

# Save the model in the recommended Keras format
model.save('chatbot_model.keras')
