import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re

# Load the document
try:
    with open('C:/Users/vikas/Saanvi AI-Chatbot/Document.txt', 'r', errors='ignore') as file:
        raw_data = file.read()
except Exception as e:
    print(f"Error loading document: {e}")
    raw_data = ""

# Preprocess the text
pattern = r'[\n]'
raw_data = re.sub(pattern, ' ', raw_data)
raw_data = raw_data.lower()

# Split the text into a list of sentences.
sentences = sent_tokenize(raw_data) 
# Split the text into a list of words and punctuation. 
words = word_tokenize(raw_data)      

# Create a set of English stopwords.
stop_words = set(stopwords.words('english'))
# Filter out stopwords from the tokenized words.
filtered_words = [word for word in words if word not in stop_words]

# Initialize the lemmatizer and stemmer.
lemmer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Method to lemmatize each token in the list.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

# Method to stem each token in the list.
def StemTokens(tokens):
    return [stemmer.stem(token) for token in tokens]


# Create a dictionary to remove punctuation.
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


# Method to normalize text by lowercasing, removing punctuation, tokenizing, lemmatizing, and stemming.
def LemStemNormalize(text):
    tokens = nltk.word_tokenize(text.lower().translate(remove_punct_dict))
    lemmed_tokens = LemTokens(tokens)
    stemmed_tokens = StemTokens(lemmed_tokens)
    return stemmed_tokens


# Define possible greeting inputs.
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
# Define possible greeting responses.
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# Method to return a random greeting response if the input contains a greeting.
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Method to generate a response based on the similarity of the input with the document sentences.
def response(user_response):
    saanvi = ''
    # Add user input to the list of sentences.
    sentences.append(user_response) 
    # Initialize TF-IDF vectorizer with custom tokenizer.
    TfidfVec = TfidfVectorizer(tokenizer=LemStemNormalize, stop_words='english')
    # Fit and transform the sentences to TF-IDF matrix.
    tfidf = TfidfVec.fit_transform(sentences)
    # Compute cosine similarity between user input and all sentences.
    vals = cosine_similarity(tfidf[-1], tfidf)
    # Find the index of the most similar sentence.
    idx = vals.argsort()[0][-2]
    # Flatten the cosine similarity array.
    flat = vals.flatten()
    # Sort the flattened array.
    flat.sort() 
    # Get the second highest similarity score.
    req_tfidf = flat[-2] 
    if req_tfidf == 0:
        # Response if no similarity is found.
        saanvi = saanvi + "I am sorry! I don't understand you please give me sometime to understand you!!!" 
    else:
        # Response with the most similar sentence.
        saanvi = saanvi + sentences[idx] 
    # Remove user input from the list of sentences.
    sentences.remove(user_response) 
    return saanvi


# Method to generate a chatbot response for the given text.
def chatbot_response(user_text):
    try:
        # Convert the user message to lowercase.
        user_text = user_text.lower()  
        if user_text != 'bye':
            if user_text == 'thanks' or user_text == 'thank you':
                # Respond to thanks.
                response_text = "You are welcome.."  
            else:
                if greeting(user_text) is not None:
                    # Respond to greetings.
                    response_text = greeting(user_text)  
                else:
                    # Generate a response based on the document.
                    response_text = response(user_text)  
        else:
            # Respond to goodbye.
            response_text = "Bye! take care.."  
        # Return the response as JSON.
        return response_text  
    except Exception as e:
        # Print error to console.
        print(f"Error : {e}")  
        