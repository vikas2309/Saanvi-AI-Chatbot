# AI-Chatbot

![Designer](https://github.com/vikas2309/Saanvi-AI-Chatbot/assets/87652402/3933358e-6dbe-4181-acd7-5d25c947db6e)


## Chat Bot

This  implements a chatbot capable of interacting with users based on predefined intents and responses. The chatbot utilizes NLP techniques and a machine learning model to understand user input and generate appropriate responses.

### Approach

* Preprocessing Data: The script preprocesses intent data from a JSON file, tokenizing patterns, and lemmatizing words to prepare them for training the model.

* Training Data Creation: It creates training data by converting tokenized words into numerical vectors and associating them with corresponding intents. This step prepares the data for training the machine learning model.

* Model Building: The script builds a neural network model using TensorFlow's Keras API. The model architecture consists of densely connected layers with dropout regularization to prevent overfitting.

* Model Training: The model is trained on the prepared training data, optimizing for categorical cross-entropy loss and using the Adam optimizer. The training process iterates over multiple epochs to improve performance.

* Intent Prediction: After training, the model can predict the intent of user input based on the trained data. It calculates the probability distribution over all intents and selects the most likely intent as the predicted response.

* Response Generation: Once the intent is predicted, the script retrieves a response from the predefined set of responses associated with that intent. This response is then returned to the user as the chatbot's reply.


## OS Bot

The chatbot uses NLTK for natural language processing, including tokenization, stemming, and lemmatization, to preprocess user input and text data. It employs TF-IDF vectorization to find and respond with the most contextually relevant sentences from a provided document (document.txt, which contains data on the concept of operating systems). The chatbot can handle basic greetings, expressions of gratitude, and farewells, responding appropriately based on predefined patterns and similarities. The chatbot's responses are returned as JSON objects for frontend display.

### Approach

* Text Preprocessing: The raw document text and user inputs are preprocessed using NLTK. This includes tokenization (splitting text into sentences and words), removing stopwords, and applying stemming and lemmatization to reduce words to their base forms.

* TF-IDF Vectorization: The preprocessed text is transformed into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This helps in quantifying the importance of words in the document relative to the corpus.

* Cosine Similarity: The similarity between the user input and the document sentences is calculated using cosine similarity. This measures the cosine of the angle between two non-zero vectors, providing a metric to find the most similar sentence in the document to the user input.

* Response Generation: Based on the cosine similarity scores, the chatbot identifies the most relevant sentence from the document and uses it as the response. If no relevant sentence is found, a fallback message is provided.

* Handling Greetings and Farewells: The chatbot includes predefined patterns to handle basic greetings and farewells, ensuring a more natural interaction with users.

## Backend
It uses Flask web application to serve as the backend for a chatbot interface. It loads a pre-trained TensorFlow model for natural language processing and handles user requests to generate responses through the chatbot. The routes `/getstart` and `/os/getosdoc` are configured to process user messages and return appropriate responses. Additionally, the application renders HTML templates for the homepage and an OS-specific page.

This approach ensures that the chatbot provides contextually relevant responses while maintaining the ability to handle common conversational elements.

