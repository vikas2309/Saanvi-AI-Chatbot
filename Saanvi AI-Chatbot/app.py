from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import contextlib # For redirecting standard output.
import os
import chatbot # Import the custom chatbot module
import osbot # Import the custom osbot module

app = Flask(__name__, static_folder='static', template_folder='templates')

# Suppress TensorFlow logging output while loading the model.
with contextlib.redirect_stdout(open(os.devnull, 'w')):
    model = tf.keras.models.load_model('chatbot_model.keras')

# Render the homepage template.
@app.route("/")
def home():
    return render_template("index_start.html")

# Handle the chatbot response for the general page.
@app.route("/getstart")
def get_bot_response():
    # Get the user message from the request.
    userText = request.args.get('msg')  
    # Generate a response using the chatbot.
    response = chatbot.chatbot_response(userText) 
    # Return the response as JSON. 
    return jsonify({'response': response})  

# Render the OS-specific page template.
@app.route("/os")
def os():
    return render_template("index_os.html")

# Handle the chatbot response for the OS-specific page.
@app.route("/os/getosdoc", methods=["GET"])
def get_osbot_response():
    # Get the user message from the request.
    userText = request.args.get('msg')  
    # Generate a response using the chatbot.
    response = osbot.chatbot_response(userText)
    # Return the response as JSON. 
    return jsonify({'response': response})  

# Run the Flask app in debug mode.
if __name__ == "__main__":
    app.run(debug=True)  