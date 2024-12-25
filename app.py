from flask import Flask, render_template, request, jsonify, session
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from better_profanity import profanity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

profanity.load_censor_words()

def ask_question(context, question):
    input_text = f"{context} question: {question} answer:"
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    filtered_response = profanity.censor(response)
    return filtered_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    question = data.get('question')
    context = session.get('context', '')
    answer = ask_question(context, question)
    context += f" question: {question} answer: {answer}"
    session['context'] = context
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)

