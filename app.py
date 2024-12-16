from flask import Flask, request
from flask_cors import CORS
print('Initializing inference script...')
from inference import SentimentClassifier, Predictor
import torch

from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
from transformers import pipeline
import random

from PreProcessing import load_json_file, create_df, extract_json_info, input_model

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# loading the predictor
print('Loading Model...')
model = SentimentClassifier()
model.load_state_dict(torch.load('inference.pth'))

print('Initializing Predictor...')
predictor = Predictor(model)

# Chat bot API
model_path = "/home/tangsan/NLP/KLTNWEB/Chatbot/chatbot"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer= BertTokenizerFast.from_pretrained(model_path)
chatbot= pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

print(chatbot("Hello"))


@app.route('/pred', methods=['POST'])
def prediction():
    path = "dataset/intents.json"

    intents = load_json_file(path)
    df = create_df()
    df = extract_json_info(intents, df)
    num_labels, id2label, label2id = input_model(df)

    # POST request
    if request.method == 'POST':
        print('Incoming..')
        payload = request.get_json()
        print("Text: ",payload['body'])

        score = chatbot(payload['body'])[0]['score']
        print("Score: ",score)
        if score < 0.7:
            return "Sorry I can't answer that", 200

        label = label2id[chatbot(payload['body'])[0]['label']]
        response = random.choice(intents['intents'][label]['responses'])

        return response, 200


if __name__ == '__main__':
   app.run(port=5000)