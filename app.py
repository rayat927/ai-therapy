# import tensorflow
import pickle
from urllib import response
import pandas as pd
from pprint import pprint
from sklearn.utils import resample
from tensorflow.keras.models import load_model
import numpy as np
import json
import random
from flask import Flask, request, Response, abort, send_file, jsonify, url_for
from flask_cors import CORS

dataset = pd.read_csv('tweet_emotions.csv')

dataset.sentiment.value_counts()
target_class = 9

# classes_ids = {name:ids for name, ids in zip(set(dataset.sentiment.to_list()),range(len(set(dataset.sentiment.to_list()))))}
classes_ids = {name: idx for idx, name  in enumerate(dataset.sentiment.unique())}
inv_classes_ids = {value:key for key, value in zip(list(classes_ids.keys()), list(classes_ids.values()))}

pprint(classes_ids)

target_majority = dataset[dataset.sentiment==inv_classes_ids[target_class]]

for cl in range(len(classes_ids)):
    train_minority = dataset[dataset.sentiment==inv_classes_ids[cl]]
    train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(target_majority), random_state=123)
    if cl == 0:
        dataset_upsampled = pd.concat([train_minority_upsampled, target_majority])
        #train_upsampled = pd.concat([train_upsampled, ])
    if cl>0 and cl!=target_class:
        dataset_upsampled = pd.concat([train_minority_upsampled, dataset_upsampled])
dataset_upsampled = dataset_upsampled.sample(frac=1).reset_index(drop=True)


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('chatbot6.h5')

test_sentence = ['i am anger']

sequence = tokenizer.texts_to_sequences(test_sentence)
# {'anger': 12,
#  'boredom': 10,
#  'empty': 0,
#  'enthusiasm': 2,
#  'fun': 7,
#  'happiness': 9,
#  'hate': 8,
#  'love': 6,
#  'neutral': 3,
#  'relief': 11,
#  'sadness': 1,
#  'surprise': 5,
#  'worry': 4}

app = Flask(__name__)
CORS(app)

@app.route('/text', methods=['POST'])
def get_text():
    text = request.json['text']
    sentence = [text]
    sequence = tokenizer.texts_to_sequences(sentence)
    predictions = model.predict(sequence)
    print(predictions)
    print(np.argmax(predictions))
    s = np.argmax(predictions)
    p = inv_classes_ids.get(s)
    print(p)

    if (p == 'hate' or p == 'anger'):
        data = json.load(open('anger.json'))
        r =random.randint(0, len(data['texts']) - 1)
        print(data['texts'][r]['text'])
        return jsonify({"msg":data['texts'][r]['text'], "emotion": p})
    
    elif(p == 'happiness' or p=='fun' or p == 'enthusiasm' or p == 'love'):
        data = json.load(open('happy.json'))
        r =random.randint(0, len(data['texts']) - 1)
        print(data['texts'][r]['text'])
        return jsonify({"msg":data['texts'][r]['text'], "emotion": p})
    elif(p == 'boredom'):
        data = json.load(open('boredom.json'))
        r =random.randint(0, len(data['texts']) - 1)
        print(data['texts'][r]['text'])
        return jsonify({"msg":data['texts'][r]['text'], "emotion": p})
    elif(p == 'worry'):
        data = json.load(open('worry.json'))
        r =random.randint(0, len(data['texts']) - 1)
        print(data['texts'][r]['text'])
        return jsonify({"msg":data['texts'][r]['text'], "emotion": p})
    else:
        data = json.load(open('notUnderstood.json'))
        r =random.randint(0, len(data['texts']) - 1)
        print(data['texts'][r]['text'])
        return jsonify({"msg":data['texts'][r]['text'], "emotion": p})
    
 
    

    

if __name__ == "__main__":
    app.run(debug=True)