import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import difflib
import numpy as np
import tensorflow as tf
import json
import pickle
import random
import matplotlib.pyplot as plt

with open("intent.json") as file:
    data = json.load(file)

words = []
labels = []
docs_patt = []
docs_tag = []

#TOKENISATION & STEMMING
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern.lower())
        words.extend(wrds)
        docs_patt.append(wrds)
        docs_tag.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])

stemmer = LancasterStemmer()
words = [stemmer.stem(w.lower()) for w in words if w.isalpha()]
words = sorted(list(set(words)))
labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

# BAG OF WORDS - FEATURE ENGINEERING
for x, doc in enumerate(docs_patt):
    bag = []
    wrds = [stemmer.stem(w.lower()) for w in doc if w.isalpha()]
    for w in words:
        bag.append(1) if w in wrds else bag.append(0)
    output_row = out_empty[:]
    output_row[labels.index(docs_tag[x])] = 1

    training.append(bag)
    output.append(output_row)

training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)


tf.compat.v1.reset_default_graph()
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, input_shape=(len(training[0]),), activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(len(output[0]), activation="softmax")
])
net.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history= net.fit(training, output, epochs=500, batch_size=8, verbose=1)

net.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words if word.isalpha()]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)


def json_to_dictionary(data):
    dictionary = []
    fil_dict = []
    vocabulary = []
    for i in data["intents"]:
        for pattern in i["patterns"]:
            vocabulary.append(pattern.lower())
    for i in vocabulary:
        dictionary.append(words_to_list(i))
    for i in range(len(dictionary)):
        for word in dictionary[i]:
            fil_dict.append(word)
    return list(set(fil_dict))

chatbot_vocabulary = json_to_dictionary(data)

from flask import Flask, Response, request
from flask_ngrok import run_with_ngrok
import pymongo

client=pymongo.MongoClient()
mydb=client['mydb']

app=Flask(__name__)
run_with_ngrok(app)


@app.route('/webhook',methods=['POST'])
def webhook():
    req=request.get_json(silent=True,force=True)
    session=mydb[req['session']]
    user=req['queryResult']['queryText']
    bot=req['queryResult']['fulfillmentText']
    data={
    "User":user,
    "Bot":bot,
    }

    coll = mydb["session"]
    coll.insert(data,check_keys=False)
    return Response(status=200)

if __name__=="__main__":
    app.run()
