import tensorflow as tf
import tflearn
import numpy as np
import random, json
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer


def preprocessing():
    filePath = 'intents.json'
    with open(filePath, 'rb') as f:
        loader = json.load(f)
    
    data = loader['intents']
    words = []
    labels = sorted([item['tag'].lower() for item in data])
    doc_x = []
    doc_y = []
    stemmer = LancasterStemmer()

    for item in data:
        for pattern in item['patterns']:
            words.extend(word_tokenize(pattern))
            doc_x.append([stemmer.stem(token) for token in word_tokenize(pattern.lower()) if token not in '?'])
            doc_y.append(item['tag'])

    words = set([stemmer.stem(word.lower()) for word in words if word not in '?'])
    training, output = word2vec(doc_x, doc_y, words, labels)
    return np.array(training), np.array(output), words, labels, data


def word2vec(doc_x, doc_y, words, labels):
    vec_x = []
    vec_y = []

    for idx, doc in enumerate(doc_x):
        vec_x.append([1 if w in doc else 0 for w in words])
        vec_y.append([1 if label == doc_y[idx] else 0 for label in labels])
    
    return vec_x, vec_y

def create_model(training, output, words, labels):
    tf.reset_default_graph()

    net = tflearn.input_data(shape=[None, len(words)])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(labels), activation='softmax')
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)

    return model

def mess2vec(mess, words):
    stemmer = LancasterStemmer()
    tokens = word_tokenize(mess.lower())
    tokens = [stemmer.stem(token) for token in tokens]

    return np.array([1 if w in tokens else 0 for w in words])

def main():
    print("-"*30)
    print("CHATBOT FOR EVERYONE!")
    print("-"*30)

    training, output, words, labels, data = preprocessing()
    model = create_model(training, output, words, labels)
    while True:
        mess = input('You: ')

        if mess == 'close':
            break

        inp = mess2vec(mess, words)
        preds = model.predict([inp])
        print(np.max(preds))                
        if np.max(preds) > 0.8:
            result = labels[np.argmax(preds)]
            responses = []
            for item in data:
                if item['tag'] == result:
                    responses = item['responses']
            res = random.choice(responses)
            print('Bot: {0}'.format(res))
        else:
            print("Bot: Sorry, I don't understand ^^'. Please try again or ask another question.")

if __name__ == '__main__':
    main()
