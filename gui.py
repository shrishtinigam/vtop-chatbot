import tkinter
from tkinter import *
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
import random

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

def bow(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    # convert to bag
    bag = [0 for i in range(len(words))]
    for s in sentence_words:
        for i, w in enumerate(words):
            if(w == s):
                bag[i] = 1
    return (np.array(bag))

def predict_class(sentence):
    sentence_bag = bow(sentence)
    res = model.predict(np.array([sentence_bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':r[1]})
    print(return_list)
    return return_list

def getresponse(ints):
    list_of_intents = intents['intents']
    if(ints[0]['probability'] < 0.70):
        for tags in list_of_intents:
                if(tags['tag'] == 'noanswer'):
                    result = random.choice(tags['responses'])
                    return result
    tag = ints[0]['intent']
    result = ""
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg)
    res = getresponse(ints)
    return res

def send():
    msg = TextEntryBox.get('1.0', 'end-1c').strip()
    TextEntryBox.delete('1.0', 'end')

    if msg != '':
        ChatHistory.config(state=NORMAL)
        ChatHistory.insert('end', 'You: ' + msg + '\n\n')

        res = chatbot_response(msg)
        print(res)
        ChatHistory.insert('end', 'Bot: ' + res + '\n\n')
        ChatHistory.config(state = DISABLED)
        ChatHistory.yview('end')

base = Tk()
base.title("VTOP \n Virtual Assistant")
base.geometry("400x500")
base.resizable(width = False, height = False)
# w = Label(base, text="Hello, world!")

ChatHistory = Text(base, bd = 0, bg = 'white', font = 'Arial')
ChatHistory.config(state=DISABLED)
SendButton = Button(base, font = ('Arial', 13, 'bold'), text = "Send", bg="#3376a5", activebackground="#5190bd", fg = "#ffffff", command=send)
TextEntryBox = Text(base, bd=0, bg='white', font='Arial')

ChatHistory.place(x=6, y=6, height=386, width=386)
TextEntryBox.place(x=128, y=400, height=80, width=265)
SendButton.place(x=6, y=400, height=80, width=125)

base.mainloop()

