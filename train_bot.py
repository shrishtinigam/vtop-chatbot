"""
pip install tensorflow
pip install Keras
pip install pickle-mixin
pip install nltk

python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
quit()
"""
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD 

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.']

data_file = open('intents.json').read()

intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        # print(f'Token is: {w}')
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

"""            
print(words)
print('\n\n\n')
print(documents)
print('\n\n\n')
print(classes)
"""

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = list(set(words))
print(words)
classes = list(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))

training = []
output_empty = [0 for i in range(len(classes))]

for doc in documents:
    bag = []
    pattern_words = doc[0] # doc[0] => bag of words, d[1] => tag
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # print(f'Pattern words: {pattern_words}')

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    # print(f'Cur bag: {bag}')

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    # print(f'Cur Output: {output_row}')

    training.append([bag, output_row])

random.shuffle(training)

training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))

# define optimizer function

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #schotastic gradient descent
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

mfit = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', mfit)