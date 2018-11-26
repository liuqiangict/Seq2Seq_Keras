
import os
import sys

from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Model
from keras.models import load_model
from keras.models import model_from_json
from keras.utils import plot_model

import pandas as pd
import numpy as np


os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'

def create_model(n_input, n_output, n_units):
    encoder_input = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    _, encoder_h, encoder_c = encoder(encoder_input)
    encoder_state = [encoder_h, encoder_c]

    decoder_input = Input(shape=(None, n_output))
    decoder = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_output = decoder_dense(decoder_output)

    model = Model([encoder_input, decoder_input], decoder_output)

    encoder_infer = Model(encoder_input, encoder_state)

    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_state_input = [decoder_state_input_h, decoder_state_input_c]

    decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input, initial_state=decoder_state_input)
    decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]
    decoder_infer_output = decoder_dense(decoder_infer_output)
    decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)

    return model, encoder_infer, decoder_infer


N_UNITS = 256
BATCH_SIZE = 64
EPOCH = 200
NUM_SAMPLES = 20
data_path = '.\data\cmn.txt'

df = pd.read_table(data_path, header=None).iloc[:NUM_SAMPLES,:,]
df.columns = ['inputs', 'targets']

df['targets'] = df['targets'].apply(lambda x : '\t' + x + '\n')

input_texts = df.inputs.values.tolist()
target_texts = df.targets.values.tolist()

input_characters = sorted(list(set(df.inputs.unique().sum())))
target_characters = sorted(list(set(df.targets.unique().sum())))

INUPT_LENGTH = max([len(i) for i in input_texts])
OUTPUT_LENGTH = max([len(i) for i in target_texts])
INPUT_FEATURE_LENGTH = len(input_characters)
OUTPUT_FEATURE_LENGTH = len(target_characters)

encoder_input = np.zeros((NUM_SAMPLES, INUPT_LENGTH, INPUT_FEATURE_LENGTH))
decoder_input = np.zeros((NUM_SAMPLES, OUTPUT_LENGTH, OUTPUT_FEATURE_LENGTH))
decoder_output = np.zeros((NUM_SAMPLES,OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH))

input_dict = {char:index for index,char in enumerate(input_characters)}
input_dict_reverse = {index:char for index,char in enumerate(input_characters)}
target_dict = {char:index for index,char in enumerate(target_characters)}
target_dict_reverse = {index:char for index,char in enumerate(target_characters)}


for seq_index, seq in enumerate(input_texts):
    for char_index, char in enumerate(seq):
        encoder_input[seq_index,char_index,input_dict[char]] = 1

for seq_index,seq in enumerate(target_texts):
    for char_index,char in enumerate(seq):
        decoder_input[seq_index,char_index,target_dict[char]] = 1.0
        if char_index > 0:
            decoder_output[seq_index,char_index-1,target_dict[char]] = 1.0

print(''.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i) !=0]))

print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i) !=0]))

print(''.join([target_dict_reverse[np.argmax(i)] for i in decoder_input[0] if max(i) !=0]))


model_train, encoder_infer, decoder_infer = create_model(INPUT_FEATURE_LENGTH, OUTPUT_FEATURE_LENGTH, N_UNITS)

plot_model(to_file='model.png',model=model_train,show_shapes=True)
plot_model(to_file='encoder.png',model=encoder_infer,show_shapes=True)
plot_model(to_file='decoder.png',model=decoder_infer,show_shapes=True)


model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model_train.summary()
encoder_infer.summary()
decoder_infer.summary()

model_file_json = 'model.json'
encoder_model_file_json = 'encoder_model.json'
decoder_model_file_json = 'decoder_model.json'
model_file = 'model.h5'
encoder_model_file = 'encoder_model.h5'
decoder_model_file = 'decoder_model.h5'

def SaveModel(model, filename):
    model_json = model.to_json()
    with open(filename, "w") as json_file:
        json_file.write(model_json)

def LoadModel(filename):
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    return model

if sys.argv[1] == 'Train':
    model_train.fit([encoder_input, decoder_input], decoder_output, batch_size=BATCH_SIZE, epochs=EPOCH, validation_split=0.2)
    SaveModel(model_train, model_file_json)
    SaveModel(encoder_infer, encoder_model_file_json)
    SaveModel(decoder_infer, decoder_model_file_json)
    model_train.save(model_file)
    encoder_infer.save(encoder_model_file)
    decoder_infer.save(decoder_model_file)

elif sys.argv[1] == 'Test':
    # load json and create model
    model_train = LoadModel(model_file_json)
    encoder_infer = LoadModel(encoder_model_file_json)
    decoder_infer = LoadModel(decoder_model_file_json)
    model_train = load_model(model_file)
    encoder_infer = load_model(encoder_model_file)
    decoder_infer = load_model(decoder_model_file)

def predict_chinese(source, encoder_inference, decoder_inference, n_steps, features):
    state = encoder_inference.predict(source)
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1

    output = ''
    for i in range(n_steps):
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':
            break
    return output

for i in range(0,10):
    test = encoder_input[i:i+1,:,:]
    out = predict_chinese(test, encoder_infer, decoder_infer, OUTPUT_LENGTH,OUTPUT_FEATURE_LENGTH)
    print(input_texts[i])
    print(out)
