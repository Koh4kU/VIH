import collections
import pathlib
import tensorflow as tf


import os
import numpy as np
from sklearn.model_selection import train_test_split

import nltk
import spacy
from datetime import datetime

import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import download
from sklearn.pipeline import Pipeline
import torch
#pip install tensorflow;pip install sklearn;pip install transformers;pip install nltk;pip install spacy;pip install torch;
#python -m spacy download es_core_news_sm
#nltk.download()



def deep_learning(flag, random_state):
    data=getDataset()
    VIH,no_VIH=data[0],data[1]

    VIH=preproccess_text(VIH, flag)
    no_VIH=preproccess_text(no_VIH, flag)

    dataset=[]
    for i in no_VIH:
        dataset.append((i, float(0)))
    for i in VIH:
        dataset.append((i, float(1)))

    split_sklearn_x=no_VIH+VIH
    split_sklearn_y=[]
    for i in no_VIH:
        split_sklearn_y.append(0)
    for i in VIH:
        split_sklearn_y.append(1)



    #print(len(split_sklearn_y))

    #dataset_train,dataset_test=tf.keras.utils.split_dataset(np.array(dataset), left_size=0.6, shuffle=True)
    x_train, x_test, y_train, y_test = train_test_split(split_sklearn_x, split_sklearn_y, test_size=0.4, train_size=0.6,
                                                        random_state=random_state, stratify=split_sklearn_y)


    encoder=tf.keras.layers.TextVectorization(max_tokens=1000)

    #encoder.adapt(dataset_train.map(lambda data: data[0]))

    encoder.adapt(x_train)

    #print(encoder.get_vocabulary())

    output_final=2

    model=tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(2)
    ])

    print([layer.supports_masking for layer in model.layers])

    sample= "Esta persona tiene VIH"
    predictions = model.predict([sample])
    print(predictions)

    #tf.keras.losses.BinaryCrossentropy()
    model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"])
    '''
    x_mapped_train=dataset_train.map(lambda data: data[0])
    y_mapped_train=dataset_train.map(lambda data: float(data[1]))
    x_mapped_test = dataset_test.map(lambda data: data[0])
    y_mapped_test = dataset_test.map(lambda data: float(data[1]))

    x_train=list(x_mapped_train.as_numpy_iterator())
    y_train=list(y_mapped_train.as_numpy_iterator())

    x_test=list(x_mapped_test.as_numpy_iterator())
    y_test=list(y_mapped_test.as_numpy_iterator())

    print(type(x_train[0]))

    history = model.fit(x_train, y_train, validation_data=((x_test, y_test)),epochs=10)
    '''
    history = model.fit(np.array(x_train), np.array(y_train),
                        epochs=30, callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],validation_data=(x_test, y_test),batch_size=50)

    test_loss, test_acc=model.evaluate(np.array(["Hola esta persona tiene VIH ya que tiene enfermedades sexuaes, papiloma y fiebre"]), np.array([1]))

    print(f'''Test loss-> {test_loss}''')
    print(f'''Test accuracy-> {test_acc}''')


    return test_acc


def getDataset():
    VIH = []
    no_VIH = []
    temp = ""
    temp2 = ""
    for directory in os.listdir("./resources/dataset/"):
        if directory == "No VIH":
            for file in os.listdir("./resources/dataset/" + directory):
                if file.endswith(".txt"):
                    with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                        temp += f.read()
                    no_VIH.append(temp)
                temp = ""
        else:
            for file in os.listdir("./resources/dataset/" + directory):
                if file.endswith(".txt"):
                    with open("./resources/dataset/" + directory + "/" + file, encoding="utf-8") as f:
                        temp2 += f.read()
                    VIH.append(temp2)
                temp2 = ""
    # print(VIH)
    # print(no_VIH)

    return (VIH, no_VIH)

def preproccess_text(x, flag):
    list_x = []
    if flag == 1 or flag == 2:
        x_proccessed = ''

        stemmer = SnowballStemmer("spanish")
        stop_words = stopwords.words("spanish")

        for sen in range(0, len(x)):
            x_proccessed = str(x[sen])
            x_proccessed = re.sub(r'\W', ' ', str(x[sen]))
            x_proccessed = re.sub(r'\s+[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\^[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\s+', ' ', x_proccessed, flags=re.I)
            x_proccessed = re.sub(r'^\s+', '', x_proccessed)
            x_proccessed = x_proccessed.lower()

            if flag == 2:
                nlp = spacy.load("es_core_news_sm")
                tagged = nlp(x_proccessed)
                x_proccessed = [i.text for i in tagged if (i.pos_ != "NOUN" or i.pos_ != "ADJ")]
                x_proccessed = [stemmer.stem(i) for i in x_proccessed]
            else:
                x_proccessed = x_proccessed.split()
                x_proccessed = [stemmer.stem(i) for i in x_proccessed]
                x_proccessed = [i for i in x_proccessed if i not in stop_words]
            x_proccessed = ' '.join(x_proccessed)

            list_x.append(x_proccessed)
    elif flag == 0:
        # print("xd")
        list_x = x
    # Proyecto de edican named entity recognition with bert
    else:
        tokenizer = AutoTokenizer.from_pretrained("lcampillos/roberta-es-clinical-trials-ner")
        model = AutoModelForTokenClassification.from_pretrained("lcampillos/roberta-es-clinical-trials-ner")
        nerpipeline = pipeline("ner", model=model, tokenizer=tokenizer)
        temp = ""
        for i in x:
            for j in nerpipeline(i):
                if j["word"].startswith("Ġ"):
                    temp += " " + j["word"][1:]
                else:
                    temp += j["word"]
            list_x.append(temp)
            temp = ""

    return list_x

if __name__=="__main__":

    start = datetime.now()

    states = [45, 42, 5, 0]
    model1, model2, model3, model4=0,0,0,0

    for i in states:
        model1+=deep_learning(0, i)
        model2+=deep_learning(1, i)
        model3+=deep_learning(2, i)
        model4+=deep_learning(3, i)
    m=max(model1,model2, model3, model4)
    if m==model1:
        best_model="model1"
    elif m==model2:
        best_model="model2"
    elif m==model3:
        best_model="model3"
    else:
        best_model="model4"

    result=m/len(states)
    print(f'''Best model {best_model}-> {result}''')
    end=datetime.now()
    print(f'''Duration -> {end - start}''')