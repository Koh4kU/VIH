from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import os
from nltk.corpus import stopwords
import re
from nltk.stem import SnowballStemmer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, log_loss
from nltk import download
import sys
import nltk
import spacy
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch

#download()
#pip install transformers==2.9
#pip install spacy;pip install nltk; pip install sklearn; pip install transformers; pip install torch;python -m spacy download es_core_news_sm;
def bestModel(resultDataset, file):


    x_train_count=resultDataset[0]
    x_test_counts=resultDataset[1]
    y_train=resultDataset[2]
    y_test=resultDataset[3]


    #Bayes
    nb_model=MultinomialNB()
    #nb_model=Pipeline(steps=[("classifier, nbClassifier")])
    nb_model.fit(x_train_count,y_train)
    y_pred_nb=nb_model.predict(x_test_counts)
    nb_score=nb_model.score(x_test_counts, y_test)
    nb_loss=log_loss(y_test, y_pred_nb, eps=1e-15)


    #Regression
    lrc_model=LogisticRegression(random_state=0, multi_class="auto", solver="lbfgs", max_iter=1000)
    #lrc_model=Pipeline(steps=[("classifier", logisticRegressionClassifier)])
    lrc_model.fit(x_train_count,y_train)
    y_pred_lrc=lrc_model.predict(x_test_counts)
    lrc_score=lrc_model.score(x_test_counts, y_test)
    lrc_loss=log_loss(y_test, y_pred_lrc, eps=1e-15)



    #K-nearest neighbours
    knn_model=KNeighborsClassifier()
    #knn_model=Pipeline(steps=[("classifier", knnClassifier)])
    knn_model.fit(x_train_count, y_train)
    y_pred_knn=knn_model.predict(x_test_counts)
    knn_score=knn_model.score(x_test_counts, y_test)
    knn_loss=log_loss(y_test, y_pred_knn, eps=1e-15)


    #SVC
    svm_model=SVC(kernel="linear", gamma="auto")
    #svm_model=Pipeline(steps=[("classifier", svmClassifier)])
    svm_model.fit(x_train_count, y_train)
    y_pred_svm=svm_model.predict(x_test_counts)
    svm_score=svm_model.score(x_test_counts, y_test)
    svm_loss=log_loss(y_test, y_pred_svm, eps=1e-15)


    #Random forest
    rfc_model=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #rfc_model=Pipeline(steps=[("classifier", randomForestClassifier)])
    rfc_model.fit(x_train_count, y_train)
    y_pred_rfc=rfc_model.predict(x_test_counts)
    rfc_score=rfc_model.score(x_test_counts, y_test)
    rfc_loss=log_loss(y_test, y_pred_rfc, eps=1e-15)

    print(f"""Unique nb->{set(y_test) - set(y_pred_nb)}""")
    print(f"""Unique svm->{set(y_test) - set(y_pred_svm)}""")
    print(f"""Unique lrc->{set(y_test) - set(y_pred_lrc)}""")
    print(f"""Unique knn->{set(y_test) - set(y_pred_knn)}""")
    print(f"""Unique rfc->{set(y_test) - set(y_pred_rfc)}""")

    with open("./results/"+file+"NB.txt", "a") as w:
        w.write(datetime.now().strftime("%Y-%m-%d-%H-%M")+"\n")
        w.write(f"""Random state->{RANDOM_STATE}\n""")
        w.write(f'''Bayes->\n{y_test}\t{y_pred_nb}\t{nb_score}\n{confusion_matrix(y_test, y_pred_nb)}\n{classification_report(y_test, y_pred_nb)}\n
        {accuracy_score(y_test, y_pred_nb)}\n''')
    with open("./results/" + file+"LRC.txt", "a") as w:
        w.write(datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n")
        w.write(f"""Random state->{RANDOM_STATE}\n""")
        w.write(f'''Regression->\n{y_test}\n{y_pred_lrc}\n{lrc_score}\n{confusion_matrix(y_test, y_pred_lrc)}\n{classification_report(y_test, y_pred_lrc)}\n
        {accuracy_score(y_test, y_pred_lrc)}\n''')
    with open("./results/" + file+"SVM.txt", "a") as w:
        w.write(datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n")
        w.write(f"""Random state->{RANDOM_STATE}\n""")
        w.write(
            f'''SVM->\n{y_test}\n{y_pred_svm}\n{svm_score}\n{confusion_matrix(y_test, y_pred_svm)}\n{classification_report(y_test, y_pred_svm)}\n
                {accuracy_score(y_test, y_pred_svm)}\n''')
    with open("./results/" + file+"KNN.txt", "a") as w:
        w.write(datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n")
        w.write(f"""Random state->{RANDOM_STATE}\n""")

        w.write(
            f'''Neighbors->\n{y_test}\n{y_pred_knn}\n{knn_score}\n{confusion_matrix(y_test, y_pred_knn)}\n{classification_report(y_test, y_pred_knn)}\n
                {accuracy_score(y_test, y_pred_knn)}\n''')
    with open("./results/" + file+"RFC.txt", "a") as w:
        w.write(datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n")
        w.write(f"""Random state->{RANDOM_STATE}\n""")
        w.write(
            f'''Random forest->\n{y_test}\n{y_pred_rfc}\n{rfc_score}\n{confusion_matrix(y_test, y_pred_rfc)}\n{classification_report(y_test, y_pred_rfc)}\n
                {accuracy_score(y_test, y_pred_rfc)}\n''')
    with open("./results/scoreAll.txt", "a") as w:
        w.write(f"""{file} Random state->{RANDOM_STATE}\n""")
        w.write(f"""nb_score {nb_score}\nlrc_score {lrc_score}\nsvm_score {svm_score}\nknn_score {knn_score}\nrfc_score {rfc_score}\n\n""")
    with open("./results/lossAll.txt", "a") as w:
        w.write(f"""{file} Random state->{RANDOM_STATE}\n""")
        w.write(f"""nb_loss {nb_loss}\nlrc_loss {lrc_loss}\nsvm_loss {svm_loss}\nknn_score {knn_loss}\nrfc_loss {rfc_loss}\n\n""")


    return (nb_score, lrc_score, svm_score, knn_score, rfc_score)


def datasetLoadFile(vectorizer, flag):
    dataset = datasets.load_files("./resources/dataset/", encoding="utf-8", allowed_extensions=[".txt"],
                                  random_state=45)
    x = dataset.data
    y = dataset.target



    return vectorize(x, y, vectorizer, flag)

def datasetFileBinary(vectorizer, flag):

    VIH=getDatasetBinary()[0]
    no_VIH=getDatasetBinary()[1]

    x, y=[],[]

    x=no_VIH+VIH


    for i in no_VIH:
        y.append(0)
    for i in VIH:
        y.append(1)

    return vectorize(x, y, vectorizer, flag)

def getDatasetBinary():
    VIH=[]
    no_VIH=[]
    temp=""
    temp2=""
    for directory in os.listdir("./resources/dataset/"):
        if directory == "No VIH":
            for file in os.listdir("./resources/dataset/" + directory):
                if file.endswith(".txt"):
                    with open("./resources/dataset/" + directory+"/"+file, encoding="utf-8") as f:
                        temp+=f.read()
                    no_VIH.append(temp)
                temp=""
        else:
            for file in os.listdir("./resources/dataset/" + directory):
                if file.endswith(".txt"):
                    with open("./resources/dataset/" + directory+"/"+file, encoding="utf-8") as f:
                        temp2+=f.read()
                    VIH.append(temp2)
                temp2=""
    #print(VIH)
    #print(no_VIH)

    return (VIH, no_VIH)

def vectorize(x, y, vectorizer, flag):
    list_x = []
    if flag==1 or flag==2:
        x_proccessed=''

        stemmer=SnowballStemmer("spanish")
        stop_words = stopwords.words("spanish")

        for sen in range(0, len(x)):
            x_proccessed = str(x[sen])
            x_proccessed = re.sub(r'\W', ' ', str(x[sen]))
            x_proccessed = re.sub(r'\s+[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\^[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\s+', ' ', x_proccessed, flags=re.I)
            x_proccessed= re.sub(r'^\s+', '', x_proccessed)
            x_proccessed = x_proccessed.lower()

            if flag==2:
                nlp=spacy.load("es_core_news_sm")
                tagged=nlp(x_proccessed)
                x_proccessed=[i.text for i in tagged if (i.pos_!="NOUN" or i.pos_!="ADJ")]
                x_proccessed = [stemmer.stem(i) for i in x_proccessed]
            else:
                x_proccessed = x_proccessed.split()
                x_proccessed = [stemmer.stem(i) for i in x_proccessed]
                x_proccessed = [i for i in x_proccessed if i not in stop_words]
            x_proccessed = ' '.join(x_proccessed)

            list_x.append(x_proccessed)
    elif flag==0:
        #print("xd")
        list_x=x
    #Proyecto de edican named entity recognition with bert
    else:
        tokenizer = AutoTokenizer.from_pretrained("lcampillos/roberta-es-clinical-trials-ner")
        model = AutoModelForTokenClassification.from_pretrained("lcampillos/roberta-es-clinical-trials-ner")
        nerpipeline=pipeline("ner", model=model,tokenizer=tokenizer)
        temp=""
        for i in x:
            for j in nerpipeline(i):
                if j["word"].startswith("Ġ"):
                    temp+=" "+j["word"][1:]
                else:
                    temp+=j["word"]
            list_x.append(temp)
            temp=""

    x_train, x_test, y_train, y_test = train_test_split(list_x, y, test_size=0.4, train_size=0.6, random_state=RANDOM_STATE, stratify=y)

    x_train_count = vectorizer.fit_transform(x_train)
    x_test_counts = vectorizer.transform(x_test)

    return (x_train_count, x_test_counts, y_train, y_test)

def getMax(m1):
    SVM,NB,LRC,KNN,RFC=0,0,0,0,0

    SVM += m1[2]
    NB += m1[0]
    LRC += m1[1]
    KNN += m1[3]
    RFC += m1[4]
    max_string = ""
    ma = max(SVM, NB, LRC, KNN, RFC)
    if ma == SVM:
        max_string = "SVM"
    elif ma == NB:
        max_string = "NB"
    elif ma == LRC:
        max_string = "LRC"
    elif ma == KNN:
        max_string = "KNN"
    else:
        max_string = "RFC"
    return (max_string,ma)

if __name__=="__main__":
    start=datetime.now()
    vectorizer1 = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
    vectorizer2 = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)

    states=[42,45,0,5]

    SVM=0
    NB=0
    KNN=0
    LRC=0
    RFC=0
    max_string = ""

    for i in states:
        RANDOM_STATE=i

        #True si quieres la version stemmed y sin stop words
        #0 texto plano
        #1 texto stemmed y sin stop words
        #2 texto solo adjetivos y nombres
        #3 texto solo enfermedades
    
        resultDataset = datasetFileBinary(vectorizer1, 0)
        m1=bestModel(resultDataset, "model1")
        print(f'''Model1 best-> {i} {getMax(m1)}''')

        resultDataset = datasetFileBinary(vectorizer1, 1)
        m2=bestModel(resultDataset, "model2")
        print(f'''Model2 best-> {i} {getMax(m2)}''')

        resultDataset = datasetFileBinary(vectorizer1, 2)
        m3=bestModel(resultDataset, "model3")
        print(f'''Model3 best-> {i} {getMax(m3)}''')

        resultDataset = datasetFileBinary(vectorizer1, 3)
        m4 = bestModel(resultDataset, "model4")
        print(f'''Model4 best-> {i} {getMax(m4)}''')

        resultDataset = datasetFileBinary(vectorizer2, 0)
        m5=bestModel(resultDataset, "model5")
        print(f'''Model5 best-> {i} {getMax(m5)}''')

        resultDataset = datasetFileBinary(vectorizer2, 1)
        m6=bestModel(resultDataset, "model6")
        print(f'''Model6 best-> {i} {getMax(m6)}''')

        resultDataset = datasetFileBinary(vectorizer2, 2)
        m7=bestModel(resultDataset, "model7")
        print(f'''Model7 best-> {i} {getMax(m7)}''')

        resultDataset = datasetFileBinary(vectorizer2, 3)
        m8 = bestModel(resultDataset, "model8")
        print(f'''Model8 best-> {i} {getMax(m8)}''')

        SVM += m1[2]+m2[2]+m3[2]+m4[2]+m5[2]+m6[2]
        NB += m1[0]+m2[0]+m3[0]+m4[0]+m5[0]+m6[0]
        LRC += m1[1]+m2[1]+m3[1]+m4[1]+m5[1]+m6[1]
        KNN += m1[3]+m2[3]+m3[3]+m4[3]+m5[3]+m6[3]
        RFC += m1[4]+m2[4]+m3[4]+m4[4]+m5[4]+m6[4]
        ma = max(SVM, NB, LRC, KNN, RFC)
    if ma == SVM:
        max_string = "SVM"
    elif ma == NB:
        max_string = "NB"
    elif ma == LRC:
        max_string = "LRC"
    elif ma == KNN:
        max_string = "KNN"
    else:
        max_string = "RFC"

    total_m1=sum(m1)
    total_m2=sum(m2)
    total_m3=sum(m3)
    total_m4=sum(m4)
    total_m5=sum(m5)
    total_m6=sum(m6)
    total_m7=sum(m7)
    total_m8=sum(m8)

    ma2=max(total_m1,total_m2,total_m3,total_m4,total_m5,total_m6,total_m7,total_m8)

    if ma2 == total_m1:
        max2_string = "Model1"
    elif ma2 == total_m2:
        max2_string = "Model2"
    elif ma2 == total_m3:
        max2_string = "Model3"
    elif ma2 == total_m4:
        max2_string = "Model4"
    elif ma2==total_m5:
        max2_string = "Model5"
    elif ma2 == total_m6:
        max2_string="Model6"
    elif ma2 == total_m7:
        max2_string = "Model7"
    else:
        max2_string = "Model8"

    print(f'''Resultado final mejor modelo de media-> {max2_string} {ma2/5}''')
    print(f'''Resultado final mejor algoritmo de media-> {max_string} {ma/(8*len(states))}''')

    end=datetime.now()
    print(f'''Duración-> {end - start}''')

