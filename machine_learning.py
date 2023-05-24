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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def bestModel(resultDataset):




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

    #Regression
    lrc_model=LogisticRegression(random_state=0, multi_class="auto", solver="lbfgs", max_iter=1000)
    #lrc_model=Pipeline(steps=[("classifier", logisticRegressionClassifier)])
    lrc_model.fit(x_train_count,y_train)
    y_pred_lrc=lrc_model.predict(x_test_counts)
    lrc_score=lrc_model.score(x_test_counts, y_test)

    #K-nearest neighbours
    knn_model=KNeighborsClassifier()
    #knn_model=Pipeline(steps=[("classifier", knnClassifier)])
    knn_model.fit(x_train_count, y_train)
    y_pred_knn=knn_model.predict(x_test_counts)
    knn_score=knn_model.score(x_test_counts, y_test)

    #SVC
    svm_model=SVC(kernel="linear", gamma="auto")
    #svm_model=Pipeline(steps=[("classifier", svmClassifier)])
    svm_model.fit(x_train_count, y_train)
    y_pred_svm=svm_model.predict(x_test_counts)
    svm_score=svm_model.score(x_test_counts, y_test)

    #Random forest
    rfc_model=RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    #rfc_model=Pipeline(steps=[("classifier", randomForestClassifier)])
    rfc_model.fit(x_train_count, y_train)
    y_pred_rfc=rfc_model.predict(x_test_counts)
    rfc_score=rfc_model.score(x_test_counts, y_test)

    #Decision tree


    #Preprocessor
    #("preprocessor", preprocessorForFeatures)
    #("preprocessor", preprocessorForCategoricalColumns)

    flag_bayes=False
    flag_neighbors=False
    for i in y_pred_nb:
        if i ==1:
            flag_bayes=True
            break
    for i in y_pred_knn:
        if i == 0:
            flag_neighbors=True
            break

    '''
    print(f"""Bayes->\n{y_pred_nb}""")
    print(f"""Regression->\n{y_pred_lrc}""")
    print(f"""SVM->\n{y_pred_svm}""")
    print(f"""Neighbors->\n{y_pred_knn}""")
    print(f"""Random forest->\n{y_pred_rfc}""")
    '''
    if flag_bayes:
        print(f"""\tBayes->{nb_score}""")
        print(confusion_matrix(y_test, y_pred_nb))
        print(classification_report(y_test, y_pred_nb))
        print(accuracy_score(y_test, y_pred_nb))
    print(f"""\tRegression->{lrc_score}""")
    print(confusion_matrix(y_test, y_pred_lrc))
    print(classification_report(y_test, y_pred_lrc))
    print(accuracy_score(y_test, y_pred_lrc))
    print(f"""\tSVM->{svm_score}""")
    print(confusion_matrix(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))
    print(accuracy_score(y_test, y_pred_svm))
    if flag_neighbors:
        print(f"""\tNeighbors->{knn_score}""")
        print(confusion_matrix(y_test, y_pred_knn))
        print(classification_report(y_test, y_pred_knn))
        print(accuracy_score(y_test, y_pred_knn))
    print(f"""\tRandom forest->{rfc_score}""")
    print(confusion_matrix(y_test, y_pred_rfc))
    print(classification_report(y_test, y_pred_rfc))
    print(accuracy_score(y_test, y_pred_rfc))


    m=0
    m=max(nb_score, lrc_score, svm_score, knn_score, rfc_score)


    '''
    match max:
        case nb_score:
            return ("Bayes", nb_score)
        case lrc_score:
            return ("Regression", lrc_score)
        case svm_score:
            return ("SVM", svm_score)
        case knn_score:
            return ("Neighbors", knn_score)
        case rfc_score:
            return ("Random forest", rfc_score)
    '''



    if m==nb_score:
        return ("Bayes", nb_score)
    elif m==lrc_score:
        return ("regression", lrc_score)
    elif m==svm_score:
        return ("SVM", svm_score)
    elif m==knn_score:
        return ("Neighbors", knn_score)
    elif m==rfc_score:
        return ("Random forest", rfc_score)



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
                    VIH.append(temp)
                temp=""
    #print(VIH)
    #print(no_VIH)

    return (VIH, no_VIH)

def vectorize(x, y, vectorizer, flag):
    list_x = []
    if flag:
        x_proccessed=''

        stemmer=SnowballStemmer("spanish")
        stop_words = stopwords.words("spanish")

        for sen in range(0, len(x)):
            x_proccessed = re.sub(r'\W', ' ', str(x[sen]))
            x_proccessed = re.sub(r'\s+[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\^[a-zA-Z]\s+', ' ', x_proccessed)
            x_proccessed = re.sub(r'\s+', ' ', x_proccessed, flags=re.I)
            x_proccessed= re.sub(r'^\s+', '', x_proccessed)
            x_proccessed = x_proccessed.lower()
            x_proccessed = x_proccessed.split()

            x_proccessed = [stemmer.stem(i) for i in x_proccessed]
            x_proccessed = [i for i in x_proccessed if i not in stop_words]
            x_proccessed = ' '.join(x_proccessed)

            list_x.append(x_proccessed)
    else:
        list_x=x


    x_train, x_test, y_train, y_test = train_test_split(list_x, y, test_size=0.4, train_size=0.6, random_state=45, stratify=y)
    print(len(list_x))
    print(len(y))


    x_train_count = vectorizer.fit_transform(x_train)
    x_test_counts = vectorizer.transform(x_test)

    return (x_train_count, x_test_counts, y_train, y_test)
if __name__=="__main__":
    vectorizer1 = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
    vectorizer2 = TfidfVectorizer(max_features=1500, min_df=5, max_df=0.7)
    print("Model1 CountVectorizer y stemmed")
    #True si quieres la version stemmed y sin stop words
    resultDataset = datasetFileBinary(vectorizer1, True)
    bestModel(resultDataset)
    print("Model2 CountVectorizer y no stemmed")
    resultDataset2 = datasetFileBinary(vectorizer1, False)
    bestModel(resultDataset2)
    print("Model3 TfidfVectorizer y stemmed")
    resultDataset = datasetFileBinary(vectorizer2, True)
    bestModel(resultDataset)
    print("Model4 TfidfVectorizer y no stemmed")
    resultDataset2 = datasetFileBinary(vectorizer2, False)
    bestModel(resultDataset2)







