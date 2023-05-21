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

def bestModel(resultDataset):




    x_train_count=resultDataset[0]
    x_test_counts=resultDataset[1]
    y_train=resultDataset[2]
    y_test=resultDataset[3]

    #Bayes
    nb_model=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
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
    knn_model=KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
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

    print(f"""Bayes->\n{y_pred_nb}""")
    print(f"""Regression->\n{y_pred_lrc}""")
    print(f"""SVM->\n{y_pred_svm}""")
    print(f"""Neighbors->\n{y_pred_knn}""")
    print(f"""Ensemble->\n{y_pred_rfc}""")

    print(f"""Bayes->{nb_score}""")
    print(f"""Regression->{lrc_score}""")
    print(f"""SVM->{svm_score}""")
    print(f"""Neighbors->{knn_score}""")
    print(f"""Ensemble->{rfc_score}""")

    return max(nb_score, lrc_score, svm_score, knn_score, rfc_score)


def datasetLoadFile():
    dataset = datasets.load_files("./resources/dataset/", encoding="utf-8", allowed_extensions=[".txt"],
                                  random_state=42)

    x = dataset.data
    y = dataset.target



    return vectorize(x, y)

def datasetFileBinary():

    VIH=getDatasetBinary()[0]
    no_VIH=getDatasetBinary()[1]

    x, y=[],[]

    x=no_VIH+VIH
    for i in no_VIH:
        y.append(0)
    for i in VIH:
        y.append(1)

    return vectorize(x, y)

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
                temp=""
    #print(VIH)
    #print(no_VIH)

    return (VIH, no_VIH)

def vectorize(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.6, random_state=42,
                                                        stratify=y)
    print(y_test)

    tfid = TfidfVectorizer()
    x_train_count = tfid.fit_transform(x_train)
    x_test_counts = tfid.transform(x_test)

    return (x_train_count, x_test_counts, y_train, y_test)
if __name__=="__main__":
    resultDataset = datasetFileBinary()
    bestModel(resultDataset)







