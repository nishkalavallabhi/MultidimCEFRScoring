import os
import time
import csv
import re
import json
import numpy as np
import collections
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from skmultilearn.adapt import MLkNN, MLTSVM, MLARAM
from skmultilearn.problem_transform import LabelPowerset,BinaryRelevance
from skmultilearn.ensemble import RakelD,RakelO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.model_selection import IterativeStratification

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from Baseline import getLangData, get_metadata_file, getScoringFeatures, getlangslist


classifiers = [MLkNN(k=5), MLARAM(), 
	       LabelPowerset(classifier=RandomForestClassifier()),BinaryRelevance(classifier=RandomForestClassifier()),
               RakelD(base_classifier=RandomForestClassifier())] #Add more

#This should return list of lists. Each list item is a list of labels for that text.
def getcatlist(filenameslist, lang):
    if lang.upper() in ["DE", "IT", "CZ"]:
        langmetadata = get_metadata_file(lang) #Dict. key is filename, value: list of all CEFR ratings
        result = []
        for name in filenameslist:
            if lang.upper() == "CZ": #file naming patterns in MERLIN are different for DE and IT compared to CZ
                actual_name = name.split("_")[0]
            else:
                actual_name = "_".join(name.split("_")[:2])
            filecats = langmetadata[actual_name]
            for ind, cat in enumerate(filecats):
                if ":0" in cat:
                    filecats[ind] = cat.replace(":0",":A1")
                elif "-1" in cat:
                    filecats[ind] = cat.replace("-1","A1")
            result.append(filecats)
    else:
        print("Wrong language entered")
        exit(1)
    return result

#Training on one language data, Stratified 10 fold CV
def singlelang_ngrams_multilabel(train_labels,train_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10)
    vectorizers = [uni_to_tri_vectorizer]
    #k_fold = StratifiedKFold(10,random_state=1234) #This does not work for multi-label problems.
    #kf = KFold(n_splits=10, shuffle=True, random_state=1234)
    kf = IterativeStratification(n_splits=10, random_state=1234) #stratified KFold implemented for multi-label probs.
    mlb = MultiLabelBinarizer()
    binarized_labels = mlb.fit_transform(train_labels)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            train_vector = vectorizer.fit_transform(train_data).toarray()
            cross_val = cross_val_score(classifier, train_vector, binarized_labels, cv=kf, n_jobs=1)
            predicted = cross_val_predict(classifier, train_vector, binarized_labels, cv=kf, n_jobs=1)
            print(sum(cross_val)/float(len(cross_val)), f1_score(binarized_labels,predicted,average='weighted'))
            print(classification_report(binarized_labels, predicted, target_names = mlb.classes_))
            print("DONE FOR THIS CLASSIFIER")
    print("SAME LANG EVAL DONE FOR THIS LANG")

#Single language, 10 fold cv for domain features - i.e., non n-gram features.
def singlelang_domain_multilabel(train_vector,train_labels): #test_vector,test_labels):
    kf = IterativeStratification(n_splits=10, random_state=1234) #stratified KFold implemented for multi-label probs.
    mlb = MultiLabelBinarizer()
    binarized_labels = mlb.fit_transform(train_labels)
    for classifier in classifiers:
        print("Printing results for: " + str(classifier))
        cross_val = cross_val_score(classifier, np.array(train_vector), binarized_labels, cv=kf)
        predicted = cross_val_predict(classifier, np.array(train_vector), binarized_labels, cv=kf)
        print(sum(cross_val)/float(len(cross_val)), f1_score(binarized_labels,predicted,average='weighted'))
        print(classification_report(binarized_labels, predicted, target_names = mlb.classes_))

def singlelag_allfeats_multilabel(langdirpath,lang):
    langfiles,langwordngrams = getLangData(langdirpath, "word")
    langfiles,langposngrams = getLangData(langdirpath, "pos")
    langfiles,langdepngrams = getLangData(langdirpath, "dep")
    if not lang == "cz":
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,True)
    else:
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,False)

    print("Extracted all features: ")
    langlabels = getcatlist(langfiles,lang)

    print("With Word ngrams:", "\n", "******")
    singlelang_ngrams_multilabel(langlabels,langwordngrams)
    print("*************")
    print("With POS ngrams: ", "\n", "******")
    singlelang_ngrams_multilabel(langlabels,langposngrams)
    print("*************")
    print("Dep ngrams: ", "\n", "******")
    singlelang_ngrams_multilabel(langlabels,langdepngrams)
    print("*************")
    print("Domain features: ", "\n", "******")
    singlelang_domain_multilabel(langdomain,langlabels)
    print("*************")

    """
    print("Combined feature rep: wordngrams + domain")
    combine_features(langlabels,langwordngrams,langdomain)
    print("Combined feature rep: posngrams + domain")
    combine_features(langlabels,langposngrams,langdomain)
    print("Combined feature rep: depngrams + domain")
    combine_features(langlabels,langdepngrams,langdomain)
    """

#cross lingual classification evaluation for non ngram features
def crosslang_domain_multilabel(train_vector, train_labels, test_vector, test_labels):
    print("CROSS LANG EVAL")
    classifiers = [MLkNN(k=5), MLARAM(),LabelPowerset(classifier=RandomForestClassifier()),RakelD(base_classifier=RandomForestClassifier())] #Add more later
    mlb = MultiLabelBinarizer()
    binarized_labels = mlb.fit_transform(train_labels + test_labels)
    for classifier in classifiers:
        print("Printing results for: " + str(classifier))
        classifier.fit(np.array(train_vector),binarized_labels[:len(train_labels)])
        predicted = classifier.predict(np.array(test_vector))
        print(f1_score(binarized_labels[len(train_labels):],predicted,average='weighted'))
        print(classification_report(binarized_labels[len(train_labels):], predicted, target_names = mlb.classes_))


def crosslang_ngrams_multilabel(train_labels,train_data, test_labels, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10) #, max_features = 2000
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [MLkNN(k=5), MLARAM(), LabelPowerset(classifier=RandomForestClassifier()),RakelD(base_classifier=RandomForestClassifier())] #Add more later
    mlb = MultiLabelBinarizer()
    binarized_labels = mlb.fit_transform(train_labels+test_labels)
    temp = len(train_labels)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
            text_clf.fit(train_data,binarized_labels[:temp])
            predicted = text_clf.predict(test_data)
            print(f1_score(binarized_labels[temp:],predicted,average='weighted'))
            print(classification_report(binarized_labels[temp:], predicted, target_names = mlb.classes_))

def crosslang_multilabel(sourcelangdirpath,sourcelang, targetlangdirpath, targetlang):
   #Read source language data
   sourcelangfiles,sourcelangposngrams = getLangData(sourcelangdirpath, "pos")
   sourcelangfiles,sourcelangdepngrams = getLangData(sourcelangdirpath, "dep")
   #Read target language data
   targetlangfiles,targetlangposngrams = getLangData(targetlangdirpath, "pos")
   targetlangfiles,targetlangdepngrams = getLangData(targetlangdirpath, "dep")
   #Get label info
   sourcelanglabels = getcatlist(sourcelangfiles,sourcelang)
   targetlanglabels = getcatlist(targetlangfiles,targetlang)

   if "cz" not in [sourcelang.lower(), targetlang.lower()]:
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,True)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,True)
   else:
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,False)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,False)
   print("Printing cross-corpus classification evaluation results: ")

   print("*******", "\n", "Setting - Train with: ", sourcelang, " Test with: ", targetlang, " ******", "\n")
   print("Features: pos")
   crosslang_ngrams_multilabel(sourcelanglabels,sourcelangposngrams, targetlanglabels, targetlangposngrams)
   print("Features: dep")
   crosslang_ngrams_multilabel(sourcelanglabels,sourcelangdepngrams, targetlanglabels, targetlangdepngrams)
   print("Features: domain")
   crosslang_domain_multilabel(sourcelangdomain,sourcelanglabels,targetlangdomain,targetlanglabels)

def multilingual_multilabel_allfeats(lang1path,lang1,lang2path,lang2,lang3path,lang3, setting):
   print("Doing: take all data as if it belongs to one large dataset, and do classification")
   if not setting == "domain":
      lang1files,lang1features = getLangData(lang1path,setting)
      lang2files,lang2features = getLangData(lang2path,setting)
      lang3files,lang3features = getLangData(lang3path,setting)

   else: #i.e., domain features only.
      lang1files,lang1features = getScoringFeatures(lang1path,lang1,False)
      lang2files,lang2features = getScoringFeatures(lang2path,lang2,False)
      lang3files,lang3features = getScoringFeatures(lang3path,lang3,False)

   lang1labels = getcatlist(lang1files, lang1)
   lang2labels = getcatlist(lang2files, lang2)
   lang3labels = getcatlist(lang3files, lang3)

   megalabels = []
   megalabels = lang1labels + lang2labels + lang3labels
   megalangs = getlangslist(lang1files) + getlangslist(lang2files) + getlangslist(lang3files)
   megadata = lang1features + lang2features + lang3features
   print("Mega classification for: ", setting, " features")
   if setting == "domain":
      singlelang_domain_multilabel(megadata,megalabels)
   else:
      singlelang_ngrams_multilabel(megalabels,megadata)

def main():
    itdirpath = "/home/bangaru/CrossLingualScoring/Datasets/IT-Parsed"
    dedirpath = "/home/bangaru/CrossLingualScoring/Datasets/DE-Parsed"
    czdirpath = "/home/bangaru/CrossLingualScoring/Datasets/CZ-Parsed"
    #singlelag_allfeats_multilabel(dedirpath, "DE")
    crosslang_multilabel(dedirpath, "de", itdirpath, "it")
    #multilingual_multilabel_allfeats(dedirpath,"de",itdirpath,"it",czdirpath,"cz","dep")

if __name__ == "__main__":
    main()
