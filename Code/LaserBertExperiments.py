"""
Using features based on LASER and BERT models to train non-neural classifiers 
for single/multi/cross language classification setups, 
for all dimensions of language proficiency
"""
from Baseline import get_metadata_file, getcatlist, getlangslist

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.impute import SimpleImputer #to replace NaN with mean values.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold, cross_validate
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
#from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC
from datetime import datetime


from scipy.stats import spearmanr, pearsonr
#import language_check

import sys

seed=1234
np.random.seed(seed)

"""
Get laser embeddings features from the feature file for all languages.
return: fileids list, langs list, features list.
"""
def getfeats(lembedspath):
    fileids = [] #get this without lang, cat, .txt info. 
    langs = [] #get only lang code. not sure if this is actually needed yet. 
    features = []
    #../Datasets/DE/1091_0000079_DE_B1.txt
    fh = open(lembedspath, "r")
    for line in fh:
      fileid, feats = line.split("\t")
      features.append([float(feat) for feat in feats.split(",")])
      fileids.append(fileid.split("/")[3]) #"_".join(fileid.split("/")[3].split("_",2)[:2]))  #Workds for DE and IT.
      #Above line only takes 1091_000079 from the full string.
      langs.append(fileid.split("/")[2])
    return fileids, langs, features

#a subset of the above function, by language. So that has to run first to run this.
#the first three arguments to this function come from the output of getfeats()
def getfeatsandids_bylang(fileids, langs, feats, lang):
    finalfeats = []
    finalids = []
    print(len(feats[1]))
    for i in range(0,len(langs)):
        if langs[i]  == lang:
            finalfeats.append(feats[i])
            finalids.append(fileids[i])
    return finalids, finalfeats


#feats come from getfeatsandids_bylang. labels come from getcatslist function imported from baseline, and change by dimension.
def do_classification_cv(feats, labels):

    k_fold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True) #randomization of order is needed I guess?
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed, solver = "lbfgs", multi_class ="multinomial")]
 #Add more later

    for classifier in classifiers:
        print(classifier)
#        cross_val = cross_val_score(classifier, feats, labels, cv=k_fold, n_jobs=1)
#        predicted = cross_val_predict(classifier, feats, labels, cv=k_fold)
#        print(cross_val)
#        print(sum(cross_val)/float(len(cross_val)))
#        print(confusion_matrix(labels, predicted))
#        print(f1_score(labels,predicted,average='weighted'))

####CODE to TEST the default implementation in SKLEARN. !!! WORKS !!!####
        weighted_f1_scores = []
        for i, indices in enumerate(k_fold.split(feats, labels)):
            train_index, test_index = indices
            feats_train = [feats[x] for x in train_index]
            labels_train = [labels[x] for x in train_index]

            feats_test = [feats[x] for x in test_index]
            labels_test = [labels[x] for x in test_index]

            classifier.fit(feats_train,labels_train)
            predicted = classifier.predict(feats_test)
            wt_f1_score = f1_score(labels_test,predicted,average='weighted')
            weighted_f1_scores.append(wt_f1_score)
            print("Fold {}".format(i))
            print(confusion_matrix(labels_test,predicted, labels = sorted(list(set(labels_test)))))
            print(classification_report(labels_test, predicted))
            print()

        print("K-fold scores",weighted_f1_scores,sep="\n")
        print("SKF f1 score mean {}".format(np.array(weighted_f1_scores).mean()))
        print()
#        cross_val_scores = cross_validate(classifier, feats, labels,  cv=k_fold, scoring=('f1_weighted', 'f1_macro'))
#        print(cross_val_scores['test_f1_weighted'])        
#        print(cross_val_scores['test_f1_weighted'].mean())

def do_classification_withtest(train_vector, train_labels, test_vector, test_labels):
    print("CROSS LANG EVAL")
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed, solver = "lbfgs", multi_class ="multinomial")]
    for classifier in classifiers:
        classifier.fit(train_vector,train_labels)
        predicted = classifier.predict(test_vector)
        print(np.mean(predicted == test_labels,dtype=float))
        print(confusion_matrix(test_labels,predicted))
        print(f1_score(test_labels,predicted,average='weighted'))
        print(classification_report(test_labels, predicted))
        print()


def do_single_lang(lembedspath):
    fileids,langs,features = getfeats(lembedspath)
    dimensions = ["OverallCEFRrating", "Grammaticalaccuracy", "Orthography","Vocabularyrange","Vocabularycontrol",
		"CoherenceCohesion", "Sociolinguisticappropriateness"]
    for lang in ["DE", "IT", "CZ"]:
       print("FOr lang: ", lang)
       langfileids, langfeats = getfeatsandids_bylang(fileids,langs,features,lang)
       for dimension in dimensions:
          print("************for dimension: ", dimension, " ***************")
          langlabels = getcatlist(langfileids,dimension,lang)
          #print(langlabels)
          print("Printing class statistics")
          print(collections.Counter(langlabels))
          do_classification_cv(langfeats,langlabels)

def do_multi_ling(lembedspath):
    fileids,langs,features = getfeats(lembedspath)
    dimensions = ["OverallCEFRrating", "Grammaticalaccuracy", "Orthography","Vocabularyrange","Vocabularycontrol",
		"CoherenceCohesion", "Sociolinguisticappropriateness"]
    defileids, defeats = getfeatsandids_bylang(fileids, langs, features, "DE")
    itfileids, itfeats = getfeatsandids_bylang(fileids, langs, features, "IT")
    czfileids, czfeats =  getfeatsandids_bylang(fileids, langs, features, "CZ")
    allfeats = defeats + itfeats+ czfeats
    for dimension in dimensions:
        print("************for dimension: ", dimension, " ***************")
        delabels = getcatlist(defileids,dimension, "DE")
        itlabels = getcatlist(itfileids,dimension, "IT")
        czlabels = getcatlist(czfileids,dimension, "CZ")
        alllabs = delabels + itlabels + czlabels
        do_classification_cv(allfeats, alllabs)

def do_cross_ling(lembedspath):
    fileids,langs,features = getfeats(lembedspath)
    dimensions = ["OverallCEFRrating", "Grammaticalaccuracy", "Orthography","Vocabularyrange","Vocabularycontrol",
		"CoherenceCohesion", "Sociolinguisticappropriateness"]
    defileids, defeats = getfeatsandids_bylang(fileids, langs, features, "DE")
    itfileids, itfeats = getfeatsandids_bylang(fileids, langs, features, "IT")
    czfileids, czfeats =  getfeatsandids_bylang(fileids, langs, features, "CZ")
    testfeats = itfeats+ czfeats
    for dimension in dimensions:
        print("************for dimension: ", dimension, " ***************")
        delabels = getcatlist(defileids,dimension, "DE")
        itlabels = getcatlist(itfileids,dimension, "IT")
        czlabels = getcatlist(czfileids,dimension, "CZ")
        testlabs = itlabels + czlabels
        print("DE Train, IT Test")
        do_classification_withtest(defeats, delabels, itfeats, itlabels)
        print("DE Train, CZ Test")
        do_classification_withtest(defeats, delabels, czfeats, czlabels)


def main():
    lembedspath = sys.argv[1]
    start = datetime.now()
    print("DOING SINGLE LANG CLASSIFICATION EXPTS")
    do_single_lang(lembedspath)
    print("DOING MULTILANG CLASSIFICATION")
    do_multi_ling(lembedspath)
    print("DOING CROSS LANG CLASSIFICATION")
    do_cross_ling(lembedspath)
    end = datetime.now()
    print(end-start)
    print(start)
    print(end)

if __name__ == "__main__":
    main()


