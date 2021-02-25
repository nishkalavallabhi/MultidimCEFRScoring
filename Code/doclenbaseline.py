#Adding a document length baseline for final version.
import os
import collections
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold,cross_validate
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
from sklearn.svm import LinearSVC

seed = 1234

def get_metadata_file(lang):
    #Lang has to be DE, IT or CZ
    metadata = {} #Key is the file name (without .txt), Values: List of all CEFR dimension ratings
    if lang.upper() not in ["DE","IT","CZ"]:
        print("Wrong language entered. Allowed ones are: DE, IT, CZ")
        exit(1)
    else:
        filename = "../Datasets/"+lang.upper()+"Metadata.txt"
        fh = open(filename)
        for line in fh:
            temp = line.strip().split(",")
            metadata[temp[0]] = temp[1:]
        fh.close()
    return metadata

#Get categories from filenames  -Classification
def getcats(filenameslist, dimension, lang):

    dimensions = {"OverallCEFRrating":0, "Grammaticalaccuracy":1, "Orthography":2, "Vocabularyrange":3, "Vocabularycontrol":4,
		"CoherenceCohesion":5, "Sociolinguisticappropriateness":6}
    if dimension in dimensions and lang.upper() in ["DE", "IT", "CZ"]:
        langmetadata = get_metadata_file(lang)
        result = []
        for name in filenameslist:
            if lang.upper() == "CZ": #file naming patterns in MERLIN are different for DE and IT compared to CZ
                actual_name = name.split("_")[0]
            else:
                actual_name = "_".join(name.split("_")[:2])
            dim_loc = dimensions[dimension]
            dim_cat = langmetadata[actual_name][dim_loc].split(":")[1]
            if dim_cat == "0": #Should we take this as A1 or NA
                dim_cat="A1"
            result.append(dim_cat)
    else:
        print("Wrong dimension or language entered. Dimension has to be one of: ", ", ".join(dimensions.keys()),
                            " and language has to be one of DE, IT, CZ" )
        exit(1)
#    print(result)
    return result

def getdoclen(conllufilepath):
    fh =  open(conllufilepath, encoding="utf-8")
    allText = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            sent_id = sent_id+1
        elif not line.startswith("#") and line.split("\t")[3] != "PUNCT":
            word = line.split("\t")[1]
            allText.append(word)
    fh.close()
    return len(allText)

def getfeatures(dirpath, dimension,lang):
    files = os.listdir(dirpath)
    cats = getcats(files,dimension,lang)
    doclenfeaturelist = []
    for filename in files:
        if filename.endswith(".txt"):
            doclenfeaturelist.append([getdoclen(os.path.join(dirpath,filename))])
    print(len(doclenfeaturelist), len(cats))
    return doclenfeaturelist,cats

def singleLangClassificationWithoutVectorizer(train_vector,train_labels): #test_vector,test_labels):
    k_fold = StratifiedKFold(5,random_state=seed, shuffle=True)
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)] #Add more later
    #classifiers = [MLPClassifier(max_iter=500)]
    #RandomForestClassifer(), GradientBoostClassifier()
    #Not useful: SVC with kernels - poly, sigmoid, rbf.
    for classifier in classifiers:
#        print(classifier)
#        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
#        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
#        print(cross_val)
#        print(sum(cross_val)/float(len(cross_val)))
#        print(confusion_matrix(train_labels, predicted))
#        print(f1_score(train_labels,predicted,average='weighted'))
        print(classifier)        
        weighted_f1_scores = []
        for i, indices in enumerate(k_fold.split(train_vector, train_labels)):
            train_index, test_index = indices
            feats_train = [train_vector[x] for x in train_index]
            labels_train = [train_labels[x] for x in train_index]

            feats_test = [train_vector[x] for x in test_index]
            labels_test = [train_labels[x] for x in test_index]

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
        

def crossLangClassificationWithoutVectorizer(train_vector, train_labels, test_vector, test_labels):
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)]
    for classifier in classifiers:
        classifier.fit(train_vector,train_labels)
        predicted = classifier.predict(test_vector)
        print(np.mean(predicted == test_labels,dtype=float))
        print(confusion_matrix(test_labels,predicted))
        print(f1_score(test_labels,predicted,average='weighted'))


def main():
    itdirpath = "../Datasets/IT-Parsed"
    dedirpath = "../Datasets//DE-Parsed"
    czdirpath = "../Datasets/CZ-Parsed"

    dimensions = ["OverallCEFRrating", "Grammaticalaccuracy", "Orthography","Vocabularyrange","Vocabularycontrol",
		"CoherenceCohesion", "Sociolinguisticappropriateness"]

    for dimension in dimensions:
        print("************for dimension: ", dimension, " ***************")

        print("************DE baseline:****************")
        defeats,delabels = getfeatures(dedirpath, dimension, "DE")
        singleLangClassificationWithoutVectorizer(defeats,delabels)
        print("************IT baseline:****************")
        itfeats,itlabels = getfeatures(itdirpath, dimension, "IT")
        singleLangClassificationWithoutVectorizer(itfeats,itlabels)
        print("************CZ baseline:****************")
        czfeats,czlabels = getfeatures(czdirpath, dimension, "CZ")
        singleLangClassificationWithoutVectorizer(czfeats,czlabels)

        print("*** Train with DE, test with IT baseline******")
        crossLangClassificationWithoutVectorizer(defeats,delabels, itfeats,itlabels)

        print("*** Train with DE, test with CZ baseline ******")
        crossLangClassificationWithoutVectorizer(defeats,delabels, czfeats,czlabels)

        bigfeats = []
        bigcats = []
        bigfeats.extend(defeats)
        bigfeats.extend(itfeats)
        bigfeats.extend(czfeats)
        bigcats.extend(delabels)
        bigcats.extend(itlabels)
        bigcats.extend(czlabels)
        print("****Multilingual classification baseline*************")
        singleLangClassificationWithoutVectorizer(bigfeats,bigcats)

    
    
main()

