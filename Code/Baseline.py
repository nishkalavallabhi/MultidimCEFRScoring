#Baseline: A multi-class classifier for each dimension of proficiency in MERLIN (for each language)
#OverallCEFR - was used in BEA 2018.
#Note: Removed Language_check dependent features (i.e., 2 error feats per lang) as it is not compatible with openjdk
#Should add again and test with oraclejdk

import pprint
import os
import collections
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.preprocessing import Imputer #to replace NaN with mean values.
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold 
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix, mean_absolute_error
#from xgboost import XGBClassifier, XGBRegressor
from sklearn.svm import LinearSVC

from scipy.stats import spearmanr, pearsonr
#import language_check


seed=1234

'''
convert a text into its POS form. i.e., each word is replaced by its POS
'''
def makePOSsentences(conllufilepath):
    fh =  open(conllufilepath)
    everything_POS = []

    pos_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            pos_string = " ".join(pos_sentence) + "\n"
            everything_POS.append(pos_string)
            pos_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            pos_tag = line.split("\t")[3]
            pos_sentence.append(pos_tag)
    fh.close()
    return " ".join(everything_POS) #Returns a string which contains one sentence as POS tag sequence per line

def makeTextOnly(conllufilepath):
    fh =  open(conllufilepath)
    allText = []
    this_sentence = []
    sent_id = 0
    for line in fh:
        if line == "\n":
            word_string = " ".join(this_sentence) + "\n"
            allText.append(word_string)
            this_sentence = []
            sent_id = sent_id+1
        elif not line.startswith("#"):
            word = line.split("\t")[1]
            this_sentence.append(word)
    fh.close()
    return " ".join(allText) #Returns a string which contains one sentence as POS tag sequence per line

'''
convert a sentence into this form: nmod_NN_PRON, dobj_VB_NN etc. i.e., each word is replaced by a dep. trigram of that form.
So full text will look like this instead of a series of words or POS tags:
root_X_ROOT punct_PUNCT_X case_ADP_PROPN nmod_PROPN_X flat_PROPN_PROPN
 root_PRON_ROOT nsubj_NOUN_PRON case_ADP_PROPN det_DET_PROPN nmod_PROPN_NOUN
 case_ADP_NOUN det_DET_NOUN nummod_NUM_NOUN obl_NOUN_VERB root_VERB_ROOT case_ADP_NOUN det_DET_NOUN obl_NOUN_VERB appos_PROPN_NOUN flat_PROPN_PROPN case_ADP_NOUN obl_NOUN_VERB cc_CCONJ_PART conj_PART_PROPN punct_PUNCT_VERB
 advmod_ADJ_VERB case_ADP_VERB case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB nmod_NOUN_ADP case_ADP_VERB det_DET_NUM obl_NUM_VERB root_VERB_ROOT punct_PUNCT_VERB
 root_PRON_ROOT obj_NOUN_PROPN det_DET_PROPN amod_PROPN_PRON cc_CCONJ_ADV conj_ADV_PROPN cc_CCONJ_ADV punct_PUNCT_PROPN advmod_ADV_PUNCT case_ADP_ADJ advmod_ADV_PUNCT conj_ADV_PROPN amod_PROPN_PRON appos_PROPN_PROPN flat_PROPN_PROPN punct_PUNCT_PROPN
'''
def makeDepRelSentences(conllufilepath):
    fh =  open(conllufilepath)
    wanted_features = []
    deprels_sentence = []
    sent_id = 0
    head_ids_sentence = []
    pos_tags_sentence = []
    wanted_sentence_form = []
    id_dict = {} #Key: Index, Value: Word or POS depending on what dependency trigram we need. I am taking POS for now.
    id_dict['0'] = "ROOT"
    for line in fh:
        if line == "\n":
            for rel in deprels_sentence:
                wanted = rel + "_" + pos_tags_sentence[deprels_sentence.index(rel)] + "_" +id_dict[head_ids_sentence[deprels_sentence.index(rel)]]
                wanted_sentence_form.append(wanted)
                #Trigrams of the form case_ADP_PROPN, flat_PROPN_PROPN etc.
            wanted_features.append(" ".join(wanted_sentence_form) + "\n")
            deprels_sentence = []
            pos_tags_sentence = []
            head_ids_sentence = []
            wanted_sentence_form = []
            sent_id = sent_id+1
            id_dict = {}
            id_dict['0'] = "root" #LOWERCASING. Some problem with case of features in vectorizer.

        elif not line.startswith("#") and "-" not in line.split("\t")[0]:
            fields = line.split("\t")
            pos_tag = fields[3]
            deprels_sentence.append(fields[7])
            id_dict[fields[0]] = pos_tag
            pos_tags_sentence.append(pos_tag)
            head_ids_sentence.append(fields[6])
    fh.close()
    return " ".join(wanted_features)


"""
As described in Lu, 2010: http://onlinelibrary.wiley.com/doi/10.1111/j.1540-4781.2011.01232_1.x/epdf
Lexical words (N_lex: all open-class category words in UD (ADJ, ADV, INTJ, NOUN, PROPN, VERB)
All words (N)
Lex.Density = N_lex/N
Lex. Variation = Uniq_Lex/N_Lex
Type-Token Ratio = Uniq_words/N
Verb Variation = Uniq_Verb/N_verb
Noun Variation
ADJ variation
ADV variation
Modifier variation
"""
"""
def getLexFeatures(conllufilepath,lang, err):
    fh =  open(conllufilepath)
    ndw = [] #To get number of distinct words
    ndn = [] #To get number of distinct nouns - includes propn
    ndv = [] #To get number of distinct verbs
    ndadj = []
    ndadv = []
    ndint = []
    numN = 0.0 #INCL PROPN
    numV = 0.0
    numI = 0.0 #INTJ
    numADJ = 0.0
    numADV = 0.0
    numIntj = 0.0
    total = 0.0
    numSent = 0.0
    for line in fh:
        if not line == "\n" and not line.startswith("#"):
            fields = line.split("\t")
            word = fields[1]
            pos_tag = fields[3]
            if word.isalpha():
                if not word in ndw:
                    ndw.append(word)
                if pos_tag == "NOUN" or pos_tag == "PROPN":
                    numN = numN +1
                    if not word in ndn:
                        ndn.append(word)
                elif pos_tag == "ADJ":
                    numADJ = numADJ+1
                    if not word in ndadj:
                        ndadj.append(word)
                elif pos_tag == "ADV":
                    numADV = numADV+1
                    if not word in ndadv:
                        ndadv.append(word)
                elif pos_tag == "VERB":
                    numV = numV+1
                    if not word in ndv:
                        ndv.append(word)
                elif pos_tag == "INTJ":
                    numI = numI +1
                    if not word in ndint:
                        ndint.append(word)
        elif line == "\n":
            numSent = numSent +1
        total = total +1

    if err:
        try:
            error_features = getErrorFeatures(conllufilepath,lang)
        except:
            print("Ignoring file:",conllufilepath)
            error_features = [0,0]
    else:
        error_features = ['NA', 'NA']

    nlex = float(numN + numV + numADJ + numADV + numI) #Total Lexical words i.e., tokens
    dlex = float(len(ndn) + len(ndv) + len(ndadj) + len(ndadv) + len(ndint)) #Distinct Lexical words i.e., types
    #Scriptlen, Mean Sent Len, TTR, LexD, LexVar, VVar, NVar, AdjVar, AdvVar, ModVar, Total_Errors, Total Spelling errors
    result = [total, round(total/numSent,2), round(len(ndw)/total,2), round(nlex/total,2), round(dlex/nlex,2), round(len(ndv)/nlex,2), round(len(ndn)/nlex,2),
              round(len(ndadj)/nlex,2), round(len(ndadv)/nlex,2), round((len(ndadj) + len(ndadv))/nlex,2),error_features[0], error_features[1]]
    if not err: #remove last two features - they are error features which are NA for cz
       return result[:-2]
    else:
       return result
"""

"""
Num. Errors. NumSpellErrors
May be other error based features can be added later.
"""
"""
def getErrorFeatures(conllufilepath, lang):
    numerr = 0
    numspellerr = 0
    if lang not in ["CZ", "cz"]:
        try:
            checker = language_check.LanguageTool(lang)
            text = makeTextOnly(conllufilepath)
            matches = checker.check(text)
            for match in matches:
                if not match.locqualityissuetype == "whitespace":
                    numerr = numerr +1
                    if match.locqualityissuetype == "typographical" or match.locqualityissuetype == "misspelling":
                        numspellerr = numspellerr +1
        except:
            print("Ignoring this text: ", conllufilepath)
           # numerr = np.nan
           # numspellerr = np.nan
    else:
        return [0, 0]
    return [numerr, numspellerr]
"""

"""
get features that are typically used in scoring models using getErrorFeatures and getLexFeatures functions.
err - indicates whether or not error features should be extracted. Boolean. True/False
"""
"""
def getScoringFeatures(dirpath,lang,err):
    files = os.listdir(dirpath)
    fileslist = []
    featureslist = []
    for filename in files:
        if filename.endswith(".txt"):
            features_for_this_file = getLexFeatures(os.path.join(dirpath,filename),lang,err)
            fileslist.append(filename)
            featureslist.append(features_for_this_file)
    return fileslist, featureslist
"""

"""
Function to get n-gram like features for Word, POS, and Dependency representations
option takes: word, pos, dep. default is word
"""
def getLangData(dirpath, option):
    files = os.listdir(dirpath)
    fileslist = []
    bagoffeatureslist = []
    for filename in files:
        if filename.endswith(".txt"):
            if option == "pos":
            	bag_version_of_file = makePOSsentences(os.path.join(dirpath,filename)) #DO THIS TO GET POS N-GRAM FEATURES later
            elif option == "dep":
                bag_version_of_file = makeDepRelSentences(os.path.join(dirpath,filename)) #DO THIS TO GET DEP-TRIAD N-gram features later
            else:
                bag_version_of_file = makeTextOnly(os.path.join(dirpath,filename)) #DO THIS TO GET Word N-gram features later
            fileslist.append(filename)
            bagoffeatureslist.append(bag_version_of_file)
    return fileslist, bagoffeatureslist

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
def getcatlist(filenameslist, dimension, lang):

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
    #print(result)
    return result

#Get langs list from filenames - to use in megadataset classification
def getlangslist(filenameslist):
    result = []
    for name in filenameslist:
        if "_DE_" in name:
           result.append("de")
        elif "_IT_" in name:
           result.append("it")
        else:
           result.append("cz")
    return result

#Training on one language data, Stratified 10 fold CV
def train_onelang_classification(train_labels,train_data,labelascat=False, langslist=None):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10)
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)	] #Add more.GradientBoostingClassifier(),
    k_fold = StratifiedKFold(n_splits=5,random_state=seed, shuffle=True)
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            train_vector = vectorizer.fit_transform(train_data).toarray()
            print(len(train_vector[0]))
            if labelascat and len(langslist) > 1:
               train_vector = enhance_features_withcat(train_vector,language=None,langslist=langslist)
            print(len(train_vector[0]))
#            #print(vectorizer.get_feature_names()) #To see what features were selected.
#            cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
#            predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
#            print(cross_val)
#            print(sum(cross_val)/float(len(cross_val)), f1_score(train_labels,predicted,average='weighted'))
#            #print(vectorizer.get_feature_names())
#            print(confusion_matrix(train_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
#            #print(predicted)

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
            
    print("SAME LANG EVAL DONE FOR THIS LANG")
    

"""
Combine features like this: get probability distribution over categories with n-gram features. Use that distribution as a feature set concatenated with the domain features - one way to combine sparse and dense feature groups.
Just testing this approach here. 
"""
def combine_features(train_labels,train_sparse,train_dense):
    k_fold = StratifiedKFold(n_splits=5,random_state=seed, shuffle=True)
    vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 2000)
    train_vector = vectorizer.fit_transform(train_sparse).toarray()
    classifier = RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed)
  #  cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
  #  print("Old CV score with sparse features", str(sum(cross_val)/float(len(cross_val))))
  #  predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
    #print(f1_score(train_labels,predicted,average='weighted'))

    #Get probability distribution for classes.
    predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold, method="predict_proba")
    #Use those probabilities as the new featureset.
    new_features = []
    for i in range(0,len(predicted)):
       temp = list(predicted[i]) + list(train_dense[i])
       new_features.append(temp)
    #predict with new features
    new_predicted = cross_val_predict(classifier, new_features, train_labels, cv=k_fold)
    cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
   # print("new CV score", str(cross_val))
    print("Acc: " ,str(sum(cross_val)/float(len(cross_val))))
    print("F1: ", str(f1_score(train_labels,new_predicted,average='weighted')))


"""
train on one language and test on another, classification
"""
def cross_lang_testing_classification(train_labels,train_data, test_labels, test_data):
    uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,5), min_df=10) #, max_features = 2000
    vectorizers = [uni_to_tri_vectorizer]
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)] #, LinearSVC()RandomForestClassifier(), RandomForestClassifier(class_weight="balanced"), GradientBoostingClassifier()] #Side note: gradient boosting needs a dense array. Testing fails for that. Should modifiy the pipeline later to account for this.
    #Check this discussion for handling the sparseness issue: https://stackoverflow.com/questions/28384680/scikit-learns-pipeline-a-sparse-matrix-was-passed-but-dense-data-is-required
    for vectorizer in vectorizers:
        for classifier in classifiers:
            print("Printing results for: " + str(classifier) + str(vectorizer))
            text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
            text_clf.fit(train_data,train_labels)
            #print(vectorizer.get_feature_names())
            predicted = text_clf.predict(test_data)
            #print(vectorizer.get_feature_names())
            print(np.mean(predicted == test_labels,dtype=float))
            print(confusion_matrix(test_labels, predicted, labels=["A1","A2","B1","B2", "C1", "C2"]))
            print("CROSS LANG EVAL DONE. F1score: ")
            print(f1_score(test_labels,predicted,average='weighted'))
"""
Note: XGBoost classifier has some issue with retaining feature names between train and test data properly. This is resulting in error while doing cross language classification.
Strangely, I did not encounter this issue with POS trigrams. Only encountering with dependency features.
Seems to be a known issue: https://github.com/dmlc/xgboost/issues/2334
"""

#Single language, 10 fold cv for domain features - i.e., non n-gram features.
def singleLangClassificationWithoutVectorizer(train_vector,train_labels): #test_vector,test_labels):
    k_fold = StratifiedKFold(n_splits=5,random_state=seed, shuffle=True)
    classifiers = [RandomForestClassifier(class_weight="balanced",n_estimators=300,random_state=seed), LinearSVC(class_weight="balanced",random_state=seed), LogisticRegression(class_weight="balanced",random_state=seed)] #Add more later
    #classifiers = [MLPClassifier(max_iter=500)]
    #RandomForestClassifer(), GradientBoostClassifier()
    #Not useful: SVC with kernels - poly, sigmoid, rbf.
    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, train_vector, train_labels, cv=k_fold, n_jobs=1)
        predicted = cross_val_predict(classifier, train_vector, train_labels, cv=k_fold)
        print(cross_val)
        print(sum(cross_val)/float(len(cross_val)))
        print(confusion_matrix(train_labels, predicted))
        print(f1_score(train_labels,predicted,average='weighted'))

#add label features as one hot vector. de - 1 0 0, it - 0 1 0, cz - 0 0 1 as sklearn has issues with combination of cat and num features.
def enhance_features_withcat(features,language=None,langslist=None):
   addition = {'de':[1,0,0], 'it': [0,1,0], 'cz': [0,0,1]}
   if language:
        for i in range(0,len(features)):
           features[i].extend(addition[language])
        return features
   if langslist:
        features = np.ndarray.tolist(features)
        for i in range(0,len(features)):
           features[i].extend(addition[langslist[i]])
        return features



"""
Goal: combine all languages data into one big model
setting options: pos, dep, domain
labelascat = true, false (to indicate whether to add label as a categorical feature)
"""
def do_mega_multilingual_model_all_features(lang1path,lang1,lang2path,lang2,lang3path,lang3,dimension, setting,labelascat):
   print("Doing: take all data as if it belongs to one large dataset, and do classification")   
   if not setting == "domain":
      lang1files,lang1features = getLangData(lang1path,setting)
      lang2files,lang2features = getLangData(lang2path,setting)
      lang3files,lang3features = getLangData(lang3path,setting)

   """
   else: #i.e., domain features only.
      lang1files,lang1features = getScoringFeatures(lang1path,lang1,False)
      lang2files,lang2features = getScoringFeatures(lang2path,lang2,False)
      lang3files,lang3features = getScoringFeatures(lang3path,lang3,False)
   """

   lang1labels = getcatlist(lang1files, dimension, lang1)
   lang2labels = getcatlist(lang2files, dimension, lang2)
   lang3labels = getcatlist(lang3files, dimension, lang3)

   megalabels = []
   megalabels = lang1labels + lang2labels + lang3labels
   megalangs = getlangslist(lang1files) + getlangslist(lang2files) + getlangslist(lang3files)
   if labelascat and setting == "domain": 
      megadata = enhance_features_withcat(lang1features,"de") + enhance_features_withcat(lang2features,"it") + enhance_features_withcat(lang3features,"cz")
   else:
      megadata = lang1features + lang2features + lang3features
   print("Mega classification for: ", setting, " features")	
   
   print(len(megalabels), len(megadata), len(megalangs), len(megadata[0]))
  
   print("Distribution of labels: ")
   print(collections.Counter(megalabels))
   if setting == "domain":
      singleLangClassificationWithoutVectorizer(megadata,megalabels)
   else:
      train_onelang_classification(megalabels,megadata,labelascat,megalangs)

"""
this function does cross language evaluation.
takes a language data directory path, and lang code for both source and target languages. 
gets all features (no domain features for cz), and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_cross_lang_all_features(sourcelangdirpath,sourcelang,modelas, targetlangdirpath, targetlang, dimension):
   #Read source language data
   sourcelangfiles,sourcelangposngrams = getLangData(sourcelangdirpath, "pos")
   sourcelangfiles,sourcelangdepngrams = getLangData(sourcelangdirpath, "dep")
   #Read target language data
   targetlangfiles,targetlangposngrams = getLangData(targetlangdirpath, "pos")
   targetlangfiles,targetlangdepngrams = getLangData(targetlangdirpath, "dep")
   #Get label info
   sourcelanglabels = getcatlist(sourcelangfiles,dimension,sourcelang)
   targetlanglabels = getcatlist(targetlangfiles,dimension,targetlang)

   """
   if "cz" not in [sourcelang, targetlang]:
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,True)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,True)
   else: 
      sourcelangfiles,sourcelangdomain = getScoringFeatures(sourcelangdirpath,sourcelang,False)
      targetlangfiles,targetlangdomain = getScoringFeatures(targetlangdirpath,targetlang,False)
   """
      #if targetlang == "it": #Those two files where langtool throws error
      #   mean_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
      #   mean_imputer = mean_imputer.fit(targetlangdomain)
      #   imputed_df = mean_imputer.transform(targetlangdomain)
      #   targetlangdomain = imputed_df
      #   print("Modified domain feature vector for Italian")
      #TODO: it can be sourcelang too! I am ignoring that for now.
   if modelas == "class":
      print("Printing cross-corpus classification evaluation results: ")

      print("*******", "\n", "Setting - Train with: ", sourcelang, " Test with: ", targetlang, " ******", "\n")
      print("Features: pos")
      cross_lang_testing_classification(sourcelanglabels,sourcelangposngrams, targetlanglabels, targetlangposngrams)
      print("Features: dep")
      cross_lang_testing_classification(sourcelanglabels,sourcelangdepngrams, targetlanglabels, targetlangdepngrams)
      #print("Features: domain")
      #crossLangClassificationWithoutVectorizer(sourcelangdomain,sourcelanglabels,targetlangdomain,targetlanglabels)
   if modelas == "regr":
          print("Did not add for regression yet")
 
"""
this function takes a language data directory path, and lang code, 
gets all features, and prints the results with those.
lang codes: de, it, cz (lower case)
modelas: "class" for classification, "regr" for regression
"""
def do_single_lang_all_features(langdirpath,lang,dimension):
    langfiles,langwordngrams = getLangData(langdirpath, "word")
    langfiles,langposngrams = getLangData(langdirpath, "pos")
    langfiles,langdepngrams = getLangData(langdirpath, "dep")
    """
    if not lang == "cz":
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,True)
    else:
       langfiles,langdomain = getScoringFeatures(langdirpath,lang,False)
    """
    print("Extracted all features: ")
    langlabels = getcatlist(langfiles,dimension,lang)

   # if lang == "it": #Those two files where langtool throws error
   #    mean_imputer = Imputer(missing_values='NA', strategy='mean', axis=0)
   #    mean_imputer = mean_imputer.fit(langdomain)
   #    imputed_df = mean_imputer.transform(langdomain)
   #    langdomain = imputed_df
   #    print("Modified domain feature vector for Italian")

    print("Printing class statistics")
    print(collections.Counter(langlabels))

    print("With Word ngrams:", "\n", "******")
    train_onelang_classification(langlabels,langwordngrams)
    print("With POS ngrams: ", "\n", "******")
    train_onelang_classification(langlabels,langposngrams)
    print("Dep ngrams: ", "\n", "******")
    train_onelang_classification(langlabels,langdepngrams)
    #print("Domain features: ", "\n", "******")
    #singleLangClassificationWithoutVectorizer(langdomain,langlabels)

    """
    print("Combined feature rep: wordngrams + domain")
    combine_features(langlabels,langwordngrams,langdomain)
    print("Combined feature rep: posngrams + domain")
    combine_features(langlabels,langposngrams,langdomain)
    print("Combined feature rep: depngrams + domain")
    combine_features(langlabels,langdepngrams,langdomain)
    """

def main():

    itdirpath = "../Datasets/IT-Parsed"
    dedirpath = "../Datasets/DE-Parsed"
    czdirpath = "../Datasets/CZ-Parsed"

    dimensions = ["OverallCEFRrating", "Grammaticalaccuracy", "Orthography","Vocabularyrange","Vocabularycontrol",
		"CoherenceCohesion", "Sociolinguisticappropriateness"]

    #Single language case:
    langpaths = {'DE':dedirpath, 'IT':itdirpath, 'CZ':czdirpath}
    for lang in langpaths.keys():
        print("Doing monolingual classification for ", lang)
        for dimension in dimensions:
            print("************for dimension: ", dimension, " ***************")
            do_single_lang_all_features(langpaths[lang],lang, dimension)

    for dimension in dimensions:
        #Cross lingual classification
        print("Cross lingual classification, DE Train, IT Test for dimension", dimension)
        do_cross_lang_all_features(dedirpath,"DE","class", itdirpath, "IT", dimension) #Run this for all combos.
        print("Cross lingual classification, DE Train, CZ Test for dimension", dimension)
        do_cross_lang_all_features(dedirpath,"DE","class", czdirpath, "CZ", dimension) #Run this for all combos.

    for dimension in dimensions:
            print("Multi lingual classification for dimension", dimension)
            #Multilingual classification
            do_mega_multilingual_model_all_features(dedirpath,"DE",itdirpath,"IT",czdirpath,"CZ", dimension, "pos", False)
            do_mega_multilingual_model_all_features(dedirpath,"DE",itdirpath,"IT",czdirpath,"CZ", dimension, "dep", False)


if __name__ == "__main__":
    main()

"""
TODO: Refactoring, reducing redundancy

"""
