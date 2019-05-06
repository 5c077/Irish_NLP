#Import libraries
import numpy as np
import pandas as pd
import string
import random
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.wrappers.scikit_learn import KerasClassifier
import re

#Establish Dictionaries for use in the analysis
indexes = {'a':24200,'ais':24200,'aisti':2890,'ait':24200,'ar':24200,'arsa':24200,'ban':7269,'cead':24200,'chas':24200,'chuig':24200,'dar':24200,'do':24200,'gaire':6068,'i':24200,'inar':24200,'leacht':3396,'leas':24200,'mo':24200,'na':24200,'os':24200,'re':12105,'scor':11497,'te':16563,'teann':5049,'thoir':4534}
asciis = {"a":0,"ais":0,"aisti":0,"ait":0,"ar":0,"arsa":0,"ban":0,"cead":0,"chas":0,"chuig":0,"dar":0,"do":0,"gaire":0,"i":0,"inar":0,"leacht":0,"leas":0,"mo":0,"na":0,"os":0,"re":0,"scor":0,"te":0,"teann":0,"thoir":0}
fadas = {"á":0, "áis":0, "aistí":0, "áit":0, "ár":0, "ársa":0, "bán":0, "céad":0, "chás":0, "chúig":0, "dár":0, "dó":0, "gáire":0, "í":0, "inár":0, "léacht":0, "léas":0, "mó":0, "ná":0, "ós":0, "ré":0, "scór":0, "té":0, "téann":0, "thóir":0}

#List of common Irish stop Words
ww = ["ach","ag","agus","an","aon","ar","arna","as","b'","ba","beirt","bhúr","caoga","ceathair","ceathrar","chomh","chtó","chun","cois","céad","cúig","cúigear","d'","daichead","de","deich","deichniúr","den","dhá","do","don","dtí","dá","dár","faoi","faoin","faoina","faoinár","fara","fiche","gach","gan","go","gur","haon","hocht","iad","idir","in","ina","ins","inár","is","le","leis","lena","lenár","m'","mar","mé","nach","naoi","naonúr","ná","ní","níor","nó","nócha","ocht","ochtar","os","roimh","sa","seacht","seachtar","seachtó","seasca","seisear","siad","sibh","sinn","sna","sé","sí","tar","thar","thú","triúr","trí","trína","trínár","tríocha","tú","um","ár","é","éis","ó","ón","óna","ónár"]

#Dictionaries that we'll use later to store our probabilities
d = {}
test_map = {}

# read in training and test data
#Expression to replace unwanted punctuations/digits in line
digits = str.maketrans('', '', string.digits)

# Clean lines
train = open("data/train.txt", "r")
tr_lines = train.readlines()
tr_lines = [l.rstrip("\n") for l in tr_lines]
tr_lines = [l.replace(",", "") for l in tr_lines]
tr_lines = [l.replace(".", "") for l in tr_lines]
tr_lines = [l.lower() for l in tr_lines]
tr_lines = [re.sub('\W+',' ', l).translate(digits).lower().strip() for l in tr_lines]
test = open("data/test.txt","r")
te_lines = test.readlines()
te_lines = [l.rstrip("\n") for l in te_lines]
te_lines = [l.replace(",", "") for l in te_lines]
te_lines = [l.replace(".", "") for l in te_lines]
te_lines = [l.replace("{", "漢") for l in te_lines]
te_lines = [l.replace("|", " ") for l in te_lines]
te_lines = [l.replace("}", "") for l in te_lines]
te_lines = [l.lower() for l in te_lines]

#Write function to parse corpus
def filter_low_freq(train, word, freq):
    cutoff = len(train) * freq
    word_freq = {}
    for sentence in train:
        for item in sentence.split():
            word_freq[item] = word_freq.get(item, 0) + 1
    low_words = 0
    for k, v in word_freq.items():
        if word_freq[k] < cutoff:
            low_words += 1
    #print(len(word_freq), low_words)
    num_words = len(word_freq) - low_words
    return num_words
def get_text(lines, a, f):  
    if a not in d:
        d[a] = {"X":[], "Y":[]}  
    for line in lines:
        if a in line.split(" "):
            #line = re.sub(r'\b{}\b'.format(a), '漢', line)
            d[a]['X'].append(line)
            d[a]['Y'].append(1)    
        else:#if f in line.split(" "):
            #line = re.sub(r'\b{}\b'.format(f), '漢', line)
            d[a]['X'].append(line)
            d[a]['Y'].append(0)
index = 0
print("######################_START!_#######################")
for a, f, (key, item) in zip(asciis, fadas, indexes.items()):
    sub_tr_lines = tr_lines[index:(index+indexes[key])]
    print(key, len(sub_tr_lines))
    get_text(sub_tr_lines, a, f)
    index += indexes[key]
print("######################_DONE!_########################")    

t = 0
for a in asciis:
    test_map[a] = {"index":[], "line":[]}

for a, f in zip(asciis, fadas):
    lookup = (str(" "+"漢" +a+" "))
    for num, line in enumerate(te_lines, 1):
        if lookup in line:
            line = line.replace(lookup,"")
            #line = line.replace(f_word,"")
            #line = line.replace("漢",'')
            test_map[a]['index'].append(num)
            test_map[a]['line'].append(line)
            t+=1
print(t)

mnb = MultinomialNB()
lr = LogisticRegression(max_iter = 1000)
rf=RandomForestClassifier()
svc = SVC(kernel="rbf", C=0.025, gamma = 'auto',probability=True)
bb = BaggingClassifier()
bagging = BaggingClassifier(KNeighborsClassifier(3), max_samples=0.8, max_features=0.8)
cv = StratifiedShuffleSplit(test_size=0.2, train_size=0.8, random_state=42)

def svc_param_selection(x, y, nfolds):
    Cs = [0.01, 0.1, 1, 10]
    gammas = ['auto', 0.01, 0.1, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(x, y)
    return grid_search.best_params_

clfs = [mnb]
for c in clfs: 
    name = c.__class__.__name__
    print("######################_"+name+"_#######################")
    for (d_key, d_value), (map_key, map_value) in zip(d.items(), test_map.items()):
        print("Processing "+d_key)
        X_train = d[d_key]["X"]
        Y_train = d[d_key]["Y"]
        TFIDF = TfidfVectorizer(binary = True)
                            #ngram_range=(1,2), stop_words=WW,                     
        x_train_tfidf = TFIDF.fit_transform(X_train)
        if c == svc:
            #search for optimal parameters
            best_params_dict = svc_param_selection(x_train_tfidf, Y_train, nfolds=5)
            print('best params:', best_params_dict)
            clf = SVC(kernel='rbf', C=best_params_dict['C'], gamma=best_params_dict['gamma'], probability=True)
        else:
            clf = c     
        clf.fit(x_train_tfidf, Y_train)
        x_test_tfidf = TFIDF.transform(test_map[d_key]['line'])
        pred = clf.predict_proba(x_test_tfidf)[:,1]
        test_map[map_key]["Pred"] = []
        for i in pred:
            test_map[map_key]["Pred"].append(i)  
print("######################_DONE_#######################")

with open ('./submissions/'+name+"_predictions.csv", "w") as S:
    print("Writing CSV for: "+name)
    S.write("Id,Expected\n")
    for key in test_map:
        for index, pred in zip(test_map[key]['index'], test_map[key]['Pred']):
            S.write(str(index))
            S.write(",")
            S.write(str(pred))
            S.write("\n")
S.close() 
