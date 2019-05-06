#This is a short program that finds the average probability of 
import pandas as pd
import itertools as itt
all_counts = {}
my_probs = {}
my_counts = {" a ":0, " á ":0
," ais ":0, " áis ":0
," aisti ":0, " aistí ":0
," ait ":0, " áit ":0
," ar ":0, " ár ":0
," arsa ":0, " ársa ":0
," ban ":0, " bán ":0
," cead ":0, " céad ":0
," chas ":0, " chás ":0
," chuig ":0, " chúig ":0
," dar ":0, " dár ":0
," do ":0, " dó ":0
," gaire ":0, " gáire ":0
," i ":0, " í ":0
," inar ":0, " inár ":0
," leacht ":0, " léacht ":0
," leas ":0, " léas ":0
," mo ":0, " mó ":0
," na ":0, " ná ":0
," os ":0, " ós ":0
," re ":0, " ré ":0
," scor ":0, " scór ":0
," te ":0, " té ":0
," teann ":0, " téann ":0
," thoir ":0, " thóir ":0}

asciis = {" a ":0
," ais ":0
," aisti ":0
," ait ":0
," ar ":0
," arsa ":0
," ban ":0
," cead ":0
," chas ":0
," chuig ":0
," dar ":0
," do ":0
," gaire ":0
," i ":0
," inar ":0
," leacht ":0
," leas ":0
," mo ":0
," na ":0
," os ":0
," re ":0
," scor ":0
," te ":0
," teann ":0
," thoir ":0}

fadas = {
    " á ":0
, " áis ":0
, " aistí ":0
, " áit ":0
, " ár ":0
, " ársa ":0
, " bán ":0
, " céad ":0
, " chás ":0
, " chúig ":0
, " dár ":0
, " dó ":0
, " gáire ":0
, " í ":0
, " inár ":0
, " léacht ":0
, " léas ":0
, " mó ":0
, " ná ":0
, " ós ":0
, " ré ":0
, " scór ":0
, " té ":0
, " téann ":0
, " thóir ":0
}
train = open("data/train.txt", "r") 
tr_lines = train.readlines()
tr_lines = [w.replace("\n", "") for w in tr_lines]
tr_lines = [w.replace(",", "") for w in tr_lines]
tr_lines = [w.replace(".", "") for w in tr_lines]
test = open("data/test.txt","r")
te_lines = test.readlines()
te_lines = [w.replace("\n", "") for w in te_lines]
te_lines = [w.replace(",", "") for w in te_lines]
te_lines = [w.replace(".", "") for w in te_lines]
te_lines = [w.replace("{", "*") for w in te_lines]
te_lines = [w.replace("|", " ") for w in te_lines]
te_lines = [w.replace("}", "") for w in te_lines]
te_lines = [w.replace("\*", "") for w in te_lines]
print(len(tr_lines))

a_set = tr_lines[0:24200]
for line in a_set:
    if 'a' in line:
        i=1
def counter(lines):
    for line in lines:
        words = line.split()
        for word in words:
            if word in my_counts:
                my_counts[word] += 1
            if word in all_counts:
                all_counts[word] +=1
            if word not in all_counts:
                all_counts[word] = 1
def assign_prob():
    for a, f in zip(asciis, fadas):
            #if a not in my_probs:
        my_probs["*"+a] = (all_counts[a] +1)/((all_counts[a] + all_counts[f]))
        #á_prob = 1 - a_prob

#ais_prob = all_counts["ais"]/(all_counts["ais"]+all_counts["áis"])
#áis_prob = 1 - ais_prob
def predict():
    index = 1
    with open("SL_submission.csv","w") as S:
        S.write("Id,")
        S.write("Expected")
        S.write('\n')
        for sentence in te_lines:
            words = sentence.split()
            for word in words:
                if word in my_probs:
                    a = word
                    if a.startswith('*'):
                        S.write(str(index))
                        S.write(",")
                        S.write(str(my_probs[a]))
                        S.write('\n')
                        index += 1
                      
counter(tr_lines)
assign_prob()
predict()
print(all_counts["a"])
print(all_counts["té"])
print(my_probs)
#print(te_lines[19995:20000])
#print(all_counts["léacht"])
#print(tr_lines[1:3])





#ais_prob = counts["ais"]/(counts["ais"]+counts["áis"])
#áis_prob = 1 - ais_prob#counts["áis"]/(counts["ais"]+counts["áis"]) # 1 - ais_prob

#aisti_prob = counts["aisti"]/(counts["aisti"]+counts["aistí"])


print("DONE")
