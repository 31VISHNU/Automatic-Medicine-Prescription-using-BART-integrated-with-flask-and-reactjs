from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle Cross-Origin Resource Sharing
import pandas as pd
# Import other necessary libraries and functions
from flask_cors import cross_origin
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

import transformers
from datasets import load_dataset, load_metric, load_from_disk
import numpy as np
import nltk
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier,_tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import nltk
from matplotlib import pyplot as plt
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import tree
nltk.download('stopwords')
import csv
import warnings
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

file_path = 'C:/Users/vishn/finalyear/project/project/src/Datasets/dataset1.csv'
df = pd.read_csv(file_path)
G = nx.Graph()

for _, row in df.iterrows():
    symptoms = row.dropna().tolist()

    for i in range(len(symptoms) - 1):
        for j in range(i + 1, len(symptoms)):
            G.add_edge(symptoms[i], symptoms[j])

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=8, font_color="black", font_weight="bold", edge_color="gray", linewidths=0.5, alpha=0.7)
training = pd.read_csv('C:/Users/vishn/finalyear/project/project/src/Datasets/Training1.csv')
diseases_in_column = set(training['prognosis'])
testing= pd.read_csv('C:/Users/vishn/finalyear/project/project/src/Datasets/Testing1.csv')
cols= training.columns
cols= cols[:-1]
x = training[cols]
y = training['prognosis']
y1= y
reduced_data = training.groupby(training['prognosis']).max()
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx    = testing[cols]
testy    = testing['prognosis']
testy    = le.transform(testy)
clf1  = DecisionTreeClassifier()
clf = clf1.fit(x_train,y_train)
scores = cross_val_score(clf, x_test, y_test, cv=3)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols
severityDictionary=dict()
description_list = dict()
precautionDictionary=dict()
symptoms_dict = {}
for index, symptom in enumerate(x):
       symptoms_dict[symptom] = index
def Description():
    global description_list
    with open('C:/Users/vishn/finalyear/project/project/src/Datasets/symptom_Description1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _description={row[0]:row[1]}
            description_list.update(_description)
def Severity():
    global severityDictionary
    with open('C:/Users/vishn/finalyear/project/project/src/Datasets/Symptom_severity1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        try:
            for row in csv_reader:
                _diction={row[0]:int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass
def Precaution():
    global precautionDictionary
    with open('C:/Users/vishn/finalyear/project/project/src/Datasets/symptom_precaution1.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            _prec={row[0]:[row[1],row[2],row[3],row[4]]}
            precautionDictionary.update(_prec)
def find_connected_symptoms(graph, input_symptom, num_connected_symptoms=8):
    neighbors = list(graph.neighbors(input_symptom))
    return neighbors[:num_connected_symptoms]
def check_pattern(dis_list,inp):
    pred_list=[]
    inp=inp.replace(' ','_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list=[item for item in dis_list if regexp.search(item)]
    if(len(pred_list)>0):
        return 1,pred_list
    else:
        return 0,[]
def sec_predict(symptoms_exp):
    df = pd.read_csv('C:/Users/vishn/finalyear/project/project/src/Datasets/Training1.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))
    for item in symptoms_exp:
      input_vector[[symptoms_dict[item]]] = 1
    return rf_clf.predict([input_vector])
def print_disease(node):
    node = node
    val  = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x:x.strip(),list(disease)))
def chat(tree, feature_names,example_sent,num_days):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    chk_dis=",".join(feature_names).split(",")
    symptoms_present = []
    while True:
        print("\nEnter the symptom you are experiencing  \t\t",end="->")
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(example_sent)
        filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
        filtered_sentence = []
        for w in word_tokens:
          if w not in stop_words:
            filtered_sentence.append(w)
        with open('C:/Users/vishn/finalyear/project/project/src/Datasets/keywords1.txt', 'r') as file:
          text_file_contents = file.read()
        words_in_text_file = [word for word in filtered_sentence if word in text_file_contents]
        for word in words_in_text_file:
          print(word)
        input_symptom=words_in_text_file[0]
        conf,cnf_dis=check_pattern(chk_dis,input_symptom)
        if conf==1:
            print("searches related to input: ")
            for num,it in enumerate(cnf_dis):
                print(num,")",it)
            if num!=0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp=0
            disease_input=cnf_dis[conf_inp]
            break
        else:
            print("Enter valid symptom.")
    while True:
        try:
            #num_days=int(input("Okay. From how many days ? : "))
            break
        except:
            print("Enter valid input.")
    def recurse(node, depth):
            print("Are you experiencing any ")
            symptoms_given = find_connected_symptoms(G, disease_input)
            symptoms_given = [symptom for symptom in symptoms_given if symptom not in diseases_in_column]
            symptoms_exp=[]
            for syms in list(symptoms_given):
                inp=""
                print(syms,"? : ",end='')
                while True:
                    inp=input("")
                    if(inp=="yes" or inp=="no"):
                        break
                    else:
                        print("provide yes/no : ",end="")
                if(inp=="yes"):
                    symptoms_exp.append(syms)
            second_prediction=sec_predict(symptoms_exp)
            print("---------------------Report Generated from ATD-----------------------")
            if(second_prediction[0]):
                input_text=second_prediction[0]
                return description_list[second_prediction[0]]
                #print(description_list[second_prediction[0]])
            precution_list=precautionDictionary[second_prediction[0]]
            print("Take following measures : ")
            for  i,j in enumerate(precution_list):
                print(i+1,")",j)
    a=recurse(0, 1)
    return a
Severity()
Description()
Precaution()

# Define an endpoint for processing the chat
@app.route('/process_chat', methods=['POST'])
def process_chat():
    data = request.get_json()
    name = data['name']
    age = data['age']
    disease_input=data['symptoms']
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(disease_input)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
    chk_dis=",".join(cols).split(",")
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    with open('C:/Users/vishn/finalyear/project/project/src/Datasets/keywords1.txt', 'r') as file:
          text_file_contents = file.read()
    words_in_text_file = [word for word in filtered_sentence if word in text_file_contents]
    for word in words_in_text_file:
          print(word)
    input_symptom=words_in_text_file[0]
    conf,cnf_dis=check_pattern(chk_dis,input_symptom)
    if conf==1:
        print("searches related to input: ")
        sys=[]
        for num,it in enumerate(cnf_dis):
            sys.append(it)
        result={'result': sys}
        """if num!=0:
            print(f"Select the one you meant (0 - {num}):  ", end="")
            conf_inp = int(input(""))
        else:
            conf_inp=0"""
        #input_symptom=cnf_dis[conf_inp]
    return jsonify(result)
@app.route('/sym', methods=['POST'])
@cross_origin()
def sym():
    dat=request.get_json()
    selected_index = dat['selected_index']
    print(selected_index)
    disease_input=dat['symptoms']
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(disease_input)
    filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
    filtered_sentence = []
    chk_dis=",".join(cols).split(",")
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    with open('C:/Users/vishn/finalyear/project/project/src/Datasets/keywords1.txt', 'r') as file:
          text_file_contents = file.read()
    words_in_text_file = [word for word in filtered_sentence if word in text_file_contents]
    for word in words_in_text_file:
          print(word)
    input_symptom=words_in_text_file[0]
    conf,cnf_dis=check_pattern(chk_dis,input_symptom)
    input_symptom=cnf_dis[selected_index]
    try:
        symptoms_given = find_connected_symptoms(G, input_symptom)
        symptoms_given = [symptom for symptom in symptoms_given if symptom not in diseases_in_column]
        sss=[]
        for i in symptoms_given:
            sss.append(i)
    except Exception as e:
        result = {'error': str(e)}

    return jsonify({'symp':sss})
@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    da=request.get_json()
    symptoms_exp=da['item']
    num_days=da['day']
    second_prediction=sec_predict(symptoms_exp)
    a=second_prediction[0]
    #print("---------------------Report Generated from ATD-----------------------")
    return jsonify({'disease':a})
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
@app.route('/medicine', methods=['POST'])
@cross_origin()
def medicine():
    dd=request.get_json()
    input_text=dd['dis']
    model_name = "C:/Users/vishn/Downloads/fine_tuned_bart_best_10epoch-20240208T140224Z-001/fine_tuned_bart_best_10epoch"
    model = BartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = BartTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(input_text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        outputs = model.generate(input_ids)
    predicted_medicine = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return jsonify({'med':predicted_medicine})

if __name__ == '__main__':
    app.run(port=5000)  
