import numpy as np
import pandas as pd
import math
import os.path
from os import path
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data_array = []
quest_array = []

for i in range(0,4):
    file = "./watch/S"+str(i)+"_quest.csv"
    data = "./watch/S"+str(i)+".csv"

    if path.exists(file):


        file_table = pd.read_csv(file)
        data_table = pd.read_csv(data)
        
        # print(file_table)
        

        for i in range(5):
            row_start = str(file_table.iloc[1,i+1])
            
            row_start = row_start.split(".")
            row_end = str(file_table.iloc[2,i+1])
            
            row_end = row_end.split(".")
            if(len(row_start) <2):
                row_start.append(0)
            if(len(row_end) <2):
                row_end.append(0)
            start = (int(row_start[0])*60) + int(row_start[1])
            end = (int(row_end[0])*60) + int(row_end[1])

            data_subset = data_table[start:end+1]
            # data_subset = data_table[start:start+1000]
            data_list = []

            for row in data_subset.itertuples():
                data_list.append(int(row._1))
            
            data_array.append(data_list)

        # TLX [4:9]
        # SEQ [10:15]
        #SMEQ [16:21]

        # for row in file_table.iloc[4:9].itertuples():
            
        #     if int(row._2) > 49:
        #         quest_array.append(1)
        #     else:
        #         quest_array.append(0)

        # for row in file_table.iloc[10:15].itertuples():
            
        #     if int(row._2) > 4:
        #         quest_array.append(1)
        #     else:
        #         quest_array.append(0)

        # for row in file_table.iloc[16:21].itertuples():
            
        #     if int(row._2) > 74:
        #         quest_array.append(1)
        #     else:
        #         quest_array.append(0)
    

pad = 0

for data in data_array:
    if len(data) > pad:
        pad = len(data)

for data in data_array:
    num = pad-len(data)
    for i in range(num):
        data.append(0)

x_train, x_test, y_train, y_test = train_test_split(data_array, quest_array, test_size=0.2)

model = MultinomialNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred) *100

print(accuracy)