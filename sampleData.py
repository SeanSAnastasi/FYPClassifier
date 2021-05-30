from datetime import time
import numpy as np
from numpy.core.fromnumeric import amax
from numpy.lib.nanfunctions import nanmin
import pandas as pd
import math
import os.path
from os import path
from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.linear_model import LassoCV

import warnings
warnings.filterwarnings('ignore')  
metric = ""

tlx = []
seq = []
smeq = []
mean = []
delta_mean = []
std_dev = []
time_taken = []
median = []
min = []
max = []
min_rest = []
max_rest = []

tasks = [[],[],[],[],[]]
task_tlx = [[],[],[],[],[]]
task_seq = [[],[],[],[],[]]
task_smeq = [[],[],[],[],[]]
task_mean = [[],[],[],[],[]]
task_delta_mean = [[],[],[],[],[]]
task_std_dev = [[],[],[],[],[]]
task_time_taken = [[],[],[],[],[]]
task_median = [[],[],[],[],[]]
task_min = [[],[],[],[],[]]
task_max = [[],[],[],[],[]]


# For Classification
data_array = []
quest_array = []

for i in range(0,20):
    # print(i)
    if (i != 9):
        file = "./watch/S"+str(i)+"_quest.csv"
        data = "./watch/S"+str(i)+".csv"

        if path.exists(file):


            file_table = pd.read_csv(file)
            data_table = pd.read_csv(data)
            
            # print(file_table)
            

            tables = []

            for j in range(5):
                row_start = str(file_table.iloc[1,j+1])
                
                row_start = row_start.split(".")
                row_end = str(file_table.iloc[2,j+1])
                
                row_end = row_end.split(".")
                if(len(row_start) <2):
                    row_start.append(0)
                if(len(row_end) <2):
                    row_end.append(0)
                start = (int(row_start[0])*60) + int(row_start[1])
                end = (int(row_end[0])*60) + int(row_end[1])
                # print("Participant: "+str(i))
                # print("Task: "+str(j))
                # print(start)
                # print(end)
                # print("-------------------------")
                

                data_subset = data_table[start:end+1]
                rest = data_table[start-61: start-1]
                # print(rest)
                
                
                
                # data_subset = data_table[start:start+1000]
                data_list = []
                rest_list = []
                for row in data_subset.itertuples():
                    data_list.append(int(row._1))
                for row in rest.itertuples():
                    rest_list.append(int(row._1))
                # print(rest_list)
                mean_rest = np.mean(rest_list)
                delta = [x-mean_rest for x in data_list]

                if len(data_list) >0 :
                    data_array.append(data_list)

                    time_taken.append(end-start)
                    mean.append(np.mean(data_list))
                    std_dev.append(np.std(data_list))
                    min.append(np.min(data_list))
                    max.append(np.max(data_list))
                    min_rest.append(np.min(rest_list))
                    max_rest.append(np.max(rest_list))
                    delta_mean.append(np.mean(delta))
                    median.append(np.median(data_list))
                    # Per task data
                    tasks[j].append(data_list)
                    task_time_taken[j].append(end-start)
                    task_mean[j].append(np.mean(data_list))
                    task_std_dev[j].append(np.std(data_list))
                    task_min[j].append(np.min(data_list))
                    task_max[j].append(np.max(data_list))
                    task_delta_mean[j].append(np.mean(delta))
                    task_median[j].append(np.median(data_list))

                    tables.append(True)
                else:
                    tables.append(False)


                    tasks[j].append(data_list)
                    task_time_taken[j].append(end-start)
                    task_mean[j].append(0)
                    task_std_dev[j].append(0)
                    task_min[j].append(0)
                    task_max[j].append(0)
                    task_delta_mean[j].append(0)
                    task_median[j].append(0)


            
            k = 0

            for row in file_table.iloc[10:15].itertuples():
                task_seq[k].append(int(row._2))
                if tables[k] == True:
                    seq.append(int(row._2)) 
                    
                    
                k = k+1

            k = 0
            for row in file_table.iloc[4:9].itertuples():
                task_tlx[k].append(int(row._2))
                if tables[k] == True:
                    tlx.append(int(row._2)) 
                    
                k = k+1

            k = 0
            for row in file_table.iloc[16:21].itertuples():
                task_smeq[k].append(int(row._2))
                if tables[k] == True:
                    smeq.append(int(row._2)) 
                    
                k = k+1


            k = 0
            # for row in file_table.iloc[4:9].itertuples():
            #     metric = "TLX"
            #     if tables[k] == True:
            #         if int(row._2) > 27:
            #             quest_array.append(1)
            #         else:
            #             quest_array.append(0)
            #     k = k+1
            
            # for row in file_table.iloc[10:15].itertuples():
            #     metric = "SEQ"
            #     if tables[i] == True:
            #         if int(row._2) > 3:
            #             quest_array.append(1)
            #         else:
            #             quest_array.append(0)
            #     i = i+1

            # for row in file_table.iloc[16:21].itertuples():
            #     metric = "SMEQ"
            #     if tables[i] == True:
            #         if int(row._2) > 30:
            #             quest_array.append(1)
            #         else:
            #             quest_array.append(0)
            #     i = i+1
            # i = 0
            # tables = []

# per task here
tables = []
for i in range(5):
    table = np.array([tasks[i],task_tlx[i], task_seq[i], task_smeq[i], task_mean[i], task_delta_mean[i], task_std_dev[i],task_time_taken[i], task_median[i], task_min[i], task_max[i]])
    tables.append(table)
    # print(len(table))
    # print("------------------------------------------------------")

i = 0
for table in tables:
    # np.savetxt('data_analysis/'+str(i)+".csv", table,  delimiter=',')
    # print(table)
    df = pd.DataFrame(table).T
    # df.to_excel(excel_writer='data_analysis/'+str(i)+'test.xls')
    i += 1


# STATS HERE
data_table_classifier = []
data_table_stats = []
print(np.median(tlx))
print(np.median(seq))
print(np.median(smeq))
for i in range(len(tlx)):

    for data in data_array[i]:

        # temp1 = [time_taken[i]]
        temp1 = [ mean[i],median[i], delta_mean[i], std_dev[i], time_taken[i], data,min[i], max[i], min_rest[i], max_rest[i]]
        temp2 = [ tlx[i], seq[i], smeq[i], mean[i], delta_mean[i], std_dev[i], time_taken[i], data, min[i], max[i], min_rest[i], max_rest[i]]
        data_table_classifier.append(temp1)
        data_table_stats.append(temp2)
        metric = "SEQ"
        
        if(seq[i] >= np.median(seq)):
            quest_array.append(1)
        else:
            quest_array.append(0)


# print(min_rest)
cor, pval = stats.spearmanr(data_table_stats)

# print ('stats.spearmanr - cor:\n', cor)
# print ('stats.spearmanr - pval\n', pval)

data_table_stats = np.array(data_table_stats)
df = pd.DataFrame(data_table_stats, columns=['TLX','SEQ','SMEQ','MEAN','DELTA MEAN', 'STANDARD DEVIATION', 'TIME TAKEN', 'HEART RATE', 'MIN', 'MAX', 'MIN REST', 'MAX REST'])
correlation = df.corr(method='spearman')

# sns.heatmap(correlation, annot= True)
# sns.heatmap(cor, annot= True)
# plt.title("Correlation table")
# plt.show()



# CLASSIFICATION HERE

pad = 0



for data in data_array:
    if len(data) > pad:
        pad = len(data)

for data in data_array:
    num = pad-len(data)
    # print(len(data))
    # data_mean = np.mean(data)
    # data_min = min(data)
    # data_max = max(data)
    # data_std = np.std(data)
    for i in range(num):
        data.append(0)
    # data.extend([data_mean, data_max, data_min, data_std])
    # feature_vector = [data_mean, data_max, data_min, data_std, data]
    # feature_vector = preprocessing.normalize(data)
    # print(feature_vector)
    # feature_vectors.append(feature_vector)
    

# data_array_split = []
# quest_array_split = []
# for data in data_array:
#     new_data = np.array_split(data,2)
#     data_array_split.append(new_data[0])
#     data_array_split.append(new_data[1])
# for quest in quest_array:
#     for i in range(2):
#         quest_array_split.append(quest)

feature_vectors = preprocessing.normalize(data_array)
total_accuracy = 0
total_precision = 0
total_recall = 0
total_f1 = 0
runs = 1
y_test_total = []
y_pred_total = []


for i in range(runs):
# data_table_classifier
    x_train, x_test, y_train, y_test = train_test_split(data_table_classifier, quest_array, test_size=0.2)

    model = GaussianNB()

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred) *100
    precision = precision_score(y_test, y_pred) *100
    recall = recall_score(y_test, y_pred) *100
    f1 = f1_score(y_test,y_pred) *100
    # print(y_test)
    # print(y_pred)

    # print(classification_report(y_test, y_pred))

    lasso = LassoCV().fit(x_train, y_train)
    importance = np.abs(lasso.coef_)
    
    feature_names = np.array(['MEAN','Median','DELTA MEAN', 'STANDARD DEVIATION', 'TIME TAKEN', 'HEART RATE', 'MIN', 'MAX', 'MIN REST', 'MAX REST'])
    # plt.bar(height=importance, x=feature_names)
    # plt.title("Feature importance using SEQ")
    # plt.show()


    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_f1 += f1
print(metric)
print("Accuracy: "+str(total_accuracy/runs))
print("Precision: "+str(total_precision/runs))
print("Recall: "+str(total_recall/runs))
print("F1: "+str(total_f1/runs))