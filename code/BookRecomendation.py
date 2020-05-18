# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 09:41:43 2018

@author: User
"""
#Making imports
import pandas as pd
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm
from collections import Counter

#IMPORTANT NOTE : My teacher, I explained code in detail in the report.
#You can read my code directly from report.

#******************Read Data Set***********************************************

#for book dataset
book = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
book.columns = ['ISBN', 'title', 'author', 'year', 'publisher', 'UrlS', 'UrlM', 'UrlL']
unnec1 = ['title','year', 'publisher', 'author', 'UrlS', 'UrlM', 'UrlL']
book = book.drop(unnec1, axis=1)
#print("Book : ")
#print(book.shape)
#print(book.head(5))

#for user dataset
user = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
user.columns = ['userID', 'location', 'age']
#print('All User Size :',user.shape)
new_user = user[user['location'].str.contains("usa|canada")]
unnec2 = ['age','location']
new_user = new_user.drop(unnec2,axis=1)
#print('Just Usa and Canada Users Size :',new_user.shape)
#print('User : ')
#print(user.head(5))

#for rating dataset
rating = pd.read_csv('BX-Book-Ratings-Train.csv', sep=';', error_bad_lines=False, encoding="latin-1")
rating.columns = ['userID', 'ISBN', 'bookRating']
#print('Rating : ')
#print(rating.shape)
#print(user.head(5))

#for test dataset
test = pd.read_csv('BXBookRatingsTest.csv', sep=';', error_bad_lines=False, encoding="latin-1")
test.columns = ['userID', 'ISBN', 'bookRating']
#print('Test: ')
#print(test.shape)
#print(user.head(5))

#********************Combine Data Set******************************************

comBookRate = pd.merge(rating, book, on='ISBN')
new_data = comBookRate.merge(new_user, on = 'userID')
#print("new_data size: ",new_data.shape) #same pdf data number (66961) 
#print(new_data.head(5))

#******************Divede data set (train and validation)**********************

train = new_data.sample(frac=0.8,random_state=200)
validation = new_data.drop(train.index)
#print('Train :',train.shape)
#print(train.head(5))
#print('Validation :',validation.shape)
#print(validation.head(5))

#*******************Create Matrix**********************************************

train_matrix = new_data.pivot(index='userID',columns='ISBN',values='bookRating').fillna(0)
train_matrix = train_matrix.astype(np.int32)
#print('train_matrix :',train_matrix.shape)
#print(train_matrix.head(2))

#*******************Now Calculating Similarities*******************************

def sim_cosine (a,b):
    return dot(a, b)/(norm(a)*norm(b))

def euclidian_distance(a,b):
    return norm(np.array(b)-np.array(a))
    
#*********************Find similarity************************************
dict_users_sim ={}

def findSimilarity(user,train):
    dict_users_sim[user] = {}
    for v in range(len(train)):
        if train.index[v] == user: #find test user for train_matrix
            for t in range(len(train)):
                if not train.index[v] == train.index[t]:
                    sim = sim_cosine(train.values[v],train.values[t]) #calculate similarity each user in train_matrix 
                    if not math.isnan(sim):
                        if not sim == 0:
                            dict_users_sim[train.index[v]][train.index[t]] = sim #add dictionary
    #print(dict_users_sim[user])
    return dict_users_sim[user] 

#*********************Find max k-similarity************************************     
dict_max_sim ={}

def findMaxSim(dict_users_sim,user,k): 
    dict_max_sim[user] ={}
    c = Counter(dict_users_sim)
    mc = c.most_common(k)
    for i in range(len(mc)):
            dict_max_sim[user][mc[i][0]] = mc[i][1]
    #print(dict_max_sim[user])
    return dict_max_sim[user] 

#*********************Find max k-similarity weight*****************************

dict_weight_sim ={}
def max_sim_weight(dict_max_sim,user,train):
    dict_weight_sim[user] = {}
    for v in range(len(train)):
        if train.index[v] == user: #find test user for train_matrix
            for m in dict_max_sim:
                for t in range(len(train)):
                    if train.index[t] == m: #calculate weight find max sim user in train_matrix 
                        weight = euclidian_distance(train.values[v],train.values[t])
                        dict_weight_sim[train.index[v]][train.index[t]] = weight #add dictionary
    #print(dict_weight_sim[user])
    return dict_weight_sim[user]

#*********************Prediction Function**************************************

def normalPredict(train,maxSim,user,book,k):
    vote_absent = 0
    sum_perdict = 0
    for m in maxSim: #max k similarity users
        if not book in train.columns: #predict book not exist train so rating equal 0
            print('*',m, ' ',book ,' 0',)
        else:
            if train.loc[m, book] == 0: #predict book exist train but rating is 0 so not rating 
                #print(m, ' ',book ,' ',train.loc[m, book])
                vote_absent = vote_absent +1
            else:
                print(m, ' ',book ,' ',train.loc[m, book]) #predict book exist and rate exist 
                sum_perdict = sum_perdict + train.loc[m, book]
    #print(vote_absent) #Not rating book (0)
    result = sum_perdict / (len(maxSim)) #total rate / k 
    #print('total vote : ',result) 
    return result

def weightedPredict(train,maxSim,user,book,k,weightUser):
    sum_perdict = 0
    vote_absent = 0
    for m in maxSim:
        if not book in train.columns: #predict book not exist train so rating equal 0
            print('*',m, ' ',book ,' 0',)
        else:
            if train.loc[m, book] == 0: #predict book exist train but rating is 0 so not rating 
                vote_absent = vote_absent +1
            else:
                print(m, ' ',book ,' ',train.loc[m, book])
                sum_perdict = sum_perdict + (weightUser[m]*train.loc[m, book]) #Unlike normal knn, I multiplication the
                                                                            #votes into weight and then divide by total weight.
    result = sum_perdict / sum(weightUser)
    #print(result) 
    return result
    
    
#*********************Now Calculating Error************************************

def findError(predict_rate,real_rate):
    error = real_rate- predict_rate
    #print('Knn error rate :',error)
    return error
    
def findErrorWeightedKnn(predict_rate,real_rate):
    error = real_rate- predict_rate
    #print('Weighted Knn error rate :',error)
    return error
    
#***********************For the test or validation file******************************************
sum_error = 0
sum_weighted_error = 0 
for i in range(len(test[:3])): #delete validation write test for test file 
    user = test.iloc[i]['userID']
    book = test.iloc[i]['ISBN']
    rate = test.iloc[i]['bookRating']
    k = 3
    print('Predict :','userID :',user,'Book ISBN : ',book,'Book Rating :',rate)
    if user in dict_users_sim.keys():
        if user in dict_max_sim.keys():
            if not bool(dict_max_sim[user]):
                sum_error = sum_error + findError(0,rate)
                sum_weighted_error = sum_weighted_error + findError(0,rate)
            else:
                pre_rate1 = normalPredict(train_matrix, dict_max_sim[user],user,book,k)
                a = max_sim_weight(dict_max_sim[user],user,train_matrix)
                prew_rate1 = weightedPredict(train_matrix, dict_max_sim[user],user,book,k,a)
                sum_error = sum_error + findError(pre_rate1,rate)
                sum_weighted_error = sum_weighted_error + findError(prew_rate1,rate)
        else:
            a = findMaxSim(dict_users_sim[user],user,k)
            b = max_sim_weight(a,user,train_matrix)
            if not bool(a):
                sum_error = sum_error + findError(0,rate)
                sum_weighted_error = sum_weighted_error + findError(0,rate)
            else: 
                pre_rate2 = normalPredict(train_matrix,a,user,book,k) 
                prew_rate2 = weightedPredict(train_matrix,a,user,book,k,b)
                sum_error = sum_error + findError(pre_rate2,rate)
                sum_weighted_error = sum_weighted_error + findError(prew_rate2,rate)
          
    else: 
        b = findSimilarity(user,train_matrix)
        c = findMaxSim(b,user,k)
        d = max_sim_weight(c,user,train_matrix)
        if not bool(c):
                sum_error = sum_error + findError(0,rate)
                sum_weighted_error = sum_weighted_error + findError(0,rate)
        else: 
            pre_rate3 = normalPredict(train_matrix,c,user,book,k)  
            prew_rate3 = weightedPredict(train_matrix,c,user,book,k,d)
            sum_error = sum_error + findError(pre_rate3,rate)
            sum_weighted_error = sum_weighted_error + findError(prew_rate3,rate)
        
sum_n = len(test[:3])
mae_error  = sum_error /  sum_n
print('mean absolute error normal knn: ',mae_error)

mae_weighted_error = sum_weighted_error / sum_n
print('mean absolute error for weighted knn: ',mae_weighted_error)
      
#***********************************END****************************************