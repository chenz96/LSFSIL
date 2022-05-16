import numpy as np
import pickle
import scipy
from scipy import linalg
from sklearn.svm import SVR, SVC
from sklearn import metrics
from sklearn.model_selection import KFold,RepeatedKFold

import matplotlib.pyplot as plt
import scipy
import os
def evaluation_s(score_labels, score_pred):
    cofscore = 0
    index_selected = np.where(score_labels >= 0)
    new_x = score_labels[index_selected]
    new_y = score_pred[index_selected]
    var_x = np.sqrt(new_x.var() + 0.00000001)
    var_y = np.sqrt(new_y.var() + 0.00000001)
    mean_x = new_x.mean()
    mean_y = new_y.mean()
    t_x = new_x - mean_x
    t_y = new_y - mean_y
    t_xy = t_x * t_y
    t_xy = t_xy.mean()
    cofscore = t_xy / (var_x * var_y)
    error = new_x - new_y
    rmse = (error**2).mean()**0.5
    return cofscore, rmse

def cal_G(input):
    normv = np.linalg.norm(input,axis=1)
    #print(normv)
    normv = 1/(2*normv + 0.000000001)
    g = np.diag(normv)
    return g

def save_obj(obj, name ):
    with open('./data/'+name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('./data/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)


def select_com_score(X, Y_cls, Y_score_n):
    New_X = list()
    New_Y_cls = list()
    New_Y_score = list()
    for idx in range(X.shape[0]):
        if (Y_score_n[idx, :] < 0).any() == False:
            New_X.append(X[idx, :])
            New_Y_score.append(Y_score_n[idx, :])
            New_Y_cls.append(Y_cls[idx, :])
    X = np.array(New_X)
    Y_score_n = np.array(New_Y_score)
    Y_cls = np.array(New_Y_cls)
    return X, Y_cls, Y_score_n

def s_fi(fi, x):
    re1 = (x - fi)
    re1 = re1* (re1>0).astype(float)
    re2 = (-1*x - fi)
    re2 = re2* (re2>0).astype(float)
    re = re1 - re2
    return re

# load data
X = np.array( load_obj('data_UCSF'))
Y_score_n  = np.array  (load_obj('scores_UCSF'))


for idx in range(X.shape[1]):
    X[:,idx] = (X[:,idx] - np.min(X[:,idx])) / (np.max(X[:,idx]) - np.min(X[:,idx]))
for idx in range(Y_score_n.shape[1]):
    Y_score_n[:,idx] = Y_score_n[:,idx] / np.max(Y_score_n[:,idx])

Y = Y_score_n

n_splits=5
n_repeats=10

kf = RepeatedKFold(n_splits = n_splits, n_repeats=n_repeats)

def onecv(X, Y, n_splits,    alpha =100,    beta = 10,    gama = 0.1,R_low = 4,N_mid = 12):

    rho = 1.1
    niumax = 1e7
    fi = 1e-8

    select_num_list = [2,4,6,8,10,12,14,16,18,20]

    N_scores = Y.shape[1]
    all_reslts_rmse = np.zeros((n_splits*n_repeats,len(select_num_list),N_scores))
    all_reslts_cc = np.zeros((n_splits*n_repeats,len(select_num_list),N_scores))

    for i_fold, (train_index, test_index) in enumerate(kf.split(X)):

        train_X , train_Y = X[train_index], Y[train_index,:]
        test_X , test_Y = X[test_index], Y[test_index,:]
        N_classes = train_Y.shape [1]   
        N_features = train_X.shape[1]
        N_samples = train_Y.shape[0]

        Y_ =  train_Y
        E = np.zeros([N_samples, N_classes])
        P = np.zeros([N_features, N_mid])
        K = np.zeros([N_samples, N_classes])
        R = np.zeros([N_samples, R_low])
        H = np.zeros([R_low, N_classes])
        Y1 = np.zeros([N_samples, N_classes])
        Y2 = np.zeros([N_samples, N_classes])
        Y3 = np.zeros([N_samples, N_classes])
        niu = 1e-7

        S = scipy.spatial.distance.cdist(train_X,train_X)
        S=np.exp(-1*S*S) - np.eye(N_samples)
        D_ = np.diag(np.sum(S, axis=1))
        S=D_-S


        for ite in range(1000):

            #update K
            M1 = Y_ - np.matmul(train_X,P) - Y2/niu
            M2 = np.matmul(R,H) - Y3/niu
            K = (M1+M2)/2

            # update Y_
            M1 = train_Y - E + Y1/niu
            M2 = K + np.matmul(train_X, P)+Y2/niu
            Mid = np.linalg.inv(2*niu*np.eye(N_samples) + 2*gama*S)
            Y_ = np.matmul(Mid, niu*M1+niu*M2)

            #update E
            M1 = train_Y - Y_ + Y1/niu
            be = beta / niu
            E = s_fi(be, M1)

            #update P
            Q2 = Y_ - K - Y2 / niu
            G = cal_G(P)

            
            M_inv = np.linalg.inv(2*alpha*G + niu * np.matmul(train_X.transpose(), train_X))
            P = niu * np.matmul( np.matmul(M_inv , train_X.transpose()), Q2)

            #update R
            EYE = np.eye(R_low)
            M1 = K + Y3/niu
            M2 = np.linalg.inv( EYE + niu * np.matmul(H, H.transpose()) )
            R = niu * np.matmul( np.matmul(M1, H.transpose() ), M2)

            #update H
            M2 = np.linalg.inv(EYE  + niu * np.matmul( R.transpose(), R))
            H = niu * np.matmul(np.matmul(M2, R.transpose()), M1)


            #update Y1 - Y3
            Y1 = Y1 + niu*(train_Y - Y_ -  E)
            Y2 = Y2 + niu*(K - Y_ + np.matmul(train_X, P) )
            Y3 = Y3 + niu*(K - np.matmul(R, H))


            stop1 = np.linalg.norm(train_Y - Y_ - E,ord=np.inf) 
            stop2 = np.linalg.norm(K - Y_ + np.matmul(train_X, P),ord=np.inf)
            stop3 = np.linalg.norm(K - np.matmul(R, H),ord=np.inf)  

            if stop1 < fi and stop2<fi and stop3<fi :
               break
            niu = min( niu * rho ,niumax)


        normv = np.linalg.norm(P,axis=1)
        sort_list = np.sort(normv)
        train_X ,_, train_Y = select_com_score(train_X,train_Y,train_Y)
        for i_prob, select_prop in enumerate(select_num_list):
            cutv = sort_list[N_features-select_prop]
            selected_index = np.where(normv>=cutv )[0]
            train_X_svm = train_X[:,selected_index]
            test_X_svm = test_X[:,selected_index]
            ans  = list()
            ans_r = list()

            for i_s in range(N_classes):
                linear_svr = SVR(kernel="linear")
                linear_svr.fit(train_X_svm, train_Y[:,i_s])
                linear_svr_y_predict = linear_svr.predict(test_X_svm)
                ans.append(evaluation_s(test_Y[:,i_s],linear_svr_y_predict)[0])
                ans_r.append(evaluation_s( test_Y[:,i_s],linear_svr_y_predict)[1])
            all_reslts_cc[i_fold,i_prob,:] = ans
            all_reslts_rmse[i_fold,i_prob,:] = ans_r

    all_reslts_cc = np.mean(all_reslts_cc,axis=0)
    all_reslts_rmse = np.mean(all_reslts_rmse,axis=0)
    regr = np.concatenate((all_reslts_cc,all_reslts_rmse),axis=0)
    meanregr = np.mean(regr, axis = 1).reshape(len(select_num_list)*2,1)
    regr = np.concatenate((regr,meanregr),axis=1)
    print(regr)



