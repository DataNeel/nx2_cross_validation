import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import datetime
import random
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool
from scipy.stats import f
pd.set_option('display.multi_sparse', False)

def evaluate_models(X,y,models,times, downsampling=[0],time_unit='s'):
    global t
    if time_unit=='m':
        t=60
    elif time_unit=='h':
        t=3600
    else:
        t=1
    results=[]
    for i in range(times):
        rows = random.sample(X.index, int(round(len(X)/2)))
        global X_a, y_a, X_b, y_b, iteration, X_a_adjusted, y_a_adjusted, X_b_adjusted, y_b_adjusted, adjustment
        iteration=i
        X_a=X.ix[rows]
        y_a=y[rows]
        X_b=X.drop(rows)
        y_b=y.drop(rows)
        yes_a=sum(y_a)
        yes_b=sum(y_b)
        len_a=len(y_a)
        len_b=len(y_b)
        for adjustment in downsampling:
            if adjustment!=0:
                keep_a=yes_a*(1-adjustment)/adjustment
                keep_b=yes_b*(1-adjustment)/adjustment
                lose_a=len_a-keep_a-yes_a
                lose_b=len_b-keep_b-yes_b
                if lose_a>=0 and lose_b>=0:
                    lose_a_rows=random.sample(X_a[y_a==0].index,int(lose_a))
                    lose_b_rows=random.sample(X_b[y_b==0].index,int(lose_b))
                    X_a_adjusted=X_a.drop(lose_a_rows)
                    y_a_adjusted=y_a.drop(lose_a_rows)
                    X_b_adjusted=X_b.drop(lose_b_rows)
                    y_b_adjusted=y_b.drop(lose_b_rows)
                    p=Pool(len(models))
                    p_results=p.map(two_fold_adjusted,models)
                    p.close()
                    p.join()
            else:
                p=Pool(len(models))
                p_results=p.map(two_fold,models)
                p.close()
                p.join()
            for pair in p_results:
                results.append(pair[0])
                results.append(pair[1])
    return pd.DataFrame(results, columns=['model','iteration','fold','downsampling','auc','fit_time','predict_time'])



def get_auc(y,probs):
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr, tpr)
    return roc_auc


def two_fold(model):
    results=[]
    start=datetime.datetime.now()
    model.fit(X_a,y_a)
    end=datetime.datetime.now()
    fit_time=(end-start).total_seconds()/t
    start=datetime.datetime.now()
    auc=get_auc(y_b,model.predict_proba(X_b)[:, 1])
    end=datetime.datetime.now()
    auc_time=(end-start).total_seconds()/t
    results.append([model.name,iteration,1,'none',auc,fit_time,auc_time])
    start=datetime.datetime.now()
    model.fit(X_b,y_b)
    end=datetime.datetime.now()
    fit_time=(end-start).total_seconds()/t
    start=datetime.datetime.now()
    auc=get_auc(y_a,model.predict_proba(X_a)[:, 1])
    end=datetime.datetime.now()
    auc_time=(end-start).total_seconds()/t
    results.append([model.name,iteration,2,'none',auc,fit_time,auc_time])
    return results


def two_fold_adjusted(model):
    results=[]
    start=datetime.datetime.now()
    model.fit(X_a_adjusted,y_a_adjusted)
    end=datetime.datetime.now()
    fit_time=(end-start).total_seconds()/t
    start=datetime.datetime.now()
    auc=get_auc(y_b,model.predict_proba(X_b)[:, 1])
    end=datetime.datetime.now()
    auc_time=(end-start).total_seconds()/t
    results.append([model.name,iteration,1,adjustment,auc,fit_time,auc_time])
    start=datetime.datetime.now()
    model.fit(X_b_adjusted,y_b_adjusted)
    end=datetime.datetime.now()
    fit_time=(end-start).total_seconds()/t
    start=datetime.datetime.now()
    auc=get_auc(y_a,model.predict_proba(X_a)[:, 1])
    end=datetime.datetime.now()
    auc_time=(end-start).total_seconds()/t
    results.append([model.name,iteration,2,adjustment,auc,fit_time,auc_time])
    return results


def rank_models(results):
    agg_results=results.groupby(['model','downsampling']).agg(np.mean)
    agg_results=agg_results.drop(['fold','iteration'],1)
    return agg_results.ix[np.argsort(agg_results.auc)[::-1]]

def significance(results):
    models=list(set(results.model))
    downsamples=list(set(results.downsampling))
    iterations=list(set(results.iteration))
    df1, df2 = len(iterations), len(iterations)*2
    sig_index=pd.MultiIndex.from_product([models,downsamples], names=['Model','Downsampling'])
    sig_matrix=pd.DataFrame(index=sig_index, columns=sig_index)
    for m1 in models:
        for d1 in downsamples:
            current_column=m1,d1
            for m2 in models:
                for d2 in downsamples:
                    current_row=m2,d2
                    numerator=[]
                    denominator=[]
                    for i in iterations:
                        model_1_fold_1=results.auc[(results.model==m1) & (results.downsampling==d1) & (results.iteration==i) & (results.fold==1)]
                        model_1_fold_2=results.auc[(results.model==m1) & (results.downsampling==d1) & (results.iteration==i) & (results.fold==2)]
                        model_2_fold_1=results.auc[(results.model==m2) & (results.downsampling==d2) & (results.iteration==i) & (results.fold==1)]
                        model_2_fold_2=results.auc[(results.model==m2) & (results.downsampling==d2) & (results.iteration==i) & (results.fold==2)]
                        diff1=float(model_1_fold_1)-float(model_2_fold_1)
                        diff2=float(model_1_fold_2)-float(model_2_fold_2)
                        numerator.append(diff1*diff1)
                        numerator.append(diff2*diff2)
                        avg_diff=(diff1+diff2)/2
                        denominator.append((diff1-avg_diff)**2 + (diff1-avg_diff)**2)
                        denominator.append((diff1-avg_diff)**2 + (diff1-avg_diff)**2)
                    if current_row!=current_column:
                        sig_matrix[current_column][current_row]=1-f.cdf(sum(numerator)/sum(denominator),10,5)
    return sig_matrix


forest = RandomForestClassifier(n_estimators = 10, n_jobs=2, max_depth=50)
forest.name='forest1'
forest2 = RandomForestClassifier(n_estimators = 50, n_jobs=4, max_depth=100)
forest2.name='forest2'
logit=LogisticRegression()
logit.name='logit1'
ada= AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators = 10)
ada.name='ada1'
gforest= GradientBoostingClassifier(n_estimators = 10, max_depth=2, subsample=.5)
gforest.name='gforest1'
gforest2= GradientBoostingClassifier(n_estimators = 50, max_depth=3, subsample=.5)
gforest2.name='gforest2'

from sklearn.datasets import make_classification
data=make_classification(n_samples=100000, n_features=100, n_informative=4, weights=[.95], flip_y=.02, n_repeated=13, class_sep=.5)
X=pd.DataFrame(data[0])
y=pd.Series(data[1])

results=evaluate_models(X,y,[logit,forest,forest2,ada,gforest,gforest2],5, downsampling=[0,.1,.2], time_unit='m')
rank_models(results)


significance(results)