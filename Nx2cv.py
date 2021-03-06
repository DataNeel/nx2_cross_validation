import pandas as pd
import datetime
import random
import numpy as np
from sklearn.metrics import roc_curve, auc
from multiprocessing import Pool
from scipy.stats import f
pd.set_option('display.multi_sparse', False)

#models for demonstration
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

#Evaluate a list of models N times with two-fold cross validation
def evaluate_models(X,y,models,times, undersampling=[0],time_unit='s'):
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
        for adjustment in undersampling:
            if adjustment!=0:
                X_a_adjusted, y_a_adjusted, a_good=undersample(X_a,y_a,adjustment)
                X_b_adjusted, y_b_adjusted, b_good=undersample(X_b,y_b,adjustment)
                if a_good & b_good:
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
    return pd.DataFrame(results, columns=['model','iteration','fold','undersampling','auc','fit_time','predict_time'])


#Calculate the area under the ROC curve for a set of actual classes and predicted probabilities
def get_auc(y,probs):
    fpr, tpr, thresholds = roc_curve(y,probs)
    roc_auc = auc(fpr, tpr)
    return roc_auc

def undersample(X,y,undersample_level):
    yes=sum(y)
    length=len(y)
    keep=yes*(1-undersample_level)/undersample_level
    lose=length-keep-yes
    if lose>0:
        lose_rows=random.sample(X.loc[y==0].index,int(lose))
        X_adjusted=X.drop(lose_rows)
        y_adjusted=y.drop(lose_rows)
        return X_adjusted, y_adjusted, True
    else:
        return X, y, False


#For a split of the data, run the cross validation and return accuracy and runtime results
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

#For a split of the data that has been undersampled, run the cross validation and return accuracy and runtime results
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

#aggregate the results table and order by average area under the ROC curve
def rank_models(results):
    agg_results=results.groupby(['model','undersampling']).agg(np.mean)
    agg_results=agg_results.drop(['fold','iteration'],1)
    return agg_results.ix[np.argsort(agg_results.auc)[::-1]]

#Generate a matrix of probabilities that two models are comparably accurate using F tests
def significance(results):
    models=list(set(results.model))
    undersamples=list(set(results.undersampling))
    iterations=list(set(results.iteration))
    df1, df2 = len(iterations), len(iterations)*2
    sig_index=pd.MultiIndex.from_product([models,undersamples], names=['Model','Undersampling'])
    sig_matrix=pd.DataFrame(index=sig_index, columns=sig_index)
    for m1 in models:
        for d1 in undersamples:
            current_column=m1,d1
            for m2 in models:
                for d2 in undersamples:
                    current_row=m2,d2
                    numerator=[]
                    denominator=[]
                    for i in iterations:
                        model_1_fold_1=results.auc[(results.model==m1) & (results.undersampling==d1) & (results.iteration==i) & (results.fold==1)]
                        model_1_fold_2=results.auc[(results.model==m1) & (results.undersampling==d1) & (results.iteration==i) & (results.fold==2)]
                        model_2_fold_1=results.auc[(results.model==m2) & (results.undersampling==d2) & (results.iteration==i) & (results.fold==1)]
                        model_2_fold_2=results.auc[(results.model==m2) & (results.undersampling==d2) & (results.iteration==i) & (results.fold==2)]
                        diff1=float(model_1_fold_1)-float(model_2_fold_1)
                        diff2=float(model_1_fold_2)-float(model_2_fold_2)
                        numerator.append(diff1*diff1)
                        numerator.append(diff2*diff2)
                        avg_diff=(diff1+diff2)/2
                        denominator.append((diff1-avg_diff)**2 + (diff1-avg_diff)**2)
                        denominator.append((diff1-avg_diff)**2 + (diff1-avg_diff)**2)
                    if current_row!=current_column:
                        sig_matrix[current_column][current_row]=1-f.cdf(sum(numerator)/sum(denominator),df2,df1)
    return sig_matrix


#Create classifiers to test
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

#Generate a fake binary classification dataset with extreme class imbalance
from sklearn.datasets import make_classification
data=make_classification(n_samples=100000, n_features=100, n_informative=4, weights=[.95], flip_y=.02, n_repeated=13, class_sep=.5)
X=pd.DataFrame(data[0])
y=pd.Series(data[1])

#Run the models
results=evaluate_models(X,y,[logit,forest,forest2,ada,gforest,gforest2],5, undersampling=[0,.1,.2], time_unit='m')

#Rank the model resutls by average area under the ROC curve
rank_models(results)

#Determine if each pair of models has statistically different accuracy levels
significance(results)