import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import mlflow
import mlflow.sklearn
import dagshub

dagshub.init(repo_owner='faithvineco', repo_name='mlflow_exp_dagshub4', mlflow=True)

mlflow.set_experiment("water_exp2")
mlflow.set_tracking_uri('https://dagshub.com/faithvineco/mlflow_exp_dagshub4.mlflow')
data = pd.read_csv(r"C:\Users\USER\exp-tracking-dvc66\data\water_potability.csv")

train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)

def fill_missing_with_median(df):
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df

processed_train_data = fill_missing_with_median(train_data)
processed_test_data = fill_missing_with_median(test_data)

X_train = processed_train_data.iloc[:,0:-1].values
y_train = processed_train_data.iloc[:,-1].values

n_estimators = 1000

with mlflow.start_run():
    clf = GradientBoostingClassifier(n_estimators=n_estimators)
    clf.fit(X_train,y_train)

    with open('model.pkl','wb')as file:
        pickle.dump(clf,file)

    X_test = processed_test_data.iloc[:,0:-1].values
    y_test = processed_test_data.iloc[:,-1].values

    with open('model.pkl','rb')as file:
        model=pickle.load(file)

    y_pred = model.predict(X_test)

    acc=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1score=f1_score(y_test,y_pred)

    mlflow.log_metric('acc',acc)
    mlflow.log_metric('precision',precision)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('f1score',f1score)

    mlflow.log_param('n_estimators',n_estimators)

    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm,annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('confusion matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    mlflow.log_artifact('confusion_matrix.png')

    mlflow.sklearn.log_model(sk_model=clf,name="GradientBoostingClassifier",serialization_format="skops")
    
    mlflow.log_artifact(__file__)
    mlflow.set_tag("author","Paul Mubiru")
    mlflow.set_tag('Model','GB')

    print('acc',acc)
    print('precision',precision)
    print('recall', recall)
    print('f1score',f1score)