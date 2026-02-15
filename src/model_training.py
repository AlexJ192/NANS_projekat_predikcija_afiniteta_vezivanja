import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sea
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def get_rsquared_adj(model, X,y):
    '''Returns adjusted r^2 score.'''
    y_pred=model.predict(X)
    r_squared = r2_score(y, y_pred)
    n = X.shape[0]
    p=X.shape[1]
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    return adjusted_r_squared

class ModelTrainer:
    def __init__(self):
        self.output_directory="../results"
        self.model_directory="../models/trained"
        os.makedirs(self.output_directory,exist_ok=True)
        os.makedirs(self.model_directory,exist_ok=True)
        self.model_config={
            'LinearRegression':LinearRegression(),
            'Ridge':Ridge(alpha=1.0),
            'Lasso':Lasso(alpha=0.01,max_iter=10000),
            'ElasticNet':ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=10000),
            'RandomForest':RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1),
            'GradientBoosting':GradientBoostingRegressor(n_estimators=100,random_state=42)
        }
        self.results={}
    #priprema
    def load_data(self):
        human=pd.read_csv("../data/reduced/human_pca.csv",sep=';')
        bovine=pd.read_csv("../data/reduced/bovine_pca.csv",sep=';')
        mixed=pd.read_csv(".../data/reduced.mixed_pca.csv",sep=';')
        for col in human.columns:
            if col.startswith('PC'):
                pc_cols=col

        #humani tripsin
        self.X_human=human[pc_cols].values
        self.Y_human=human['pKi'].values
        #kravlji tripsin
        self.X_bovine=bovine[pc_cols].values
        self.Y_bovine=bovine['pKi'].values
        #mixed
        self.X_mixed=mixed[pc_cols].values
        self.Y_mixed=mixed['pKi'].values
        self.source_mixed-mixed['source'].values

        print(f"Humani tripisn: {self.X_human.shape[0]} x {self.X_human.shape[1]}")
        print(f"Kravlji tripisn: {self.X_bovine.shape[0]} x {self.X_bovine.shape[1]}")
        print(f"Humani tripisn: {self.X_mixed.shape[0]} x {self.X_mixed.shape[1]}")

    def split_data(self):
        #70/15/15 split humanog
        x,self.X_test,y,self.Y_test=train_test_split(self.X_human,self.Y_human,test_size=0.15,random_state=42)
        self.X_train,self.X_val,self.Y_train,self.Y_val=train_test_split(x,y,test_size=0.15/0.85,random_state=42)
        print(f"Trening skup: {self.X_train.shape[0]} uzoraka")
        print(f"Validacioni skup: {self.X_val.shape[0]} uzorka")
        print(f"Test skup: {self.X_test.shape[0]} uzoraka")

        #mixed- isti test set, tj samo humani, ostalo ide u treniranje
        #izvalcenje indeksa humanog tripsina iz mixed
        human_mask=self.source_mixed=='human'
        X_mixed_human=self.X_mixed[human_mask]
        Y_mixed_human=self.Y_mixed[human_mask]
        X_mixed_bovine=self.X_mixed[~human_mask]
        Y_mixed_bovine=self.Y_mixed[~human_mask]

        #humani deo biva deljenj na train val test 
        X_m,_,y_m,_=train_test_split(X_mixed_human,Y_mixed_human,test_size=0.15,random_state=42)
        X_m_train,X_m_val,Y_m_train, Y_m_val=train_test_split(X_m,y_m,test_size=0.15/0.85,random_state=42)
        #mixed train= train humani i ceo kravlji
        self.X_mixed_train=np.vstack([X_m_train,X_mixed_bovine])
        self.Y_mixed_train=np.concatenate([Y_m_train,Y_mixed_bovine])
        self.X_mixed_val=X_m_val
        self.Y_miced_val=Y_m_val
        #sample weights za random forest scenario2
        #humani 1.0, bovine 0.3
        human_weights=np.ones(len(X_m_train))
        bovine_weights=np.full(len(X_mixed_bovine),0.3)
        self.sample_weights_mixed=np.concatenate([human_weights,bovine_weights])
        print(f"Mixed trening skup: {self.X_mixed_train.shape[0]}, od kojih je {len(X_m_train)} humani, a {len(X_mixed_bovine)} kravlji tripsin")

    def evaluate(self,model,X,y,label=""):
        y_pred=model.predict(X)
        rmse=np.sqrt(mean_squared_error(y,y_pred))
        r2=r2_score(y,y_pred)
        mae=mean_absolute_error(y,y_pred)
        #r2_adj=get_rsquared_adj(model,X,y) DODATIIIIIII
        if label:
            print(f"{label}: RMSE={rmse:.4f},R^2={r2:.4f},MAE={mae:.4f}")
        return {'rmse':rmse,'r2':r2,'mae':mae,'y_pred':y_pred}
    
    def train_scenario_1(self):
        #prvi scenario- samo humani tripsin
        scenario_results={}
        for name,model in self.model_config.items():
            print(f"{name}: ")
            model.fit(self.X_train,self.Y_train)
            validation_metrics=self.evaluate(model,self.X_val,self.Y_val,"Validacioni skup")
            test_metrics=self.evaluate(model,self.X_test,self.Y_test,"Test skup")
            scenario_results[name]={'model':model,'val':validation_metrics,'test':test_metrics}
            joblib.dump(model,f"{self.model_directory}/{name}_scenario1.pkl")
        self.results['scenario1']=scenario_results
    
