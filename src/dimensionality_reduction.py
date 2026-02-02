import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold,SelectKBest,f_regression,mutual_info_regression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sea
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DimensionalityReducer:
    def __init__(self,n_components=100,k_best=200,correlation_threshold=0.95):
        self.n_components=n_components
        self.k_best=k_best
        self.correlation_threshold=correlation_threshold
        self.pca=PCA(n_components=n_components,random_state=42)
        self.selector=SelectKBest(score_func=f_regression,k=k_best)
        self.variance_threshold=VarianceThreshold(threshold=0.01)
        self.scaler=StandardScaler()
        self.results_history={}
    def lead_data(self,features_path):
        df=pd.read_csv(features_path)
        self.original_df=df.copy() #cuvanje originalnih podataka
        self.metadata=df[['Smiles','pKi','source']].copy() #odavajanje features of metadata
        X=df.drop(columns=['Smiles','pKi','source'],errors='ignore')
        return X,self.metadata
    
    def analyze_variance(self,X,plot=True):
        variances=np.var(X,axis=0) #racuna varijansu za svaki feature
        variance_stats=pd.Series(variances).describe()
        print(f"Minimalna varijansa: {variance_stats['min']:.6f}")
        print(f"Maksimalna varijansa: {variance_stats['max']:.6f}")
        print(f"Srednja vrednost varijansa: {variance_stats['mean']:.6f}")
        print(f"Features sa nultom varijansom: {(variances==0).sum()}")
        print(f"Features sa varijansom <0.01: {(variances<0.01).sum()}")

        if plot:
            plt.figure(figsize=(12,5))
            #histogram
            plt.subplot(1,2,1)
            plt.hist(variances,bins=50,edgecolor='black',alpha=0.7)
            plt.axvline(x=0.01,coler='red',linestyle='--',label='Threshold=0.01')
            plt.xlabel('Variance')
            plt.ylabel('Number of features')
            plt.title('Distribution of feature variances')
            plt.legend()
            plt.grid(True,alpha=0.3)

            #box plot
            plt.subplot(1,2,2)
            plt.boxplot(variances,vert=False)
            plt.axvline(x=0.01,color='pink',linestyle='--',label='Threshold=0.01')
            plt.xlabel('Variance')
            plt.title('Box plot of feature variances')
            plt.legend()
            plt.grid(True,alpha=0.3)
            plt.tight_layout()
            plt.savefig('../results/variance_analysis.png',dpi=300,bbox_inches='tight')
            plt.show()
        
        #priprema threshold
        self.variance_threshold.fit(X)
        X_variance=self.variance_threshold.transform(X)
        kept_feat_mask=self.variance_threshold.get_support()
        kept_feat=X.columns[kept_feat_mask]

        print(f"Sacuvani features: {X_variance.shape[1]}")
        print(f"Uklonjeni features: {X.shape[1] - X_variance.shape[1]}")
        print(f"Procenat uklonjenih features: {(1-X_variance.shape[1])*100:.1f} %")
        return X_variance,kept_feat
    