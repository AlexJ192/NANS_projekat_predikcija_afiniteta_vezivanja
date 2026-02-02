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
    
    def analyze_correlations(self,X,feature_names,plot=True):
        corr_matrix=pd.DataFrame(X,columns=feature_names).corr().abs()
        print(f"Correlation matrix shape: {corr_matrix.shape()}")
        #nalazimo visoko korelisane feature parove
        upper=corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
        #analiza korelacija
        high_corr_pairs=[]
        for col in upper.columns:
            high_corr=upper[col][upper[col]>self.correlation_threshold]
            if len(high_corr)>0:
                for feature, corr_value in high_corr.items():
                    high_corr_pairs.append((col,feature,corr_value))
        
        print(f"Pronadjeno {len(high_corr_pairs)} parova feature-a sa korelacijom > {self.correlation_threshold}")
        if high_corr_pairs:
            #prikaz top 10 najvise korelisanih parova
            high_corr_pairs.sort(key=lambda x:x[2],reverse=True)
            print("Top 10 najkorelisanijih parova: ")
            for i,(f1,f2,corr) in enumerate(high_corr_pairs[:10]):
                print(f"{i+1}. {f1},{f2} (korelacija: {corr:.4f})")

        #identifikacija features-a za uklanjanje
        for_removal=set()
        for f1,f2,_ in high_corr_pairs:
            #zadrzavamo onaj u paru sa vecom varijansom
            var_f1=np.var(X[:,list(feature_names).index(f1)])
            var_f2=np.var(X[:,list(feature_names).index(f2)])
            for_removal.add(f1 if var_f1<var_f2 else f2)
        
        print(f"Broj feature-sa koje treba izbaciti zbog visoke korelacije: {len(for_removal)}")

        if plot and len(high_corr_pairs)>0:
            self.plot_correlation_analysis(corr_matrix,high_corr_pairs,feature_names)
        
        #uklanjanje visoko korelisanih feature-a
        keep_indices=[i for i,f in enumerate(feature_names) if f not in for_removal]
        X_corr=X[:,keep_indices]
        kept_features=[feature_names[i] for i in keep_indices]

        print(f"Sacuvani features: {X_corr.shape[1]}")
        print(f"Uklonjeni features: {X.shape[1]-X_corr.shape[1]}")

        return X_corr,kept_features
    
    def plot_correlation_analysis(self,corr_matrix,high_corr_pairs,feature_names):
        fig=plt.figure(figsize=(15,10))
        plt.subplot(2,2,1) #heatmap korelacione matrice
        sample_feat=np.random.choice(feature_names,min(50,len(feature_names)),replace=False) #uzimamo nasumcinih 50 za vizuelizaciju
        sample_corr=corr_matrix.loc[sample_feat,sample_feat]
        sea.heatmap(sample_corr,cmap='colorwarm',cemter=0,square=True,cbar_kws={"shrink":0.8})
        plt.title(f'Correlation heatmap (50 random features)')
        #distriubucija
        plt.subplot(2,2,2)
        #uzimamo gornji trougao bez  dijagonale
        corr_values=corr_matrix.values[np.triu_indices_from(corr_matrix.values,k=1)]
        plt.hist(corr_values,bins=50,edgecolor='purple',alpha=0.7)
        plt.axvline(x=self.correlation_threshold,color='red',linestyle='--',label=f'Threshold={self.correlation_threshold}')
        plt.xlabel('Absolute correlation')
        plt.ylabel('Frequency')
        plt.title('Distribution of feature correlations')
        plt.legend()
        plt.grid(True,alpha=0.3)
        #dendrogram za klastersku analizu
        plt.subplot(2,2,3)
        #uzimamo nasumicnih 30 features za dendrogram
        if len(feature_names)>30:
            sample_indices=np.random.choice(len(feature_names),30,replace=False)
            sample_feat=[feature_names[i] for i in sample_indices]
            sample_corr=corr_matrix.loc[sample_feat,sample_feat]
        else:
            sample_corr=corr_matrix
        
        #konvertovanje korelacione matrice u distancu
        distance_matrix=1-np.abs(sample_corr)
        condensed_dist=squareform(distance_matrix)
        linkage_matrix=hierarchy.linkage(condensed_dist,method='average')
        hierarchy.dendrogram(linkage_matrix,labels=sample_corr.columns,leaf_rotation=90,leaf_font_size=8)
        plt.title('Feature clustering dendrogram')
        plt.xlabel('Features')
        plt.ylabel('Distance (1 - |correlation|)')

        #scatterplot najvise korelisanog para
        plt.subplot(2,2,4)
        if high_corr_pairs:
            #uzimamo najvise kroelisan par
            top_pair=high_corr_pairs[0]
            f1_index=list(feature_names).index(top_pair[0])
            f2_index=list(feature_names).index(top_pair[1])
            #ucitavanje originalnih podataka
            original_X=self.original_df.drop(columns=['Smiles','pKi','source'],errors='ignore')
            plt.scatter(original_X.iloc[:,f1_index],original_X.iloc[:,f2_index],alpha=0.5,s=20)
            plt.xlabel(top_pair[0])
            plt.ylabel(top_pair[1])
            plt.title(f'Most correlated pairL r={top_pair[2]:.4f}')
            #dodavanje regression line
            z=np.polyfit(original_X.iloc[:,f1_index],original_X.iloc[:,f2_index],1)
            p=np.poly1d(z)
            x_range=np.linspace(original_X.iloc[:,f1_index].min(),original_X.iloc[:,f1_index].max(),100)
            plt.plot(x_range,p(x_range),"r--",alpha=0.8)
            plt.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/correlation_analysis.png',dpi=300,bbox_inches='tight')
        plt.show()

    def analyze_feature_importance(self,X,y,feature_names,plot=True):
        #analiza vaznosti features koristeci statisticke testove
        #iracunavanje f-rezultata za svaki feature
        self.selector.fit(X,y)
        f_scores=self.selector.scores_
        #izracunavanje mutual information
        mi_scores=mutual_info_regression(X,y,random_state=42)
        #kreiranje datafrejma sa rezultatima
        importance_df=pd.DataFrame({
            'Feature': feature_names,
            'F_Score':f_scores,
            'Mutual_Info':mi_scores
        })
        #sortiranje po f-score
        importance_df=importance_df.sort_values('F_Score',ascending=False)

        print(f"Ukupno feature-a: {len(feature_names)}")
        print(f"Top 10 najvaznijih feature-a po f-testu: ")
        print(importance_df.head(10).to_string())

        print("Bottom 10 najmanje vaznih feature-a po f-testu: ")
        print(importance_df.tail(10).to_string())

        if plot:
            self.plot_feature_importance(importance_df)
        
        #selektovanje k najboljih feture-a
        X_selected=self.selector.transform(X)
        selected_features=importance_df.head(self.k_best)['Feature'].tolist()

        print(f"Selektovano {self.k_best} najboljih feature-a")
        print(f"Minimalna f-test vrednost u selektovanim: {importance_df.iloc[self.k_best-1]['F_Score']:.4f}")
        print(f"Maksimalna f-test vrednost u neselektovanim: {importance_df.iloc[self.k_best]['F_Score']:.4f}")

        return X_selected,selected_features
    
    def plot_feature_importance(self,importance_df):
        fig,axes=plt.subplots(2,2,figsize=(15,12))
        #top feats by f-score
        top_n=20
        top_features=importance_df.head(top_n)
        axes[0,0].barh(range(top_n),top_features['F_Score'][::-1])
        axes[0,0].set_yticks(range(top_n))
        axes[0,0].set_yticklabels(top_features['Feature'][::-1],fontsize=8)
        axes[0,0].set_xlabel('F-Score')
        axes[0,0].set_title(f'Top {top_n} features by f-score')
        axes[0,0].grid(True,alpha=0.3)
        #distribucija f-scores
        axes[0,1].hist(importance_df['F_Score'],bins=50,edgecolor='black',alpha=0.7)
        axes[0,1].axvline(x=importance_df.iloc[self.k_best-1]['F_Score'],color='purple',linestyle='--',label=f'Cutoff for top {self.k_best}')
        axes[0,1].set_xlabel('F-Score')
        axes[0,1].set_ylabel('Number of features')
        axes[0,1].set_title('Distribution of f-scores')
        axes[0,1].legend()
        axes[0,1].grid(True,alpha=0.3)
        #scatter plot - f-score vs mutual info
        axes[1,0].scatter(importance_df['F_Score'],importance_df['Mutual_Info'],alpha=0.5,s=10)
        axes[1,0].set_xlabel('F-Score')
        axes[1,0].set_ylabel('Mutual Information')
        axes[1,0].set_title('F-Score vs Mutual Information')
        axes[1,0].grid(True,alpha=0.3)
        #cumualitve importance
        sorted_f=np.sort(importance_df['F_Score'])[::-1]
        cumulative_importance=np.cumsum(sorted_f)/np.sum(sorted_f)

        axes[1,1].plot(range(1,len(cumulative_importance)+1),cumulative_importance)
        axes[1,1].axhline(y=0.95,color='green',linestyle='--',label=f'Top {self.k_best} features')
        axes[1,1].axvline(x=self.k_best,color='red',linestyle='--',label=f'Top {self.k_best} features')
        axes[1,1].set_xlabel('Number of features')
        axes[1,1].set_ylabel('Cumulative f-score')
        axes[1,1].set_title('Cumulative feature importance')
        axes[1,1].legend()
        axes[1,1].grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/feature_importance_analysis.png',dpi=300,bbox_index='tight')
        plt.show()

    def apply_pca(self,X,feature_names,plot=True):
        X_scaled=self.scaler.fit_transform(X)
        X_pca=self.pca.fit_transform(X_scaled)
        explained_variance=self.pca.explained_variance_ratio_
        cumulative_variance=np.cumsum(explained_variance)

        print(f"Rezultati PCA analize: ")
        print(f"Originalna dimenzija: {X.shape}")
        print(f"Redukovana dimenzija: {X_pca.shape}")
        print(f"Broj komponenti: {self.n_components}")

        #pronalazenje broja komponenti za 95% varijanse
        n_components_95=np.argmax(cumulative_variance>=0.95)+1
        print(f"Broj komponenti neophodan za varijansu od 95% varijase: {n_components_95} ")
        if plot:
            self.plot_pca_analysis(explained_variance,cumulative_variance,X_pca,feature_names)
        loadings=self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        print(f"Top contributing features za prva 3 princial components: ")
        for i in range(3):
            component_loadings=loadings[:,i]
            top_indices=np.argsort(np.abs(component_loadings))[-5:][::-1]

            print(f"PC{i+1} (objasnjava {explained_variance[i]*100:.1f}% varijanse):")
            for index in top_indices:
                feature_name=feature_names[index] if index<len(feature_names) else f"Feature_{index}"
                loading_value=component_loadings[index]
                print(f"{feature_name}: {loading_value:.4f}")
    
        return X_pca
    
    def plot_pca_analysis(self,explained_variance,cumulative_variance,X_pca,feature_names):
        fig=plt.figure(figsize=(15,10))
        #scree plot
        plt.subplot(2,3,1)
        plt.plot(range(1,len(explained_variance)+1),explained_variance,'bo-')
        plt.xlabel('Principal component')
        plt.ylabel('Explained variance ratio')
        plt.title('Scree plot')
        plt.grid(True,alpha=0.3)
        #cumulative explained variance
        plt.subplot(2,3,2)
        plt.plot(range(1,len(cumulative_variance)+1),cumulative_variance,'ro-')
        plt.axline(y=0.95,color='green',linestyle='--',label='95% variance')
        plt.axhline(y=0.90,color='orange',linestyle='--',label='90% variance')
        plt.xlabel('Number of components')
        plt.ylabel('Cumulative explained variance')
        plt.title('Cumulative explained variance')
        plt.legend()
        plt.grid(True,alpha=0.3)
        #PC1 vs PC2
        plt.subplot(2,3,3)
        scatter=plt.scatter(X_pca[:,0],X_pca[:,1],alpha=0.5,s=20)
        plt.xlabel(f'PC1 ({explained_variance[0]*100:.1f}% variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]*100:.1f}% variance)')
        plt.title('PC1 vs PC2')
        plt.grid(True,alpha=0.3)
        #biplot
        plt.subplot(2,3,4)
        plt.scatter(X_pca[:,0],X_pca[:,1],alpha=0.3,s=10)

        loadings=self.pca.components_.T * np.sqrt(self.pca.explained_variance_)
        top_features=5 
        for i in range(top_features):
            plt.arrow(0,0,loadings[i,0]*3,loadings[i,1]*3,color='r',alpha=0.5,head_width=0.05)
            feature_name=feature_names[i] if i<len(feature_names) else f"Feature_{i}"
            plt.text(loadings[i,0]*3.2,loadings[i,1]*3.2,feature_name,color='r',fontsize=8)
        
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('Biplot (PC1 vs PC2 with loadings)')
        plt.grid(True,alpha=0.3)

        #heatmap loadings za prvih 10PCs
        plt.subplot(2,3,5)
        #top 20 features po apsolutnom loadingu
        loadings_abs=np.abs(loadings[:,:10]).sum(axis=1)
        top_indices=np.argsort(loadings_abs)[-20:][::-1]
        top_loadings=loadings[top_indices,:10]
        top_feature_names=[feature_names[i] for i in top_indices]

        sea.heatmap(top_loadings,cmap='coolwarm',center=0,xticklabels=[f'PC{i+1}' for i in range(10)],yticklabels=top_feature_names)
        plt.title('Loadings of top 20 features (first 10PCs)')

        plt.subplot(2,3,6)
        x=range(1,len(cumulative_variance)+1)
        plt.fill_between(x,0,cumulative_variance,alpha=0.3,label='Cumulative')
        plt.plot(x,explained_variance,'bo-',label='Individual')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance ratio')
        plt.title('Variance explained by components')
        plt.legend()
        plt.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/pca_analysis.png',dpi=300,bbox_inches='tight')
        plt.show()
        