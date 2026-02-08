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
        self.correlation_mask=None #cuva features koje zadrzavam nakon corr reduct
        self.selected_feature_names=None #top k features
        self.feature_mask=None #koje kolone su izabrane

        self.original_feature_names=None #features iz humanog
        self.is_fitted=False #da li je treniran na humanom
        self.variance_mask=None #boolean maska za variance thresh

    def load_data(self,features_path):
        df=pd.read_csv(features_path)
        self.original_df=df.copy() #cuvanje originalnih podataka
        self.metadata=df[['Smiles','pKi','source']].copy() #odavajanje features of metadata
        X=df.drop(columns=['Smiles','pKi','source'],errors='ignore')
        return X,self.metadata
    
    def analyze_variance(self,X,plot=True,plot_prefix=""):
        if isinstance(X,pd.DataFrame):
            X_values=X.values
            feature_names=X.columns.tolist()
            original_shape=X.shape
        else:
            X_values=X
            feature_names=None
            original_shape=X.shape
        print(f"\n=== DEBUG INFO ===")
        print(f"X_values shape: {X_values.shape}")
        print(f"X_values dtype: {X_values.dtype}")

        # Proveri nekoliko originalnih vrednosti
        print(f"\nOriginal values sample (first 3 rows, first 5 columns):")
        print(X_values[:3, :5])

        #standardizacija
        scaler=StandardScaler()
        X_scaled=scaler.fit_transform(X_values)

        print(f"\nScaled values sample (first 3 rows, first 5 columns):")
        print(X_scaled[:3, :5])

        variances=np.var(X_scaled,axis=0) #racuna varijansu za svaki feature


        #
        print(f"\nVariances sample (first 10):")
        for i in range(min(10, len(variances))):
            print(f"  Feature {i}: {variances[i]:.10f}")
        
        print(f"\nUnique variance values: {np.unique(np.round(variances, 10))}")
        print(f"Number of unique variance values: {len(np.unique(np.round(variances, 10)))}")
        
        # OVO JE KLJUČNO: da li su sve varijanse 0 ili 1?
        zero_var = np.sum(np.abs(variances) < 1e-10)
        one_var = np.sum(np.abs(variances - 1.0) < 1e-5)
        other_var = len(variances) - zero_var - one_var
        
        print(f"\nVariance distribution:")
        print(f"  Exactly 0 variance: {zero_var}")
        print(f"  ~1 variance (±1e-5): {one_var}")
        print(f"  Other values: {other_var}")
        #


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
            plt.hist(variances,bins=150,edgecolor='black',alpha=0.7)
            plt.axvline(x=0.01,color='red',linestyle='--',label='Threshold=0.01')
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
            plt.savefig(f'../results/{plot_prefix}_variance_analysis_scaled.png',dpi=300,bbox_inches='tight')
            plt.show()
        
        #priprema threshold
        self.variance_threshold.fit(X_scaled)
        X_variance=self.variance_threshold.transform(X_scaled)
        kept_feat_mask=self.variance_threshold.get_support()

        self.variance_mask=kept_feat_mask #cuvanje variance maske za transform

        if feature_names is not None:
            kept_feat=[feature_names[i] for i in range(len(feature_names)) if kept_feat_mask[i]]
        else:
            kept_feat=[f"Feature_{i}" for i in range(len(kept_feat_mask)) if kept_feat_mask[i]]
        
        X_original_filtered=X_values[:,kept_feat_mask]
        print(f"Sacuvani features: {X_variance.shape[1]}")
        print(f"Uklonjeni features: {original_shape[1] - X_variance.shape[1]}")
        print(f"Procenat uklonjenih features: {(1-X_variance.shape[1]/original_shape[1])*100:.1f} %")

        self.variance_result={'X':X_original_filtered,'kept_features':kept_feat} #cuvamo rezultat
        return X_original_filtered,kept_feat
    
    def analyze_correlations(self,X,feature_names,plot=True,plot_prefix=""):
        corr_matrix=pd.DataFrame(X,columns=feature_names).corr().abs()
        print(f"Correlation matrix shape: {corr_matrix.shape}")
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
            self.plot_correlation_analysis(corr_matrix,high_corr_pairs,feature_names,plot_prefix)
        
        #uklanjanje visoko korelisanih feature-a
        keep_indices=[i for i,f in enumerate(feature_names) if f not in for_removal]
        self.correlation_mask=keep_indices #cuvam masku za kasniju upotrebu
        X_corr=X[:,keep_indices]
        kept_features=[feature_names[i] for i in keep_indices]

        print(f"Sacuvani features: {X_corr.shape[1]}")
        print(f"Uklonjeni features: {X.shape[1]-X_corr.shape[1]}")

        self.correlation_result={'X':X_corr,'kept_features':kept_features} #cuvamo rez
        return X_corr,kept_features
    
    def plot_correlation_analysis(self,corr_matrix,high_corr_pairs,feature_names,plot_prefix=""):
        fig=plt.figure(figsize=(15,10))
        plt.subplot(2,2,1) #heatmap korelacione matrice
        sample_feat=np.random.choice(feature_names,min(50,len(feature_names)),replace=False) #uzimamo nasumcinih 50 za vizuelizaciju
        sample_corr=corr_matrix.loc[sample_feat,sample_feat]
        sea.heatmap(sample_corr,cmap='coolwarm',center=0,square=True,cbar_kws={"shrink":0.8})
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
        plt.savefig(f'../results/{plot_prefix}_correlation_analysis.png',dpi=300,bbox_inches='tight')
        plt.show()

    def analyze_feature_importance(self,X,y,feature_names,plot=True,plot_prefix=""):
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
            self.plot_feature_importance(importance_df,plot_prefix)
        
        #selektovanje k najboljih feture-a
        X_selected=self.selector.transform(X)
        selected_features=importance_df.head(self.k_best)['Feature'].tolist()

        print(f"Selektovano {self.k_best} najboljih feature-a")
        print(f"Minimalna f-test vrednost u selektovanim: {importance_df.iloc[self.k_best-1]['F_Score']:.4f}")
        print(f"Maksimalna f-test vrednost u neselektovanim: {importance_df.iloc[self.k_best]['F_Score']:.4f}")

        self.importance_result={'X':X_selected,'kept_features':selected_features}

        return X_selected,selected_features
    
    def plot_feature_importance(self,importance_df,plot_prefix=""):
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
        axes[1,1].axhline(y=0.95,color='green',linestyle='--',label=f'95% importance')
        axes[1,1].axvline(x=self.k_best,color='red',linestyle='--',label=f'Top {self.k_best} features')
        axes[1,1].set_xlabel('Number of features')
        axes[1,1].set_ylabel('Cumulative f-score')
        axes[1,1].set_title('Cumulative feature importance')
        axes[1,1].legend()
        axes[1,1].grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'../results/{plot_prefix}_feature_importance_analysis.png',dpi=300,bbox_inches='tight')
        plt.show()

    def apply_pca(self,X,feature_names,plot=True,plot_prefix=""):
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
        print(f"Broj komponenti neophodan za varijansu od 95%: {n_components_95} ")
        if plot:
            self.plot_pca_analysis(explained_variance,cumulative_variance,X_pca,feature_names,plot_prefix)
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

        self.pca_result={'X':X_pca} #cuvamo rez
        return X_pca
    
    def plot_pca_analysis(self,explained_variance,cumulative_variance,X_pca,feature_names,plot_prefix=""):
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
        plt.axhline(y=0.95,color='green',linestyle='--',label='95% variance')
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
        plt.savefig(f'../results/{plot_prefix}_pca_analysis.png',dpi=300,bbox_inches='tight')
        plt.show()

    def generate_summary(self):
        original_features=2265
        print(f"  Total features: {original_features}")
        print(f"  Total samples: {len(self.metadata)}")
        print(f"  - Human: {sum(self.metadata['source'] == 'human')}")
        print(f"  - Bovine: {sum(self.metadata['source'] == 'bovine')}")
        
        print(f"\nREDUCTION STEPS:")
        
        # Proveri da li smo uradili sve korake
        steps_completed = []
        if hasattr(self, 'variance_result'):
            steps_completed.append('Variance Threshold')
            kept_variance = self.variance_result['kept_features']
            print(f"  1. Variance Threshold:")
            print(f"     - Removed features with variance < 0.01")
            print(f"     - Kept features: {len(kept_variance)}")
            print(f"     - Reduction: {((original_features - len(kept_variance))/original_features)*100:.1f}%")
        
        if hasattr(self, 'correlation_result'):
            steps_completed.append('Correlation Removal')
            kept_corr = self.correlation_result['kept_features']
            print(f"  2. Correlation Removal (threshold: {self.correlation_threshold}):")
            print(f"     - Removed highly correlated features")
            print(f"     - Kept features: {len(kept_corr)}")
            print(f"     - Reduction from previous step: {((len(kept_variance) - len(kept_corr))/len(kept_variance))*100:.1f}%")
        
        if hasattr(self, 'importance_result'):
            steps_completed.append('Feature Selection')
            kept_importance = self.importance_result['kept_features']
            print(f"  3. Feature Selection (SelectKBest, k={self.k_best}):")
            print(f"     - Selected top {self.k_best} features by F-score")
            print(f"     - Kept features: {len(kept_importance)}")
            print(f"     - Reduction from previous step: {((len(kept_corr) - len(kept_importance))/len(kept_corr))*100:.1f}%")
        
        if hasattr(self, 'pca_result'):
            steps_completed.append('PCA')
            print(f"  4. Principal Component Analysis:")
            print(f"     - Number of components: {self.n_components}")
            print(f"     - Explained variance: {np.sum(self.pca.explained_variance_ratio_):.3f}")
            print(f"     - Final dimensions: {self.n_components}")
            print(f"     - Overall reduction: {((original_features - self.n_components)/original_features)*100:.1f}%")
        
    
    
    def apply_saved_feat_selection(self,X,y,feature_names,saved_features_path):
        saved_features=pd.read_csv(saved_features_path)
        selected_feat_names=saved_features['feature_name'].tolist()

        #
        print(f"Broj prethodno odabranih features: {len(selected_feat_names)}")
        print(f"Primer odabranih features: {selected_feat_names[:5]}")
        #

        kept_indices=[]
        kept_features=[]

        for i,feature_name in enumerate(feature_names):
            if feature_name in selected_feat_names:
                kept_indices.append(i)
                kept_features.append(feature_name)
        X_selected=X[:,kept_indices]
        print(f"Finalne dimenzije: {X_selected.shape}")
        return X_selected,kept_features

    def save_selected_features(self,feature_names,prefix):
        features_df=pd.DataFrame({'feature_name':feature_names, 'rank':range(1,len(feature_names)+1)})
        feature_file=f"../models/{prefix}_selected_features.csv"
        features_df.to_csv(feature_file,index=False,sep=";")
        self.selected_feature_names=feature_names

    def _align_features(self,X,current_features,expected_features):
        curr_feat_map={name:index for index,name in enumerate(current_features)}
        n_samples=X.shape[0]
        n_expected=len(expected_features)
        X_aligned=np.zeros((n_samples,n_expected))
        missing_feats=[]
        extra_feats=[]
        matched_feats=0
        for exp_index,exp_feature in enumerate(expected_features):
            if exp_feature in curr_feat_map:
                curr_index=curr_feat_map[exp_feature]
                X_aligned[:,exp_index]=X[:,curr_index]
                matched_feats+=1
            else:
                X_aligned[:,exp_index]=0.0 #marker za missing, ne stavljamo vrednist na 0
                missing_feats.append(exp_feature)
        expected_set=set(expected_features)
        for curr_feat in current_features:
            if curr_feat not in expected_set:
                extra_feats.append(curr_feat)
        aligment={
            'n_matched':matched_feats,
            'n_missing':len(missing_feats),
            'n_extra':len(extra_feats),
            'missing_examples':missing_feats[:10],
            'extra_examples':extra_feats[:10],
            'match_percentage':(matched_feats/n_expected)*100
        }
        return X_aligned,aligment
    
    def _plot_bovine_pca(self,X_pca,plot_prefix="bovine"):
        y=self.metadata['pKi'].values
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        #pc1 vs pc2
        scatter=axes[0].scatter(X_pca[:,0],X_pca[:,1],c=y,cmap='viridis',alpha=0.6,s=30,edgecolors='k',linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({self.pca.explained_variance_ratio_[0]*100:.1f}%)')
        axes[0].set_ylabel(f'PC2 ({self.pca.explained_variance_ratio_[1]*100:.1f}%)')
        axes[0].set_title(f'{plot_prefix.upper()}:PC1 vs PC2') #colored by pKi
        axes[0].grid(True,alpha=0.3)
        cbar=plt.colorbar(scatter,ax=axes[0])
        cbar.set_label('pKi',rotation=270,labelpad=15)

        axes[1].hist(X_pca[:,0],bins=50,alpha=0.7,edgecolor='black',color='steelblue')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{plot_prefix.upper()}: PC1 Distribution')
        axes[1].grid(True,alpha=0.3,axis='y')

        mean_pc1=X_pca[:,0].mean()
        std_pc1=X_pca[:,0].std()
        axes[1].axvline(mean_pc1,color='red',linestyle='--',linewidth=2, label=f'Mean: {mean_pc1:.2f}')
        axes[1].axvline(mean_pc1+std_pc1,color='orange',linestyle=':',linewidth=1.5, label=f'+-1 Std: {std_pc1:.2f}')
        axes[1].axvline(mean_pc1-std_pc1,color='orange',linestyle=':',linewidth=1.5)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(f'../results/{plot_prefix}_pca_overview.png',dpi=300,bbox_inches='tight')
        plt.close()
        
    def fit(self,X,y,feature_names,plot_prefix='human'):
        self.original_feature_names=feature_names.copy() #cuvamo originalne feature names zaaligment
        if isinstance(X,pd.DataFrame):
            X=X.values
        X_variance,kept_variance=self.analyze_variance(pd.DataFrame(X,columns=feature_names),plot=True,plot_prefix=plot_prefix)
        X_corr,kept_corr=self.analyze_correlations(X_variance,kept_variance,plot=True,plot_prefix=plot_prefix)
        X_importance,kept_importance=self.analyze_feature_importance(X_corr,y,kept_corr,plot=True,plot_prefix=plot_prefix)
        self.save_selected_features(kept_importance,plot_prefix)
        X_pca=self.apply_pca(X_importance,kept_importance,plot=True,plot_prefix=plot_prefix)
        self.generate_summary()
        self.is_fitted=True
        return X_pca
    
    def transform(self,X,feature_names,plot_prefix="bovine"):
        if not self.is_fitted:
           raise ValueError("Neophodno je prvo redukovati humani tripsin")
        if isinstance(X,pd.DataFrame):
           X=X.values
        X_aligned,alignment_info=self._align_features(X,feature_names,self.original_feature_names)
        print(f"Original features: {len(feature_names)}")
        print(f"Expected features: {len(self.original_feature_names)}")
        print(f"Matched: {alignment_info['n_matched']} ({alignment_info['match_percentage']:.2f}%)")
        print(f"Missing: {alignment_info['n_missing']}")
        print(f"Extra (ignored): {alignment_info['n_extra']}")

        if alignment_info['n_missing']>0:
            print(f"Missing features: ")
            for feat in alignment_info['missing_examples']:
                print(f"{feat}")
        
        if alignment_info['match_percentage']<95:
            print(f"Samo {alignment_info['match_percentage']:.1f}% features se poklapa. model moze imati smanjenu preciznost")
        else:
            print(f"Aligment je okej ({alignment_info['match_percentage']:.1f}% poklapanja)")
        X=X_aligned
        X_variance=X[:,self.variance_mask]
        print(f"Zadrzano {X_variance.shape[1]} features")
        X_corr=X_variance[:,self.correlation_mask]
        print(f"Zadrzano {X_corr.shape[1]} features")
        X_selected=self.selector.transform(X_corr)
        print(f"Zadrzano {X_selected.shape[1]}")
        X_scaled=self.scaler.transform(X_selected)
        print(f"Zadrzano {X_scaled.shape[1]} features")
        X_pca=self.pca.transform(X_scaled)
        print(f"Zadrzano {X_pca.shape[1]} features")
        self._plot_bovine_pca(X_pca,plot_prefix)
        return X_pca
    
    def save(self,path="../models/saved_data"):
        os.makedirs(path,exist_ok=True)
        data={
            'variance_threshold':self.variance_threshold,
            'variance_mask':self.variance_mask,
            'scaler':self.scaler,
            'selector':self.selector,
            'pca':self.pca,
            'correlation_mask':self.correlation_mask,
            'original_features_names':self.original_feature_names,
            'selected_feature_names':self.selected_feature_names,
            'is_fitted':self.is_fitted,
            'config':{
                'n_components':self.n_components,
                'k_best':self.k_best,
                'correlation_threshold':self.correlation_threshold
            }
        }
        joblib.dump(data,f"{path}/data.pkl")

    @classmethod
    def load_existing(cls,path="../models/saved_data"):
        data=joblib.load(f"{path}/data.pkl")
        config=data['config']
        data_processing_elements=cls(n_components=config['n_components'],k_best=config['k_best'],correlation_threshold=config['correlation_threshold'])
        data_processing_elements.variance_threshold=data['variance_threshold']
        data_processing_elements.variance_mask=data['variance_mask']
        data_processing_elements.scaler=data['scaler']
        data_processing_elements.selector=data['selector']
        data_processing_elements.pca=data['pca']
        data_processing_elements.correlation_mask=data['correlation_mask']
        data_processing_elements.original_feature_names=data.get('original_features_names',None)
        data_processing_elements.selected_feature_names=data.get('selected_features_names',None)
        data_processing_elements.is_fitted=data.get('is_fitted',False)
        return data_processing_elements
    
    def verify_consistency(self,human_pca,bovine_pca):
        print("Dimenzije PCA: ")
        print(f"Humani {human_pca.shape}")
        print(f"Bovine {bovine_pca.shape}")
        if human_pca.shape[1]==bovine_pca.shape[1]:
            print("Dimenzije su identicne")
        else:
            print("Dimenzije PCA se razlikuju :(")
        
        for i in range(1,min(4,human_pca.shape[1])): #pc1,pc2,pc3
            pc_col=f'PC{i}'
            h_mean=human_pca[pc_col].mean()
            h_std=human_pca[pc_col].std()
            b_mean=bovine_pca[pc_col].mean()
            b_std=bovine_pca[pc_col].std()

            print(f"{pc_col}:")
            print(f"Humani: Mean({h_mean:>8.4f}), std({h_std:>8.4f})")
            print(f"Bovine: Mean({b_mean:>8.4f}), std({b_std:>8.4f})")

        fig=plt.figure(figsize=(16,12))
        #humani pc1 vs pc2
        ax1=plt.subplot(2,2,1)
        scatter1=ax1.scatter(human_pca['PC1'],human_pca['PC2'],c=human_pca['pKi'],cmap='viridis',alpha=0.6,s=40,edgecolors='k',linewidth=0.5)
        ax1.set_xlabel('PC1')
        ax1.set_ylabel('PC2')
        ax1.set_title('Human PC1 vs PC2',fontweight='bold',fontsize=12)
        ax1.grid(True,alpha=0.3)
        cbar1=plt.colorbar(scatter1,ax=ax1)
        cbar1.set_label('pKi',rotation=270,labelpad=15)

        #bovine pc1 vs pc2
        ax2=plt.subplot(2,2,2)
        scatter2=ax2.scatter(bovine_pca['PC1'],bovine_pca['PC2'],c=bovine_pca['pKi'],cmap='viridis',alpha=0.6,s=40,edgecolors='k',linewidth=0.5)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('Bovine PC1 vs PC2',fontweight='bold',fontsize=12)
        ax2.grid(True,alpha=0.3)
        cbar2=plt.colorbar(scatter2,ax=ax2)
        cbar2.set_label('pKi',rotation=270,labelpad=15)

        #humani + bovine
        ax3=plt.subplot(2,2,3)
        ax3.scatter(human_pca['PC1'],human_pca['PC2'],c='blue',alpha=0.5,s=30,edgecolors='darkblue',linewidth=0.3,label=f'Human (n={len(human_pca)})')
        ax3.scatter(bovine_pca['PC1'],bovine_pca['PC2'],c='red',alpha=0.5,s=30,edgecolors='darkred',linewidth=0.3,label=f'Bovine (n={len(bovine_pca)})')
        ax3.set_xlabel('PC1')
        ax3.set_ylabel('PC2')
        ax3.set_title('Human vs Bovine',fontweight='bold',fontsize=12)
        ax3.legend(loc='best',framealpha=0.9)
        ax3.grid(True,alpha=0.3)

        #pc1 distribucije
        ax4=plt.subplot(2,2,4)
        ax4.hist(human_pca['PC1'],bins=50,alpha=0.6,label='Human',density=True,color='blue',edgecolor='darkblue',linewidth=0.5)
        ax4.hist(bovine_pca['PC1'],bins=50,alpha=0.6,label='Bovine',density=True,color='red',edgecolor='darkred',linewidth=0.5)
        h_mean=human_pca['PC1'].mean()
        b_mean=bovine_pca['PC1'].mean()
        ax4.axvline(h_mean,color='blue',linestyle='--',linewidth=2,label=f'Human mean: {h_mean:.2f}')
        ax4.axvline(b_mean,color='red',linestyle='--',linewidth=2,label=f'Bovine mean: {b_mean:.2f}')
        ax4.set_xlabel('PC1')
        ax4.set_ylabel('Density')
        ax4.set_title('PC1 Distribution comparison',fontweight='bold',fontsize=12)
        ax4.legend(loc='best',framealpha=0.9)
        ax4.grid(True,alpha=0.3,axis='y')
        plt.tight_layout()
        plt.savefig('../results/comparison.png',dpi=300,bbox_inches='tight')
        plt.close()

        fig2,axes=plt.subplots(1,3,figsize=(18,5))
        scatter_h=axes[0].scatter(human_pca['PC2'],human_pca['PC3'],c=human_pca['pKi'],cmap='viridis',alpha=0.6,s=30)
        axes[0].set_xlabel('PC2')
        axes[0].set_ylabel('PC3')
        axes[0].set_title('Human PC2 vs PC3',fontweight='bold')
        axes[0].grid(True,alpha=0.3)
        plt.colorbar(scatter_h,ax=axes[0],label='pKi')

        scatter_b=axes[1].scatter(bovine_pca['PC2'],bovine_pca['PC3'],c=bovine_pca['pKi'],cmap='viridis',alpha=0.6,s=30)
        axes[1].set_xlabel('PC2')
        axes[1].set_ylabel('PC3')
        axes[1].set_title('Bovine PC2 vs PC3',fontweight='bold')
        axes[1].grid(True,alpha=0.3)
        plt.colorbar(scatter_b,ax=axes[1],label='pKi')

        #overlay
        axes[2].scatter(human_pca['PC2'],human_pca['PC3'],alpha=0.5,s=20,c='blue',label='Human')
        axes[2].scatter(bovine_pca['PC2'],bovine_pca['PC3'],alpha=0.5,s=20,c='red',label='Bovine')
        axes[2].set_xlabel('PC2')
        axes[2].set_ylabel('PC3')    
        axes[2].set_title('PC2 vs PC3', fontweight='bold')
        axes[2].legend()
        axes[2].grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig('../results/comparison_pc2_pc3.png',dpi=300,bbox_inches='tight')
        plt.close()

def main():

    os.makedirs("../results",exist_ok=True)
    os.makedirs("../data/reduced/consistent",exist_ok=True)
    os.makedirs("../models/saved_data",exist_ok=True)

    #ucitavanje podataka
    human_feats=pd.read_csv("../data/features/human_trypsin_features.csv")
    X_human=human_feats.drop(['Smiles','pKi','source'],axis=1).values
    y_human=human_feats['pKi'].values
    human_features_names=human_feats.drop(['Smiles','pKi','source'],axis=1).columns.tolist()
    metadata_human=human_feats[['Smiles','pKi','source']].copy()

    bovine_feats=pd.read_csv("../data/features/bovine_trypsin_features.csv")
    X_bovine=bovine_feats.drop(['Smiles','pKi','source'],axis=1).values
    y_bovine=bovine_feats['pKi'].values
    bovine_feature_names=bovine_feats.drop(['Smiles','pKi','source'],axis=1).columns.tolist()
    metadata_bovine=bovine_feats[['Smiles','pKi','source']].copy()
    print("Ucitani podaci: ")
    print(f'Humani {X_human.shape[0]:>4} uzoraka x {X_human.shape[1]:>4} features')
    print(f"Bovine {X_bovine.shape[0]:>4} uzoraka x {X_bovine.shape[1]:>4} features")

    human_reducer=DimensionalityReducer(n_components=100,k_best=200,correlation_threshold=0.95)
    human_reducer.original_df=human_feats
    human_reducer.metadata=metadata_human
    X_human_pca=human_reducer.fit(X_human,y_human,human_features_names,plot_prefix="human")

    human_reducer.original_df=bovine_feats
    human_reducer.metadata=metadata_bovine

    X_bovine_pca=human_reducer.transform(X_bovine,bovine_feature_names,plot_prefix="bovine")

    human_pca_df=pd.DataFrame(X_human_pca,columns=[f'PC{i+1}' for i in range(X_human_pca.shape[1])])
    human_pca_df=pd.concat([human_pca_df,metadata_human.reset_index(drop=True)],axis=1)

    bovine_pca_df=pd.DataFrame(X_bovine_pca,columns=[f'PC{i+1}' for i in range(X_bovine_pca.shape[1])])
    bovine_pca_df=pd.concat([bovine_pca_df,metadata_bovine.reset_index(drop=True)],axis=1)

    combined_pca_df=pd.concat([human_pca_df,bovine_pca_df],ignore_index=True)

    human_pca_df.to_csv("../data/reduced/consistent/human_pca.csv",sep=';',index=False)
    bovine_pca_df.to_csv("../data/reduced/consistent/bovine_pca.csv",sep=';',index=False)
    combined_pca_df.to_csv("../data/reduced/consistent/mixed_pca.csv",sep=';',index=False)

    print("Sacuvani datasetovi: ")
    print(f"Humani PCA: {human_pca_df.shape}")
    print(f"Bovine PCA: {bovine_pca_df.shape}")
    print(f"Mixed PCA: {combined_pca_df.shape}")

    human_reducer.save()
    human_reducer.verify_consistency(human_pca_df,bovine_pca_df)
    return human_reducer,human_pca_df,bovine_pca_df,combined_pca_df

if __name__=="__main__":
    reducer,results_h,results_b,results_mixed=main()