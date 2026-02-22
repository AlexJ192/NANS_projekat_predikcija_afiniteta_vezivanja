import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ShapAnalyzer:
    def __init__(self):
        self.output_directory='../results/shap'
        os.makedirs(self.output_directory,exist_ok=True)
    
    def load_data(self):
        human_pca=pd.read_csv("../data/reduced/human_pca.csv",sep=';')
        pc_cols=[c for c in human_pca.columns if c.startswith('PC')]
        self.X_pca=human_pca[pc_cols].values
        self.y=human_pca['pKi'].values
        self.pc_cols=pc_cols
        print(f"PCA podaci: {self.X_pca.shape}")
        self.rf_model=joblib.load("../models/trained/RandomForest_scenario1.pkl") #ucitavanje najboljeg modela
        data=joblib.load("../models/saved_data/data.pkl") #ucitavanje originalnih features-a
        self.pca_transformer=data['pca']
        self.selector=data['selector']
        self.scaler=data['scaler']
        self.variance_mask=data['variance_mask']
        self.correlation_mask=data['correlation_mask']
        human_og=pd.read_csv("../data/features/human_trypsin_features.csv",sep=';') #og feat names
        all_feats=human_og.drop(columns=['Smiles','pKi','source'],errors='ignore').columns.tolist()
        feats_variance=[all_feats[i] for i in range(len(all_feats)) if self.variance_mask[i]]
        feats_correlation=[feats_variance[i] for i in self.correlation_mask]
        selector_mask=self.selector.get_support()
        self.original_feats_names=[feats_correlation[i] for i in range(len(feats_correlation)) if selector_mask[i]] 
        print(f"Broj originalnih features: {len(self.original_feats_names)}")
        print(f"Top 5: {self.original_feats_names[:5]}")
        #replikacija trening mdoela
        x,self.X_test_pca,y,self.y_test=train_test_split(self.X_pca,self.y,test_size=0.15,random_state=42)
        self.X_train_pca,self.X_val_pca,_,_=train_test_split(x,y,test_size=0.15/0.85,random_state=42)
        print(f"Test set:{self.X_test_pca.shape[0]} uzoraka")
    
    def _pca_to_og(self,X_pca):
        X_scaled_approx=self.pca_transformer.inverse_transform(X_pca)
        X_og_approx=self.scaler.inverse_transform(X_scaled_approx)
        return X_og_approx
    def compute_shap(self):
        explainer=shap.TreeExplainer(self.rf_model)
        self.shap_values_pca=explainer.shap_values(self.X_test_pca)
        self.expected_value=explainer.expected_value
        print(f"Shap shape (pc): {self.shap_values_pca.shape}")
        self.X_test_og=self._pca_to_og(self.X_test_pca)
        print(f"X_test_original shape: {self.X_test_og.shape}")
        self.shape_values_og=self.shap_values_pca @ self.pca_transformer.components_
        print(f"Shap shape(og): {self.shape_values_og.shape}")
    def plot_summary_bar(self):
        mean_abs_shap=np.abs(self.shape_values_og).mean(axis=0)
        top_indices=np.argsort(mean_abs_shap)[-20:][::-1]
        top_feats=[self.original_feats_names[i] for i in top_indices]
        top_shap_vals=mean_abs_shap[top_indices]
        fig,ax=plt.subplots(figsize=(10,8))
        colors=plt.cm.RdYlGn(np.linspace(0.3,0.9,20))[::-1]
        bars=ax.barh(range(20),top_shap_vals[::-1],color=colors,edgecolor='black',linewidth=0.5)
        ax.set_yticks(range(20))
        ax.set_yticklabels(top_feats[::-1],fontsize=9)
        ax.set_xlabel('Mean',fontsize=12)
        ax.set_title('Top 20 most important features',fontsize=13,fontweight='bold')
        ax.grid(True,alpha=0.3,axis='x')
        for bar,val in zip(bars,top_shap_vals[::-1]):
            ax.text(val+0.001,bar.get_y()+bar.get_height()/2,f'{val:.4f}',va='center',fontsize=7)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/shap_summary_bar.png',dpi=300,bbox_inches='tight')
        plt.close()
    def plot_beeswarm(self):
        #top 15 feats
        mean_abs_shap=np.abs(self.shape_values_og).mean(axis=0)
        top_indices=np.argsort(mean_abs_shap)[-15:]
        shap_top=self.shape_values_og[:,top_indices]
        X_top=self.X_test_og[:,top_indices]
        feature_names=[self.original_feats_names[i] for i in top_indices]
        shap_exp=shap.Explanation(values=shap_top,base_values=self.expected_value,data=X_top,feature_names=feature_names)
        fig,ax=plt.subplots(figsize=(12,8))
        shap.plots.beeswarm(shap_exp,max_display=15,show=False)
        plt.title('Distribution of influence of features',fontsize=13,fontweight='bold',pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/shap_beeswarm.png',dpi=300,bbox_inches='tight')
        plt.close()
    def plot_waterfall(self):
        y_pred=self.rf_model.predict(self.X_test_pca)
        high_pKi=np.argmax(y_pred)
        low_pKi=np.argmin(y_pred)
        mid_pKi=np.argmin(np.abs(y_pred-np.median(y_pred)))
        molecules={
            'Visoki pKi':high_pKi,
            'Srednji pKi': mid_pKi,
            'Niski pKi':low_pKi
        }
        mean_abs_shap=np.abs(self.shape_values_og).mean(axis=0)
        top_indices=np.argsort(mean_abs_shap)[-15:] #top 15 feats
        feat_names=[self.original_feats_names[i] for i in top_indices]
        fig,axes=plt.subplots(1,3,figsize=(20,8))
        for ax,(label,pki) in zip(axes,molecules.items()):
            shap_mol=self.shape_values_og[pki,top_indices]
            X_mol=self.X_test_og[pki,top_indices]
            shap_exp=shap.Explanation(values=shap_mol,base_values=self.expected_value,data=X_mol,feature_names=feat_names)
            plt.sca(ax)
            shap.plots.waterfall(shap_exp,max_display=12,show=False)
            ax.set_title(f'{label}\npKi_pred={y_pred[pki]:.3f},pki_stvarno={self.y_test[pki]:.3f}',fontweight='bold',fontsize=10,pad=10)
        plt.suptitle('Explanation of prediction for individual molecules',fontsize=10,fontweight='bold',y=1.02)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/shap_waterfall.png',dpi=300,bbox_inches='tight')
        plt.close()

    def plot_dependence(self):
        mean_abs_shap=np.abs(self.shape_values_og).mean(axis=0)
        top4_indices=np.argsort(mean_abs_shap)[-4:][::-1]
        fig,axes=plt.subplots(2,2,figsize=(14,10))
        axes=axes.flatten() 
        for ax,feat_index in zip(axes,top4_indices):
            feat_name=self.original_feats_names[feat_index]
            feat_vals=self.X_test_og[:,feat_index]
            shap_vals=self.shape_values_og[:,feat_index]
            scatter=ax.scatter(feat_vals,shap_vals,c=feat_vals,cmap='viridis',alpha=0.7,s=40,edgecolors='black',linewidth=0.3)
            ax.axhline(y=0,color='black',linestyle='--',linewidth=1)
            ax.set_xlabel(feat_name,fontsize=11)
            ax.set_ylabel('Shap values',fontsize=11)
            ax.set_title(f'Dependance:{feat_name}',fontweight='bold',fontsize=11)
            ax.grid(True,alpha=0.3)
            plt.colorbar(scatter,ax=ax,label=feat_name)
            #trend linija
            z=np.polyfit(feat_vals,shap_vals,1)
            p=np.poly1d(z)
            x_range=np.linspace(feat_vals.min(),feat_vals.max(),100)
            ax.plot(x_range,p(x_range),'hotpink--',linewidth=1.5,alpha=0.8,label=f'Trend')
            ax.legend(fontsize=8)
        plt.suptitle('Influence of the top 4 features on prediction of pKi',fontsize=13,fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/shap_dependence.png',dpi=300,bbox_inches='tight')
        plt.close()

    def print_summary(self):
        mean_abs_shap=np.abs(self.shape_values_og).mean(axis=0)
        top10_indices=np.argsort(mean_abs_shap)[-10:][::-1]
        print("*"*65)
        print("Top 10 najvaznijih features-a za predikciju pKi:")
        print("*"*65)
        print(f"{'Rbr':<5} {'Feature':<25} {'Mean':<15} {'Tip'}")
        print("*"*65)
        for rbr,index in enumerate(top10_indices,1):
            name=self.original_feats_names[index]
            val=mean_abs_shap[index]
            if name.startswith('MFP_'):
                tip='Morganov fingerprint'
            elif name.startswith('fr_'):
                tip='Funkcionalna grupa'
            elif name in ['qed','MolWt','LogP','TPSA']:
                tip='Fizicko-hemisjka osobina'
            elif 'VSA' in name or 'EState' in name:
                tip='Toploski deskriptor'
            else:
                tip='Molekulski deskriptor'
            print(f"{rbr:<5} {name:<25} {val:<15.4f} {tip}")
        print("Grupna analiza svih features-a: ")
        mfp_mask=[n.startswith('MFP_') for n in self.original_feats_names]
        fr_mask=[n.startswith('fr_') for n in self.original_feats_names]
        other=[not (m or f) for m,f in zip(mfp_mask,fr_mask)]
        print(f"Morgan fingerprints: mean={mean_abs_shap[mfp_mask].mean():.4f}")
        print(f"Funkcionalne grupe: mean={mean_abs_shap[fr_mask].mean():.4f}")
        print(f"Ostali deskriptori: mean={mean_abs_shap[other].mean():.4f}")
    
    def run_shap(self):
        self.load_data()
        self.compute_shap()
        self.plot_summary_bar()
        self.plot_beeswarm()
        self.plot_waterfall()
        self.plot_dependence()
        self.print_summary()

def main():
    analyzer=ShapAnalyzer()
    analyzer.run_shap()
    return analyzer

if __name__=="__main__":
    analyzer=main()

