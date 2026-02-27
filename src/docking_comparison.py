import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sea
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from scipy import stats
import joblib
import warnings
warnings.filterwarnings('ignore')
from meeko import MoleculePreparation,PDBQTWriterLegacy
from sklearn.model_selection import train_test_split

Config={
    "vina_exe":r"C:\Users\Daisy\Documents\NUMERICKI_ALGORITMI\vina\vina_1.2.6_win.exe",
    "tripsin_pdbqt":"../data/docking/tripsin/1trn_bezliganda.pdbqt",
    "ligands_dir":"../data/docking/ligands",
    "results_dir":"../data/docking/results",
    "output_dir":"../results/docking",
    "center_x":48.5,
    "center_y":21.5,
    "center_z":30.5,
    "size_x":25.0,
    "size_y":25.0,
    "size_z":25.0,
    "exhaustiveness":15,
    "num_modes":9,
    "energy_range":3,
    "max_ligands":10,
}

def setup_dirs(config):
    for key in ["ligands_dir","results_dir","output_dir"]:
        os.makedirs(config[key],exist_ok=True)
    receptor_dir=os.path.dirname(config["tripsin_pdbqt"])
    os.makedirs(receptor_dir,exist_ok=True)

def affinity_to_pKi(affinty_kcal_mol,temp=298):
    R=0.001987
    RT=R*temp
    Kd_M=np.exp(affinty_kcal_mol/RT)
    pKi=-np.log10(Kd_M)
    return pKi

class PreparationOfLigands:
    def __init__(self,config):
        self.config=config
        os.makedirs(config["ligands_dir"],exist_ok=True)
    
    def smiles_to_pdbqt(self,smiles,name):
        try:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            mol=Chem.AddHs(mol) #vodonici
            embed_result=AllChem.EmbedMolecule(mol,AllChem.ETKDGv3()) #3d konformacija
            if embed_result==-1:
                AllChem.EmbedMolecule(mol,randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol,maxIters=500) #mmff force field
            sdf_path=os.path.join(self.config["ligands_dir"],f"{name}.sdf")
            writer=Chem.SDWriter(sdf_path)
            writer.write(mol)
            writer.close()
            pdbqt_path=os.path.join(self.config["ligands_dir"],f"{name}.pdbqt")
            self._sdf_to_pdqt(sdf_path,pdbqt_path,mol)
            if os.path.exists(pdbqt_path):
                return pdbqt_path
            else:
                return None
        except Exception as e:
            print(f"Doslo je do greske: {e}")
            return None
        
    def _sdf_to_pdqt(self,sdf_path,pdbqt_path,mol):
        prep=MoleculePreparation()
        mol_setup=prep.prepare(mol)
        pdbqt_string,is_ok,error=PDBQTWriterLegacy.write_string(mol_setup[0])
        if not is_ok:
            raise ValueError(f"Meeko error {error}")
        with open(pdbqt_path,'w') as fajl:
            fajl.write(pdbqt_string)
    
    def prepare_batch(self,df,smiles_col='Smiles',name_col=None):
        prepared=[]
        total=len(df)
        for index,row in df.iterrows():
            smiles=row[smiles_col]
            name=f"ligand_{index}" if name_col is None else str(row[name_col])
            name="".join(c if c.isalnum() or c=='_' else '_' for c in name)
            pdbqt_path=self.smiles_to_pdbqt(smiles,name)
            if pdbqt_path:
                prepared.append({'original_index':index,'name':name,'Smiles':smiles,'pdbqt_path':pdbqt_path})
        return pd.DataFrame(prepared)
    
class DockingRunner:
    def __init__(self,config):
        self.config=config
        os.makedirs(config["results_dir"],exist_ok=True)   

    def dock_ligand(self,ligand_pdbqt,ligand_name):
        output_pdbqt=os.path.join(self.config["results_dir"],f"{ligand_name}_out.pdbqt")
        log_file=os.path.join(self.config["results_dir"],f"{ligand_name}_log.txt")
        cmd = [
                self.config["vina_exe"],
                "--receptor",self.config["tripsin_pdbqt"],
                "--ligand",ligand_pdbqt,
                "--out",output_pdbqt,
                "--log",log_file,
                "--center_x",str(self.config["center_x"]),
                "--center_y",str(self.config["center_y"]),
                "--center_z",str(self.config["center_z"]),
                "--size_x",str(self.config["size_x"]),
                "--size_y",str(self.config["size_y"]),
                "--size_z",str(self.config["size_z"]),
                "--exhaustiveness", str(self.config["exhaustiveness"]),
                "--num_modes",str(self.config["num_modes"]),
                "--energy_range",str(self.config["energy_range"]),
            ]    
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=300,encoding='utf-8',errors='replace')
            best_affinity=self._parse_vina_log(log_file)
            return {
                'ligand_name':ligand_name,
                'best_affinity_kcal_mol':best_affinity,
                'output_pdbqt':output_pdbqt,
                'log_file':log_file,
                'success':best_affinity is not None
            }
        except subprocess.TimeoutExpired:
            return {'ligand_name':ligand_name,'best_affinity_kcal_mol':None,'success':False,'error':'timeout'}
        except Exception as e:
            return {'ligand_name':ligand_name,'best_affinity_kcal_mol':None,'success':False,'error':str(e)}
        
    def _parse_vina_log(self,log_file):
        if not os.path.exists(log_file):
            return None
        try:
            with open(log_file,'r',encoding='utf-8',errors='replace') as f:
                lines=f.readlines()
            for i, line in enumerate(lines):
                if '-----+------------+----------+----------' in line:
                    if i+1<len(lines):
                        next_line=lines[i+1].strip()
                        parts=next_line.split()
                        if len(parts)>=2:
                            return float(parts[1])
        except Exception:
            pass
        return None
    
    def run_batch(self,prepared_df):
        results=[]
        total=len(prepared_df)
        for i,(_, row) in enumerate(prepared_df.iterrows()):
            print(f"({i+1}/{total}) {row['name'][:30]}...",end=' ',flush=True)
            result=self.dock_ligand(row['pdbqt_path'],row['name'])
            result['Smiles']=row['Smiles']
            result['original_index']=row['original_index']
            results.append(result)
            if result['success']:
                affinity=result['best_affinity_kcal_mol']
                pki=affinity_to_pKi(affinity)
                print(f"{affinity:.2f} kcal/mol (pKi≈{pki:.2f})")
            else:
                print(f"{result.get('error','neuspešno')}")
        results_df=pd.DataFrame(results)
        success_rate=results_df['success'].mean() * 100
        successful=results_df[results_df['success']]
        print(f"Uspešnost dokinga: {success_rate:.1f}% ({len(successful)}/{total})")
        if len(successful)>0:
            avg_affinity=successful['best_affinity_kcal_mol'].mean()
            print(f"Prosečan afinitet vezivanja: {avg_affinity:.2f} kcal/mol")
        return results_df
    
class AnalyzeDocking:
    def __init__(self,config):
        self.config=config
        os.makedirs(config["output_dir"],exist_ok=True)
    
    def load_model_predictions(self):
        pred_path="../results/model_predictions_test.csv"
        if os.path.exists(pred_path):
            return pd.read_csv(pred_path, sep=';')
        try:
            model=joblib.load("../models/trained/RandomForest_scenario1.pkl")
            data=joblib.load("../models/saved_data/data.pkl")
            human_df=pd.read_csv("../data/features/human_trypsin_features.csv", sep=',')
            indices=np.arange(len(human_df))
            _,test_indices=train_test_split(indices,test_size=0.10,random_state=42)
            test_df=human_df.iloc[test_indices].reset_index(drop=True)
            feat_cols=[c for c in human_df.columns if c not in ['Smiles','pKi','source']]
            X_test=test_df[feat_cols].values
            y_test=test_df['pKi'].values
            X_var=X_test[:,data['variance_mask']]
            X_corr=X_var[:,data['correlation_mask']]
            X_selected=data['selector'].transform(X_corr)
            X_scaled=data['scaler'].transform(X_selected)
            X_pca=data['pca'].transform(X_scaled)
            predictions=model.predict(X_pca)
            pred_df=pd.DataFrame({
                'Smiles': test_df['Smiles'].values,
                'pKi_actual': y_test,
                'pKi_predicted': predictions
            })
            pred_df.to_csv(pred_path,sep=';',index=False)
            return pred_df
        except Exception as e:
            print(f"Model nije ucitan: {e}")
            return None
        
    def merge_results(self,docking_results,model_predictions,original_df):
        docking_success=docking_results[docking_results['success']].copy()
        docking_success['docking_pseudo_pKi']=docking_success['best_affinity_kcal_mol'].apply(affinity_to_pKi)
        merged=pd.merge(docking_success,original_df[['Smiles', 'pKi']],on='Smiles',how='left')
        if model_predictions is not None:
            merged=pd.merge(merged,model_predictions[['Smiles', 'pKi_predicted']],on='Smiles',how='left')
        return merged    
    
    def plot_comparison(self,merged_df):
        self.plot_docking_vs_experimental(merged_df)
        self.plot_model_vs_docking(merged_df)
        self.plot_triple_comparison(merged_df)
    
    def plot_docking_vs_experimental(self,df):
        valid=df.dropna(subset=['pKi','docking_pseudo_pKi'])
        if len(valid)==0:
            return
        fig,ax=plt.subplots(figsize=(8,7))
        scatter=ax.scatter(
            valid['pKi'],valid['docking_pseudo_pKi'],c=valid['pKi'],cmap='viridis',alpha=0.6,s=50,edgecolors='black', linewidth=0.5)
        min_val=min(valid['pKi'].min(),valid['docking_pseudo_pKi'].min())
        max_val=max(valid['pKi'].max(),valid['docking_pseudo_pKi'].max())
        ax.plot([min_val,max_val],[min_val,max_val],'r--',lw=2,alpha=0.7,label='Ideal')
        slope,intercept,r,_,_=stats.linregress(valid['pKi'],valid['docking_pseudo_pKi'])
        x_fit=np.linspace(valid['pKi'].min(),valid['pKi'].max(), 100)
        ax.plot(x_fit,slope * x_fit+intercept,'b-',lw=2,alpha=0.7,label=f'Fit (R={r:.3f})')
        rmse=np.sqrt(np.mean((valid['pKi']-valid['docking_pseudo_pKi'])**2))
        mae=np.mean(np.abs(valid['pKi']-valid['docking_pseudo_pKi']))
        ax.text(0.05, 0.95, f'RMSE={rmse:.3f}\nMAE={mae:.3f}\nR={r:.3f}',transform=ax.transAxes,fontsize=11,verticalalignment='top',bbox=dict(boxstyle='round',facecolor='lavender',alpha=0.7))
        plt.colorbar(scatter, ax=ax, label='Eksperimentalni pKi')
        ax.set_xlabel('Eksperimentalni pKi', fontsize=13, fontweight='bold')
        ax.set_ylabel('Doking pKi', fontsize=13, fontweight='bold')
        ax.set_title('AutoDock Vina vs Eksperiment\n',fontsize=14,fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/docking_vs_experimental.png",dpi=300,bbox_inches='tight')
        plt.close() 
    
    def plot_model_vs_docking(self,df):
        valid=df.dropna(subset=['pKi_predicted','docking_pseudo_pKi'])
        if len(valid)==0:
            return 
        fig,ax=plt.subplots(figsize=(8,7))
        ax.scatter(valid['pKi_predicted'],valid['docking_pseudo_pKi'],alpha=0.6,s=50,color='purple' edgecolors='black', linewidth=0.5)
        min_val=min(valid['pKi_predicted'].min(),valid['docking_pseudo_pKi'].min())
        max_val=max(valid['pKi_predicted'].max(),valid['docking_pseudo_pKi'].max())
        ax.plot([min_val, max_val],[min_val,max_val],'r--',lw=2,alpha=0.7)
        slope,intercept,r,_,_=stats.linregress(valid['pKi_predicted'],valid['docking_pseudo_pKi'])
        rmse=np.sqrt(np.mean((valid['pKi_predicted']-valid['docking_pseudo_pKi'])**2))
        ax.text(0.05,0.95,f'R = {r:.3f}\nRMSE={rmse:.3f}',transform=ax.transAxes,fontsize=11,verticalalignment='top',bbox=dict(boxstyle='round',facecolor='lightblue',alpha=0.7))
        ax.set_xlabel('ML Model pKi',fontsize=13,fontweight='bold')
        ax.set_ylabel('AutoDock Vina pKi',fontsize=13,fontweight='bold')
        ax.set_title('Model vs AutoDock Vina',fontsize=14,fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/model_vs_docking.png",dpi=300,bbox_inches='tight')
        plt.close()

    def plot_triple_comparison(self, df):
        valid = df.dropna(subset=['pKi', 'pKi_predicted', 'docking_pseudo_pKi'])
        if len(valid)==0:
            return
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        comparisons = [
            ('pKi','pKi_predicted','Eksperiment vs Model','steelblue'),
            ('pKi', 'docking_pseudo_pKi', 'Eksperiment vs Docking', 'teal'),
            ('pKi_predicted', 'docking_pseudo_pKi', 'Model vs Docking', 'purple')
        ]
        for ax, (x_col, y_col, title, color) in zip(axes, comparisons):
            ax.scatter(valid[x_col],valid[y_col],alpha=0.6, s=40,color=color,edgecolors='black',linewidth=0.3)
            min_val=min(valid[x_col].min(),valid[y_col].min())
            max_val=max(valid[x_col].max(),valid[y_col].max())
            ax.plot([min_val,max_val],[min_val,max_val],'r--',lw=1.5,alpha=0.7)
            slope,intercept,r,_,_=stats.linregress(valid[x_col],valid[y_col])
            rmse=np.sqrt(np.mean((valid[x_col]-valid[y_col])**2))
            ax.text(0.05, 0.95, f'R={r:.3f}\nRMSE={rmse:.3f}',transform=ax.transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle='round',facecolor='lightyellow',alpha=0.7))
            ax.set_xlabel(x_col.replace('_', ' ').title(),fontsize=11)
            ax.set_ylabel(y_col.replace('_', ' ').title(),fontsize=11)
            ax.set_title(title,fontsize=12,fontweight='bold')
            ax.grid(alpha=0.3)
        plt.suptitle('Model vs AutoDock Vina vs Eksperiment',fontsize=14,fontweight='bold',y=1.02)
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/triple_comparison.png",dpi=300,bbox_inches='tight')
        plt.close()

    def analyze_cinnamic_derivatives(self,docking_results,model_predictions):
        results = []
        for deriv in self.config["cinnamic_derivatives"]:
            name=deriv["name"]
            smiles=deriv["smiles"]
            print(f"\n{name}:")
            print(f"SMILES: {smiles}")
            docking_match=docking_results[docking_results['ligand_name']==name]
            docking_pki=None
            if not docking_match.empty and docking_match.iloc[0]['success']:
                affinity=docking_match.iloc[0]['best_affinity_kcal_mol']
                docking_pki=affinity_to_pKi(affinity)
            model_pki=None
            if model_predictions is not None:
                model_match=model_predictions[model_predictions['Smiles']==smiles]
                if not model_match.empty:
                    model_pki=model_match.iloc[0]['pKi_predicted']
            results.append({
                'name': name,
                'smiles': smiles,
                'docking_pseudo_pKi': docking_pki,
                'model_pKi': model_pki,
            })
        self._plot_cinnamic_bar(results)
        return pd.DataFrame(results)
    
    def _plot_cinnamic_bar(self, results):
        df=pd.DataFrame(results)
        fig,ax=plt.subplots(figsize=(10,6))
        x=np.arange(len(df))
        width = 0.25
        colors=['forestgreen', 'steelblue']
        labels=['AutoDock Vina', 'ML Model']
        cols=['docking_pseudo_pKi', 'model_pKi']
        for i,(col,label,color) in enumerate(zip(cols, labels, colors)):
            vals=df[col].values
            mask=~pd.isna(vals)
            if mask.any():
                ax.bar(x[mask]+i*width,vals[mask],width,label=label,color=color,alpha=0.8,edgecolor='black',lw=1.5)
        ax.set_xticks(x+width)
        ax.set_xticklabels(df['name'],fontsize=11)
        ax.set_ylabel('pKi vrednost',fontsize=12,fontweight='bold')
        ax.set_title('Derivati cimetne kiseline: Doking vs Model',fontsize=13,fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/cinnamic_derivatives.png",dpi=300,bbox_inches='tight')
        plt.close()

    def print_summary(self, merged_df):
        print("\n" + "*"*70)
        print("Model vs AutoDock Vina vs Eksperimentalne vrednosti")
        print("*"*70)
        docking_valid=merged_df.dropna(subset=['pKi','docking_pseudo_pKi'])
        d_rmse=np.sqrt(np.mean((docking_valid['pKi']-docking_valid['docking_pseudo_pKi'])**2))
        d_mae=np.mean(np.abs(docking_valid['pKi']-docking_valid['docking_pseudo_pKi']))
        d_r,_=stats.pearsonr(docking_valid['pKi'],docking_valid['docking_pseudo_pKi'])
        if 'pKi_predicted' in merged_df.columns:
            model_valid=merged_df.dropna(subset=['pKi','pKi_predicted'])
            m_rmse=np.sqrt(np.mean((model_valid['pKi']-model_valid['pKi_predicted'])**2))
            m_mae=np.mean(np.abs(model_valid['pKi']-model_valid['pKi_predicted']))
            m_r,_=stats.pearsonr(model_valid['pKi'], model_valid['pKi_predicted'])
        else:
            model_valid=pd.DataFrame()
            m_rmse=m_mae=m_r=None
        print(f"{'Metrika':<25} {'Model':<20} {'AutoDock Vina':<20}")
        print("*"*70)
        print(f"{'Broj uzoraka':<25} {len(model_valid):<20} {len(docking_valid):<20}")
        if m_rmse:
            print(f"{'RMSE':<25} {m_rmse:<20.3f} {d_rmse:<20.3f}")
            print(f"{'MAE':<25} {m_mae:<20.3f} {d_mae:<20.3f}")
            print(f"{'Pearson R':<25} {m_r:<20.3f} {d_r:<20.3f}")
        else:
            print(f"{'RMSE':<25} {'N/A':<20} {d_rmse:<20.3f}")
            print(f"{'MAE':<25} {'N/A':<20} {d_mae:<20.3f}")
            print(f"{'Pearson R':<25} {'N/A':<20} {d_r:<20.3f}")
        print("*"*70)

    def save_results(self, merged_df, cinnamic_df):
        merged_df.to_csv(f"{self.config['output_dir']}/complete_results.csv",index=False,sep=';')
        if cinnamic_df is not None:
            cinnamic_df.to_csv(f"{self.config['output_dir']}/cinnamic_derivatives.csv",index=False, sep=';')
    
class Docking:
    def __init__(self,config=None):
        self.config=config or Config
        setup_dirs(self.config)
        self.preparator=PreparationOfLigands(self.config)
        self.runner=DockingRunner(self.config)
        self.analyzer=AnalyzeDocking(self.config)   

    def load_test_set(self):
        human_df=pd.read_csv("../data/features/human_trypsin_features.csv", sep=',')
        indices=np.arange(len(human_df))
        _,test_indices=train_test_split(indices,test_size=0.10,random_state=42)
        test_df=human_df.iloc[test_indices][['Smiles', 'pKi']].reset_index(drop=True)
        max_lig = self.config.get("max_ligands", 50)
        if len(test_df) > max_lig:
            test_df = test_df.sample(n=max_lig, random_state=42).reset_index(drop=True)
        print(f"Test set: {len(test_df)} molekula")
        return test_df
    
    def run(self, test_df=None):
        if test_df is None:
            test_df=self.load_test_set()
        prepared_df=self.preparator.prepare_batch(test_df, smiles_col='Smiles')
        for deriv in self.config["cinnamic_derivatives"]:
            pdbqt_path=self.preparator.smiles_to_pdbqt(deriv["smiles"], deriv["name"])
            if pdbqt_path:
                cinn_row=pd.DataFrame([{
                    'original_index':-1,
                    'name':deriv["name"],
                    'Smiles':deriv["smiles"],
                    'pdbqt_path':pdbqt_path
                }])
                prepared_df=pd.concat([prepared_df,cinn_row],ignore_index=True)
        docking_results=self.runner.run_batch(prepared_df)
        docking_results.to_csv(f"{self.config['results_dir']}/raw_docking_results.csv",index=False,sep=';')
        model_predictions=self.analyzer.load_model_predictions()
        merged_df=self.analyzer.merge_results(docking_results,model_predictions,test_df)
        self.analyzer.plot_comparison(merged_df)
        cinnamic_df=self.analyzer.analyze_cinnamic_derivatives(docking_results,model_predictions)
        self.analyzer.save_results(merged_df)
        self.analyzer.print_summary(merged_df)
        return merged_df,docking_results,cinnamic_df
    

