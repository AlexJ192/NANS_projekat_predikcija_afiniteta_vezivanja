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
        mixed=pd.read_csv("../data/reduced/mixed_pca.csv",sep=';')
        pc_cols=[col for col in human.columns if col.startswith('PC')]

        #humani tripsin
        self.X_human=human[pc_cols].values
        self.Y_human=human['pKi'].values
        #kravlji tripsin
        self.X_bovine=bovine[pc_cols].values
        self.Y_bovine=bovine['pKi'].values
        #mixed
        self.X_mixed=mixed[pc_cols].values
        self.Y_mixed=mixed['pKi'].values
        self.source_mixed=mixed['source'].values

        print(f'Humani tripsin: {self.X_human.shape[0]} x {self.X_human.shape[1]}')
        print(f'Kravlji tripsin: {self.X_bovine.shape[0]} x {self.X_bovine.shape[1]}')
        print(f'Mixed dataset: {self.X_mixed.shape[0]} x {self.X_mixed.shape[1]}')

    def split_data(self):
        #70/15/15 split humanog
        x,self.X_test,y,self.Y_test=train_test_split(self.X_human,self.Y_human,test_size=0.15,random_state=42)
        self.X_train,self.X_val,self.Y_train,self.Y_val=train_test_split(x,y,test_size=0.15/0.85,random_state=42)
        print(f'Trening skup: {self.X_train.shape[0]} uzoraka')
        print(f'Validacioni skup: {self.X_val.shape[0]} uzorka')
        print(f'Test skup: {self.X_test.shape[0]} uzoraka')

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
        self.Y_mixed_val=Y_m_val
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
        n=X.shape[0]
        p=X.shape[1]
        r2_adj=1-(1-r2)*(n-1)/(n-p-1)
        if label:
            print(f"{label}: RMSE={rmse:.4f},R^2={r2:.4f}, R^2_adj= {r2_adj:.4f}, MAE={mae:.4f}")
        return {'rmse':rmse,'r2':r2,'r2_adj':r2_adj,'mae':mae,'y_pred':y_pred}
    
    def train_scenario_1(self):
        #prvi scenario- samo humani tripsin
        print("*"*30)
        print("Trening samo na humanim podacima")
        scenario_results={}
        for name,model in self.model_config.items():
            print(f"{name}: ")
            model.fit(self.X_train,self.Y_train)
            validation_metrics=self.evaluate(model,self.X_val,self.Y_val,"Validacioni skup")
            test_metrics=self.evaluate(model,self.X_test,self.Y_test,"Test skup")
            scenario_results[name]={'model':model,'val':validation_metrics,'test':test_metrics}
            joblib.dump(model,f"{self.model_directory}/{name}_scenario1.pkl")
        self.results['scenario1']=scenario_results
    
    def train_scenario_2(self):
        print("*"*30)
        print("Transfer learning")
        scenario_results={}
        for name in self.model_config.keys():
            print(f"{name}:")
            if name=='LinearRegression':
                model=self._transfer_linear(LinearRegression())
            elif name=='Ridge':
                model=self._transfer_linear(Ridge(alpha=1.0))
            elif name=='Lasso':
                model=self._transfer_linear(Lasso(alpha=0.01,max_iter=10000))
            elif name=='ElasticNet':
                model=self._transfer_linear(ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=10000))
            elif name=='RandomForest':
                model=self._transfer_rf()
            elif name=='GradientBoosting':
                model=self._transfer_gb()
            
            validation_metrics=self.evaluate(model,self.X_val,self.Y_val,"Validacioni skup")
            test_metrics=self.evaluate(model,self.X_test,self.Y_test,"Test skup")

            scenario_results[name]={'model':model,'val':validation_metrics,'test':test_metrics}
            joblib.dump(model,f"{self.model_directory}/{name}_scenario2.pkl")
        self.results['scenario2']=scenario_results
    
    def _transfer_linear(self,model):
        model.fit(self.X_mixed_train,self.Y_mixed_train,sample_weight=self.sample_weights_mixed)
        return model
    
    def _transfer_gb(self):
        #trening 50 stabala na bovine, pa dodajemo jos 50 za humani
        model=GradientBoostingRegressor(n_estimators=50,warm_start=True,random_state=42)
        model.fit(self.X_bovine,self.Y_bovine)
        print(f"Pretrain na kravljem tripsinu: RMSE={np.sqrt(mean_squared_error(self.Y_bovine,model.predict(self.X_bovine))):.4f}")
        model.n_estimators=100
        model.fit(self.X_train,self.Y_train)
        return model
    def _transfer_rf(self):
        model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
        model.fit(self.X_mixed_train,self.Y_mixed_train,sample_weight=self.sample_weights_mixed)
        return model

    def train_scenario_3(self):
        print("*"*30)
        print("Kombinovani dataset")
        scenario_results={}
        for name,_ in self.model_config.items():
            print(f"{name}:")
            if name=='LinearRegression':
                model=LinearRegression()
            elif name=='Ridge':
                model=Ridge(alpha=1.0)
            elif name=='Lasso':
                model=Lasso(alpha=0.01,max_iter=10000)
            elif name=='ElasticNet':
                model=ElasticNet(alpha=0.01,l1_ratio=0.5,max_iter=10000)
            elif name=='RandomForest':
                model=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1)
            elif name=='GradientBoosting':
                model=GradientBoostingRegressor(n_estimators=100,random_state=42)
            
            #fittovanje na mixed setu
            model.fit(self.X_mixed_train,self.Y_mixed_train)
            validation_metrics=self.evaluate(model,self.X_mixed_val,self.Y_mixed_val,"Validacioni skup")
            test_metrics=self.evaluate(model,self.X_test,self.Y_test,"Test skup")
            scenario_results[name]={'model':model,'val':validation_metrics,'test':test_metrics}
            joblib.dump(model,f"{self.model_directory}/{name}_scenario3.pkl")
        self.results['scenario3']=scenario_results
    
    def compare_scenarios(self):
        scenario_labels={'scenario1':'Samo Humani', 'scenario2':'Transfer Learning','scenario3': 'Kombinovani dataset'}
        #tabela rezultata
        rows=[]
        for sc_key,sc_label in scenario_labels.items():
            if sc_key not in self.results:
                continue
            for model_name,res in self.results[sc_key].items():
                rows.append({'Scenario':sc_label,'Model':model_name,'RMSE':round(res['test']['rmse'],4),'R^2':round(res['test']['r2'],4),'R^2_adj':round(res['test']['r2_adj'],4),'MAE':round(res['test']['mae'],4)})
        results_df=pd.DataFrame(rows)
        #efikasnost transfera
        if 'scenario1' in self.results and 'scenario2' in self.results:
            for model_name in self.model_config.keys():
                rmse_base=self.results['scenario1'][model_name]['test']['rmse']
                rmse_tl=self.results['scenario2'][model_name]['test']['rmse']
                transfer_efficiency=rmse_tl/rmse_base
                mask=(results_df['Scenario']=='Transfer Learning') & (results_df['Model']==model_name)
                results_df.loc[mask,'Transfer Efficiency']=round(transfer_efficiency,4)
        print("*"*30)
        print("Rezultati test seta: ")
        print(results_df.to_string(index=False))
        results_df.to_csv(f"{self.output_directory}/model_comparison.csv",index=False)

        self._plot_rmse(results_df)
        self._plot_r2(results_df)
        self._plot_r2_adj(results_df)
        self._plot_best_model()
        self._plot_transfer_efficiency()
        self._plot_residuals()
        return results_df
    
    def _plot_rmse(self,results_df):
        fig,ax=plt.subplots(figsize=(14,6))
        models=list(self.model_config.keys())
        scenarios=['Samo Humani', 'Transfer Learning','Kombinovani dataset']
        colors=['orchid','teal','mediumpurple']
        x=np.arange(len(models))
        width=0.25
        for i,(sc,color) in enumerate(zip(scenarios,colors)):
            sc_data=results_df[results_df['Scenario']==sc]
            rmse_vals=[sc_data[sc_data['Model']==m]['RMSE'].values[0] if len(sc_data[sc_data['Model']==m])>0 else 0 for m in models]
            bars=ax.bar(x+i*width,rmse_vals,width,label=sc,color=color,alpha=0.85,edgecolor='black')
            for bar,val in zip(bars,rmse_vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.3f}',ha='center',va='bottom',fontsize=7)
        ax.set_xlabel('Model',fontsize=12)
        ax.set_ylabel('RMSE',fontsize=12)
        ax.set_title('RMSE Overview',fontsize=13,fontweight='bold')
        ax.set_xticks(x+width)
        ax.set_xticklabels(models,rotation=0)
        ax.legend()
        ax.grid(True,alpha=0.3,axis='y')
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/rmse_comparison.png',dpi=300,bbox_inches='tight')
        plt.close()
    
    def _plot_r2(self,results_df):
        fig,ax=plt.subplots(figsize=(14,6))
        models=list(self.model_config.keys())
        scenarios=['Samo Humani', 'Transfer Learning','Kombinovani dataset']
        colors=['orchid','teal','mediumpurple']
        x=np.arange(len(models))
        width=0.25
        for i,(sc,color) in enumerate(zip(scenarios,colors)):
            sc_data=results_df[results_df['Scenario']==sc]
            r2_vals=[sc_data[sc_data['Model']==m]['R^2'].values[0] if len(sc_data[sc_data['Model']==m])>0 else 0 for m in models]
            bars=ax.bar(x+i*width,r2_vals,width,label=sc,color=color,alpha=0.85,edgecolor='black')
            for bar,val in zip(bars,r2_vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.3f}',ha='center',va='bottom',fontsize=7)

        ax.set_xlabel('Model',fontsize=12)
        ax.set_ylabel('R^2',fontsize=12)
        ax.set_title('R^2 Overview',fontsize=13,fontweight='bold')
        ax.set_xticks(x+width)
        ax.set_xticklabels(models,rotation=0)
        ax.legend()
        ax.grid(True,alpha=0.3,axis='y')
        ax.axhline(y=0,color='black',linestyle='-',linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/r2_comparison.png',dpi=300,bbox_inches='tight')
        plt.close()
    
    def _plot_r2_adj(self,results_df):
        fig,ax=plt.subplots(figsize=(14,6))
        models=list(self.model_config.keys())
        scenarios=['Samo Humani', 'Transfer Learning','Kombinovani dataset']
        colors=['orchid','teal','mediumpurple']
        x=np.arange(len(models))
        width=0.25
        for i,(sc,color) in enumerate(zip(scenarios,colors)):
            sc_data=results_df[results_df['Scenario']==sc]
            r2_adj_vals=[sc_data[sc_data['Model']==m]['R^2_adj'].values[0] if len(sc_data[sc_data['Model']==m])>0 else 0 for m in models]
            bars=ax.bar(x+i*width,r2_adj_vals,width,label=sc,color=color,alpha=0.85,edgecolor='black')
            for bar,val in zip(bars,r2_adj_vals):
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.3f}',ha='center',va='bottom',fontsize=7)
        ax.set_xlabel('Model',fontsize=12)
        ax.set_ylabel('R^2_adj',fontsize=12)
        ax.set_title('R^2_adj Overview',fontsize=13,fontweight='bold')
        ax.set_xticks(x+width)
        ax.set_xticklabels(models,rotation=0)
        ax.legend()
        ax.grid(True,alpha=0.3,axis='y')
        ax.axhline(y=0,color='black',linestyle='-',linewidth=0.8)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/r2_adj_comparison.png',dpi=300,bbox_inches='tight')
        plt.close()

    def _plot_best_model(self):
        scenarios={
            'scenario1':('Samo Humani','orchid'),
            'scenario2':('Transfer Learning','teal'),
            'scenario3':('Kombinovani dataset','mediumpurple')
            }
        fig,axes=plt.subplots(1,3,figsize=(18,5))
        for ax,(sc_key,(sc_label,color)) in zip(axes,scenarios.items()):
            if sc_key not in self.results:
                continue
            #trazimo najbolji model po rmse
            best_name=min(self.results[sc_key],key=lambda m: self.results[sc_key][m]['test']['rmse'])
            best_res=self.results[sc_key][best_name]
            y_pred=best_res['test']['y_pred']
            rmse=best_res['test']['rmse']
            r2=best_res['test']['r2']
            ax.scatter(self.Y_test,y_pred,alpha=0.6,s=30,color=color,edgecolors='k',linewidth=0.4)
            #idealna linija
            mn=min(self.Y_test.min(),y_pred.min())
            mx=max(self.Y_test.max(),y_pred.max())
            ax.plot([mn,mx],[mn,mx],'k--',linewidth=1.5,label='Optimalna vrednost')
            ax.set_xlabel('Eksperimentalni pKi',fontsize=11)
            ax.set_ylabel('Predvidjeni pKi',fontsize=11)
            ax.set_title(f'{sc_label}\n{best_name}',fontweight='bold',fontsize=11)
            ax.text(0.05,0.92,f'RMSE={rmse:.3f}\nR^2={r2:.3f}',transform=ax.transAxes,fontsize=10, bbox=dict(boxstyle='round',facecolor='white',alpha=0.8))
            ax.legend()
            ax.grid(True,alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/best_model.png',dpi=300,bbox_inches='tight')
        plt.close()
    
    def _plot_transfer_efficiency(self):
        #ako je TE <1.0 transfer pomaze
        #ako je TE>1.0 tranfer odmaze
        if 'scenario1' not in self.results or 'scenario2' not in self.results:
            return
        models=list(self.model_config.keys())
        te_values=[]
        for m in models:
            rmse_base=self.results['scenario1'][m]['test']['rmse']
            rmse_tl=self.results['scenario2'][m]['test']['rmse']
            te_values.append(rmse_tl/rmse_base)
        fig,ax=plt.subplots(figsize=(10,5))
        colors=['orchid' if te<1.0 else 'steelblue' for te in te_values]
        bars=ax.bar(models,te_values,color=colors,edgecolor='black',alpha=0.85)
        for bar,val in zip(bars,te_values):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{val:.3f}',ha='center',va='bottom',fontsize=9)
        ax.axhline(y=1.0,color='black',linestyle='--',linewidth=1.5,label='Baseline (TE=1.0)')
        ax.set_xlabel('Model',fontsize=12)
        ax.set_ylabel('Transfer Efficiency',fontsize=11)
        ax.set_title('Transfer Efficiency Overview',fontsize=12,fontweight='bold')
        ax.legend()
        ax.grid(True,alpha=0.3,axis='y')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/transfer_efficiency.png',dpi=300,bbox_inches='tight')
        plt.close()
    def _plot_residuals(self):
        scenarios = {
            'scenario1': ('Samo Humani','orchid'),
            'scenario2': ('Transfer Learning','teal'),
            'scenario3': ('Kombinovani dataset','mediumpurple')
        }
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Reziduali',fontsize=14, fontweight='bold', y=1.02)

        for ax, (sc_key, (sc_label, color)) in zip(axes, scenarios.items()):
            if sc_key not in self.results:
                continue
            best_name = min(self.results[sc_key],key=lambda m: self.results[sc_key][m]['test']['rmse'])
            best_res  = self.results[sc_key][best_name]
            y_pred    = best_res['test']['y_pred']
            residuals = self.Y_test - y_pred  
            rmse  = best_res['test']['rmse']
            r2    = best_res['test']['r2']
            r2adj = best_res['test']['r2_adj']
            ax.scatter(y_pred, residuals,alpha=0.6, s=30, color=color,edgecolors='k', linewidth=0.4)
            ax.axhline(y=0, color='black', linestyle='--',linewidth=1.5, label='optimalna vrednost')
            ax.axhline(y= rmse, color='deeppink', linestyle=':',linewidth=1, alpha=0.7, label=f'±RMSE ({rmse:.3f})')
            ax.axhline(y=-rmse, color='deeppink', linestyle=':',linewidth=1, alpha=0.7)
            ax.set_xlabel('Predviđeni pKi', fontsize=11)
            ax.set_ylabel('Rezidual', fontsize=11)
            ax.set_title(f'{sc_label}\n{best_name}', fontweight='bold', fontsize=11)
            ax.text(0.05, 0.95,f'RMSE={rmse:.3f}\nR²={r2:.3f}\nR²_adj={r2adj:.3f}',transform=ax.transAxes, fontsize=9, va='top',bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_directory}/residual_analysis.png',dpi=300, bbox_inches='tight')
        plt.close()
    def activate(self):
        self.load_data()
        self.split_data()
        self.train_scenario_1()
        self.train_scenario_2()
        self.train_scenario_3()
        results_df=self.compare_scenarios()
        print("*"*30)
        print("Finalni rezultati: ")
        for sc_key,sc_label in [('scenario1','Samo Humani'),('scenario2','Transfer Learning'),('scenario3','Kombinovani dataset')]:
            best=min(self.results[sc_key],key=lambda m: self.results[sc_key][m]['test']['rmse'])
            rmse=self.results[sc_key][best]['test']['rmse']
            r2=self.results[sc_key][best]['test']['r2']
            r2_adj=self.results[sc_key][best]['test']['r2_adj']
            print(f"{sc_label}  Najbolji:{best} (RMSE={rmse:.4f}, R^2={r2:.4f}, R^2 Adj={r2_adj:.4f})")
        return results_df
    
def main():
    trainer=ModelTrainer()
    results=trainer.activate()
    return trainer,results

if __name__=="__main__":
    trainer,results=main()
