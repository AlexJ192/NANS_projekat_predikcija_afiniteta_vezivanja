import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from io import BytesIO
from datetime import datetime
import base64
import shap
from rdkit import Chem
from rdkit.Chem import Draw,Descriptors,Crippen,Lipinski,AllChem,DataStructs
from rdkit.Chem.Lipinski import FractionCSP3
from rdkit.ML.Descriptors import MoleculeDescriptors
import joblib 
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Image as RLImage,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
st.set_page_config(
    page_title="Prediktor inhibitora tripsina",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stApp{
        background-color:#F5F2F9;
    }
    [data-testid="stSidebar"]{
        background-color:#E8DEF8;
    }
    .stButton>button{
        background-color:#B8A7D6;
        color:white;
        border-radius:10px;
        border:none;
        padding:0.5rem 1rem;
        font-weight:600;
    }
    .stButton>button:hover{
        background-color:#9B84C5;
    }
    h1,h2,h3{
        color:#4A3F5C;
    }
    .stSuccess{
        background-color:#D4F1D4;
    }
    .stInfo{
        background-color:#D4C5E8;
    }
    [data-testid="stMetricValue"]{
        color:#4A3F5C;
    }
    .dataframe{
        border:2px solid #D1C4E1;
    }
    .stProgress>div>div>div>div{
        background-color:#B8A7D6        
    }
</style>
""",unsafe_allow_html=True)

@st.cache_data
def generate_single_mol_feats(smiles):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        #2d deskriptora
        descriptor_names=[desc[0] for desc in Descriptors.descList]
        desc_calculator=MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        desc_vals=desc_calculator.CalcDescriptors(mol)
        #morganovi fingerprintovi
        mfp=AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048)
        mfp_list=list(mfp)
        #feat vektor
        all_feat_names=descriptor_names+[f'MFP_{i}' for i in range(2048)]
        all_feat_values=list(desc_vals)+mfp_list
        features=pd.Series(all_feat_values,index=all_feat_names)
        return features
    except:
        return None

def mol_to_image_bytes(mol,size=(400,400)):
    if mol is None:
        return None
    img=Draw.MolToImage(mol,size=size)
    buf=BytesIO()
    img.save(buf,format='PNG') 
    return buf.getvalue()

def calculate_descriptors(mol):
    if mol is None:
        return None
    return {
        'MW':Descriptors.MolWt(mol),
        'LogP':Crippen.MolLogP(mol),
        'HBD':Lipinski.NumHDonors(mol),
        'HBA':Lipinski.NumHAcceptors(mol),
        'QED':Descriptors.qed(mol),
        'TPSA':Descriptors.TPSA(mol),
        'NumAromaticRings':Descriptors.NumAromaticRings(mol),
        'NumRotatableBonds':Descriptors.NumRotatableBonds(mol),
        'FractionCsp3':FractionCSP3(mol),
    }
def analyze_inhibitor_quality(descriptors):
    #lipinski role of five
    reasons=[]
    score=0
    max_score=15
    #lipinski pravila
    if descriptors['MW']<=500:
        reasons.append(f"Molekulska masa({descriptors['MW']:.1f} Da)<=500")
        score+=1
    else:
        reasons.append(f"Molekulska masa({descriptors['MW']:.1f} Da)>500")

    if 0<=descriptors['LogP']<=5:
        reasons.append(f"Lipofilnost({descriptors['LogP']:.2f}) je u intervalu [0,5]")
        score+=1
    else:
        reasons.append(f"Lipofilnost({descriptors['LogP']:.2f}) se nalazi van optimalnog opsega")
    
    if descriptors['HBD']<=5:
        reasons.append(f"Broj vodonik vezujucih donora({descriptors['HBD']})<=5")
        score+=1
    else:
        reasons.append(f"Broj vodonik vezujucih donora({descriptors['HBD']})>5")

    if descriptors['HBA']<=10:
        reasons.append(f"Broj vodonik vezujucih akceptora({descriptors['HBA']})<=10")
        score+=1
    else:
        reasons.append(f"Broj vodonik vezujucih akceptora({descriptors['HBA']})>10")
    
    qed=descriptors['QED']
    if qed>0.7:
        reasons.append(f"QED({qed:.2f}), savrsen drug-likeness")
        score+=1
    elif qed>0.5:
        reasons.append(f"QED({qed:.2f}), dobar drug-likeness")
        score+=0.5
    else:
        reasons.append(f"QED({qed:.2f}), nizak drug-likeness")
    #bonus kriterijumi
    tpsa=descriptors['TPSA']
    if tpsa<=60:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})<=60, podrazumeva odličnu permeabilnost")
        score+=2
    elif tpsa<=90:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})<=90, podrazumeva dobru permeabilnost")
        score+=1.5
    elif tpsa<=140:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})<=140, podrazumeva umerenu permeabilnost")
        score+=1
    else:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})>140, poseduje slabu permeabilnost")
    
    if descriptors['NumAromaticRings']>=1: #tripsin zahteva bar 1 aromaticni prsten
        reasons.append(f"Broj aromaticnih prstenova: {descriptors['NumAromaticRings']}")
        score+=2
    else:
        reasons.append(f"Molekul ne poseduje minimalan broj aromaticnih prstenova({descriptors['NumAromaticRings']})")
    if descriptors['NumAromaticRings']>=2:
        reasons.append(f"Molekul sadrzi vise aromaticnih prstenova: {descriptors['NumAromaticRings']}")
        score+=1

    if descriptors['FractionCsp3']>=0.3:
        reasons.append(f"Optimalna 3D kompleksnost({descriptors['FractionCsp3']:.2f})")
        score+=2
    elif descriptors['FractionCsp3']>=0.1:
        reasons.append(f"Delimicno ravna struktura({descriptors['FractionCsp3']:.2f})")
        score+=1
    else:
        reasons.append(f"Potpuno ravna struktura({descriptors['FractionCsp3']:.2f})")

    if descriptors['MW']>=200:
        reasons.append(f"Molekul je dovoljno velik da popuni aktivni centar({descriptors['MW']}g/mol)")
        score+=2
    elif descriptors['MW']>=100:
        reasons.append(f"Molekul moze umereno da popuni aktivni centar({descriptors['MW']}g/mol)")
        score+=1
    else:
         reasons.append(f"Molekul je previse mali da popuni aktivni centar({descriptors['MW']}g/mol)")

    if descriptors['NumRotatableBonds']>=2:
        reasons.append(f"Optimalan broj rotacionih veza({descriptors['NumRotatableBonds']}), fleksibilnost je zadovoljena")
        score+=1
    else:
        reasons.append(f"Broj rotacionih veza je mali({descriptors['NumRotatableBonds']}), molekul je rigidan")
    
    percentage=(score/max_score)*100
    if percentage>=80:
        summary="Odlican kandidat za inhibitor"
        color="success" 
    elif percentage>=45:
        summary="Srednji kandidat za inhibitor (postoje manji nedostaci)"
        color='warning'
    else:
        summary="Los kandidat za inhibitor"
        color='error'
    return summary,reasons,percentage,color
    
def calculate_tanimoto(mol1,mol2):
    if mol1 is None or mol2 is None:
        return 0.0
    fp1=AllChem.GetMorganFingerprintAsBitVect(mol1, 2,nBits=2048)
    fp2=AllChem.GetMorganFingerprintAsBitVect(mol2, 2,nBits=2048)
    return DataStructs.TanimotoSimilarity(fp1,fp2)

@st.cache_resource
def load_models_data():
    models={}
    model_dir=Path("../models/trained")
    model_files={
        'Random Forest':'RandomForest_scenario1.pkl',
        'Gradient Boosting':'GradientBoosting_scenario1.pkl',
        'Lasso':'Lasso_scenario1.pkl',
        'Elastic Net':'ElasticNet_scenario1.pkl',
        'Ridge':'Ridge_scenario1.pkl',
        'Linear Regression':'LinearRegression_scenario1.pkl',
    }
    for name,filename in model_files.items():
        path=model_dir / filename
        if path.exists():
            models[name]=joblib.load(str(path))
    
    data_path=Path("../models/saved_data/data.pkl")
    data=None
    if data_path.exists():
        data=joblib.load(str(data_path))
    feat_names_path=Path("../models/human_selected_features.csv")
    feat_names=None
    if feat_names_path.exists():
        df_names=pd.read_csv(feat_names_path,sep=';')
        feat_names=df_names['feature_name'].tolist()
    return models,data,feat_names

@st.cache_data
def load_top1000_from_db():
    df=pd.read_csv("../data/features/human_trypsin_features.csv",sep=',')
    top5=df.nlargest(1000,'pKi')[['Smiles','pKi']].reset_index(drop=True)
    return top5

def predict_pKi(feature_series,model,data,model_name):
    try:
        feat_array=feature_series.values.reshape(1,-1)
        X_var=feat_array[:,data['variance_mask']]
        X_corr=X_var[:,data['correlation_mask']]
        X_selected=data['selector'].transform(X_corr)
        X_scaled=data['scaler'].transform(X_selected)
        X_pca=data['pca'].transform(X_scaled)
        if model_name=='Random Forest':
            tree_preds=np.array([tree.predict(X_pca)[0] for tree in model.estimators_])
            pKi_mean=tree_preds.mean()
            pKi_std=tree_preds.std()
        else:
            pKi_mean=model.predict(X_pca)[0]
            pKi_std=0.3
        return pKi_mean,pKi_std,X_pca
    except Exception as e:
        st.error(f"Doslo je do greske pri predikciji: {e}")
        return None,None,None

def compute_shap(X_pca,model,data,feature_names,model_name):
    try:
        if model_name !='Random Forest':
            return None
        explainer=shap.TreeExplainer(model)
        shap_values_pca=explainer.shap_values(X_pca)
        shap_values_og=shap_values_pca @ data['pca'].components_
        abs_shap=np.abs(shap_values_og[0])
        top3_indices=np.argsort(abs_shap)[-3:][::-1]
        top3_feat=[]
        for index in top3_indices:
            feat_name=feature_names[index]
            shap_val=shap_values_og[0,index]
            direction="povecava" if shap_val>0 else "smanjuje"
            if feat_name.startswith('MFP_'):
                tip='Morganov fingerprint'
            elif feat_name.startswith('fr_'):
                tip='Funkcionalna grupa'
            elif feat_name in ['qed','MolWt','LogP','TPSA']:
                tip='Fizicko-hemisjka osobina'
            elif 'VSA' in feat_name or 'EState' in feat_name:
                tip='Toploski deskriptor'
            else:
                tip='Molekulski deskriptor'
            top3_feat.append({'name':feat_name,'shap':abs(shap_val),'direction':direction,'type':tip})
        return top3_feat
    except Exception as e:
        print(f"Doslo je do greske prilikom racunjanja shap vrednosti: {e}")
        return None
                
def generate_txt_report(smiles,mol_descriptors,pKi_mean,pKi_std,summary,reasons,tanimoto_scores):
    report=f"""
{"*"*70}
Izvestaj predikcije afiniteta
{"*"*70}

SMILES: {smiles}


Molekulska masa: {mol_descriptors['MW']:.2f} g/mol
Lipofilnost: {mol_descriptors['LogP']:.2f}
Broj vodonik vezujucih donora: {mol_descriptors['HBD']}
Broj vodonik vezujucih akceptora: {mol_descriptors['HBA']}
QED(drug-likeness): {mol_descriptors['QED']:.3f}
Topoloski polarana povrsina: {mol_descriptors['TPSA']:.1f}
Broj aromaticnih prstenova: {mol_descriptors['NumAromaticRings']}
Broj rotacionih veza: {mol_descriptors['NumRotatableBonds']}
3D kompleksnost(Fsp3): {mol_descriptors['FractionCsp3']:.2f}

{"*"*70}
Analiza kvaliteta inhibitora
{"*"*70}
{summary}
{"*"*70}
Lipinski pravilo petice
{"*"*70}
"""
    for reason in reasons:
        report+=f"\n {reason}"
    
    report+=f"""
{"*"*70}
Predikcija afiniteta vezivanja
{"*"*70}

Predvidjen pKi: {pKi_mean:.2f} ± {pKi_std:.2f}
Ki: {10**(-pKi_mean)*1e9:.2f} nM

"""
    if pKi_mean>=8:
        report+="Odabrani inhibitor poseduje visok afinitet vezivanja i bice odlican inhibitor.\n"
    elif pKi_mean>=7:
        report+="Odabrani inhibitor poseduje dobar afinitet vezivanja i bice realtivno dobar inhibitor\n"
    elif pKi_mean>=6:
        report+="Odabrani inhibitor poseduje umeren afinitet vezivanja i moze biti potencijalni kandidat za inhibitor\n"
    else:
        report+="Odabrani inhibitor poseduje nizak afinitet vezivanja i najverovatnije nije dobar kadnidat za inhibitor\n"
    
    report+=f"""
{"*"*70}
Strukturna slicnost sa vec poznatim inhibitorima
{"*"*70}
Top 5 najslicnijih inhibitora iz baze podataka:
"""
    for i,(smiles_db,pki_db,tanimoto) in enumerate(tanimoto_scores,1):
        report+=f"\n {i} (Tanimoto {tanimoto:.3f}, pKi: {pki_db:.2f})"

    return report

def main():
    st.title("Prediktor afiniteta vezivanja inhibitora za humani tripsin")
    models,data,feature_names=load_models_data()
    top100_db=load_top1000_from_db()
    if 'smiles_input' not in st.session_state:
        st.session_state.smiles_input="" 
    if 'history' not in st.session_state:
        st.session_state.history=[]
    with st.sidebar:
        st.header("Podesavanja")
        model_name=st.selectbox("Izaberi model: ",list(models.keys()),index=0,help="Random forest je najbolji model (RMSE=0.58, R^2=0.90)")
        st.markdown("---")
        st.markdown("Informacije o modelu: ")
        model_info={
            'Random Forest':{'rmse':'0.58','r2':'0.90','r2_adj':'0.80','best':True},
            'Gradient Boosting':{'rmse':'0.61','r2':'0.89','r2_adj':'0.78','best':False},
            'Lasso':{'rmse':'0.77','r2':'0.83','r2_adj':'0.65','best':False},
            'Elastic Net':{'rmse':'0.77','r2':'0.83','r2_adj':'0.65','best':False},
            'Ridge':{'rmse':'0.78','r2':'0.83','r2_adj':'0.64','best':False},
            'Linear Regression':{'rmse':'0.78','r2':'0.83','r2_adj':'0.64','best':False},
        }
        info=model_info.get(model_name,{})
        if info.get('best'):
            st.success("Odabrali ste najbolji model")
        st.metric("RMSE",info.get('rmse','N/A'))
        st.metric("R^2",info.get('r2','N/A'))
        st.metric("R^2_adj",info.get('r2_adj','N/A'))
        st.markdown("---")
        st.markdown("Istorija pretrage")
        if st.session_state.history:
            for i,item in enumerate(reversed(st.session_state.history[-5:]),1):
                with st.expander(f"{i}. {item['smiles'][:20]}..."):
                    st.write(f"pKi: {item['pKi']:.2f}±{item['std']:.2f}")
                    st.write(f"Model: {item.get('model','N/A')}")
        else:
            st.info("Nema prethodnih pretraga")
    tab1,tab2=st.tabs(["Predikicja pKi","Analiza karakteristika"])
    with tab1:
        st.header("Unos molekula")
        col1,col2=st.columns([3,1])
        with col1:
            smiles_input=st.text_input("Unesite SMILES strukturu: ", value=st.session_state.smiles_input,key="smiles_widget",placeholder="npr.CC(C)Cc1ccc(cc1)C(C)C(=O)O",help="SMILES notacija molekula")
            st.session_state.smiles_input=smiles_input
        if st.session_state.smiles_input:
            mol=Chem.MolFromSmiles(st.session_state.smiles_input)
            if mol is None:
                st.error("Uneli ste neodgovarajući SMILES format. Pokušajte ponovo.")
            col1,col2=st.columns(2)
            with col1:
                st.subheader("Struktura molekula")
                img_bytes=mol_to_image_bytes(mol)
                st.image(img_bytes,width='stretch')
                if st.button("Predvidi pKi",type="primary",width='stretch'):
                    with st.spinner("Predikcija je u toku..."):
                        features=generate_single_mol_feats(smiles_input)
                        if features is None:
                            st.error("Došlo je do greške prilikom generisanja features-a")
                        else:
                            pKi_mean,pKi_std,X_pca=predict_pKi(features,models[model_name],data,model_name)
                            if pKi_mean is not None:
                                st.session_state.history.append({'smiles':smiles_input,'pKi':pKi_mean,'std':pKi_std,'model':model_name})
                                shap_feat=None
                                if model_name=='Random Forest' and feature_names:
                                    with st.spinner("SHAP izračunavanje u toku..."):
                                        shap_feat=compute_shap(X_pca,models[model_name],data,feature_names,model_name)
                                st.markdown("---")
                                st.subheader("Rezultati predikcije")
                                col_a,col_b,col_c=st.columns(3)
                                col_a.metric("Predvidjeni pKi ",f"{pKi_mean:.2f}")
                                col_b.metric("Interval poverenja: ",f"±{pKi_std:.2f}")
                                ki_nm=10**(-pKi_mean)*1e9
                                col_c.metric("Ki",f"{ki_nm:.1f} nM")
                                if shap_feat:
                                    st.markdown("---")
                                    st.subheader("SHAP analiza")
                                    st.info("Top 3 najuticajnije karakteristike")
                                    for i,feat in enumerate(shap_feat,1):
                                        with st.expander(f"{i}.{feat['name']} ({feat['type']})"):
                                            st.write(f"Uticaj: {feat['direction']} predviđeni pKi")
                                            st.write(f"SHAP vrednost: {feat['shap']:.4f}")
                                            st.progress(min(feat['shap']/0.2,1.0))
                                st.markdown("---")
                                st.subheader("Distribucija pKi")
                                fig,ax=plt.subplots(figsize=(8,4))
                                ax.hist(top100_db['pKi'],bins=20,alpha=0.6,color='#B8A7D6',edgecolor='black',label='Top 1000 inhibitora iz baze')
                                ax.axvline(pKi_mean,color="#690978",linewidth=3,linestyle='--',label=f'Korisnikov molekul ({pKi_mean:.2f})')
                                ax.set_xlabel('pKi')
                                ax.set_ylabel('Broj molekula')
                                ax.legend()
                                ax.grid(alpha=0.3)
                                st.pyplot(fig)
                                st.subheader("Strukturna sličnost(Top 5)")
                                tanimoto_scores=[]
                                for _,row in top100_db.iterrows():
                                    mol_db=Chem.MolFromSmiles(row['Smiles'])
                                    tanimoto=calculate_tanimoto(mol,mol_db)
                                    tanimoto_scores.append((row['Smiles'],row['pKi'],tanimoto))
                                tanimoto_scores.sort(key=lambda x: x[2],reverse=True)
                                df_tanimoto=pd.DataFrame(tanimoto_scores[:5],columns=['SMILES','pKi','Tanimoto'])
                                st.dataframe(df_tanimoto,width='stretch')
                                st.markdown("---")
                                col_exp1,col_exp2=st.columns(2)
                                with col_exp1:
                                    mol_desc=calculate_descriptors(mol)
                                    summary,reasons,perc,color_cat=analyze_inhibitor_quality(mol_desc)
                                    txt_report=generate_txt_report(smiles_input,mol_desc,pKi_mean,pKi_std,summary,reasons,tanimoto_scores)
                                    st.download_button("Preuzmi izveštaj(TXT)",txt_report,file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",mime="text/plain",width='stretch')
            with col2:
                st.subheader("Molekulske karakteristike")
                descriptors=calculate_descriptors(mol)
                st.metric("Molekulska masa",f"{descriptors['MW']:.1f}g/mol")
                st.metric("Lipofilnost",f"{descriptors['LogP']:.2f}")
                col_hb1,col_hb2=st.columns(2)
                col_hb1.metric("Broj vodonik vezujucih donora",descriptors['HBD'])
                col_hb2.metric("Broj vodonik vezujucih akceptora",descriptors['HBA'])
                st.metric("QED(drug-likeness",f"{descriptors['QED']:.3f}")
                st.metric("Topološki polarna površina",f"{descriptors['TPSA']:.1f}")
                st.metric("Broj aromatičnih prstenova",descriptors['NumAromaticRings'])
                summary,reasons,percentage,color_type=analyze_inhibitor_quality(descriptors)
                st.markdown("---") 
                st.write(f"{percentage:.1f}% zadovoljava Lipinski pravila")
                st.progress(percentage/100)
                if color_type=="success":
                    st.success(summary)
                elif color_type=="warning":
                    st.warning(summary)
                else:
                    st.error(summary)
                with st.expander("Detaljna analiza"):
                    for reason in reasons:
                        st.write(reason)
    with tab2:
        st.header("Analiza karakteristika")
        smiles_analysis=st.text_input("SMILES za analizu: ",key="analysis_smiles")
        if smiles_analysis:
            mol_a=Chem.MolFromSmiles(smiles_analysis)
            if mol_a:
                col_an1,col_an2=st.columns(2)
                with col_an1:
                    img_a=mol_to_image_bytes(mol_a)
                    st.image(img_a)
                with col_an2:
                    desc_a=calculate_descriptors(mol_a)
                    summary_a,reasons_a,perc_a,color_a=analyze_inhibitor_quality(desc_a)
                    st.subheader("Analiza kvaliteta potencijalnog inhibitora")
                    if color_a=="success":
                        st.success(summary_a)
                    elif color_a=="warning":
                        st.warning(summary_a)
                    else:
                        st.error(summary_a)
                    st.write(f"{perc_a:.1f}% zadovoljava Lipinski pravila")
                    st.progress(perc_a/100)
                    for reason in reasons_a:
                        st.write(reason)

if __name__=="__main__":
    main()
