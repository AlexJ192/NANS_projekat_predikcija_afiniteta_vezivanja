import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sea
from io import BytesIO
from datetime import datetime
import base64
from rdkit import Chem
from rdkit.Chem import Draw,Descriptors,Crippen,Lipinski,AllChem,DataStructs
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
        'FractionCsp3':Descriptors.FractionCsp3(mol),
    }
def analyze_inhibitor_quality(descriptors):
    #lipinski role of five
    reasons=[]
    score=0
    max_score=8
    if descriptors['MW']<=500:
        reasons.append(f"Molekulska masa({descriptors['MW']:.1f} Da)<=500")
        score+=1
    else:
        reasons.append(f"Molekulska masa({descriptors['MW']:.1f} Da)>500")

    if descriptors['logP']<=5:
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
    
    if descriptors['TPSA']<=140:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})<=140")
        score+=1
    else:
        reasons.append(f"Topoloski polarna povrsina({descriptors['TPSA']:.1f})>140, poseduje slabu permeabilnost")
    
    if descriptors['NumAromaticRings']<=3:
        reasons.append(f"Optimalan broj aromaticnih prstenova({descriptors['NumAromaticRings']})")
        score+=1
    else:
        reasons.append(f"Suboptimalan broj aromaticnih prstenova({descriptors['NumAromaticRings']})")

    if descriptors['FractionCsp3']>=0.25:
        reasons.append(f"Optimalna 3D kompleksnost({descriptors['FractionCsp3']:.2f})")
        score+=1
    else:
        reasons.append(f"Ravna struktura({descriptors['FractionCsp3']:.2f})")

    percentage=(score/max_score)*100
    if percentage>=75:
        summary="Odlican kandidat za inhibitor"
        color="success"
    elif percentage>=50:
        summary="Dobar kandidat za inhibitor (postoje manji nedostaci)"
        color='warning'
    else:
        summary="Los kandidat za inhibitor"
        color='error'
    return summary,reasons,percentage,color
    
def calculate_tanimoto(mol1,mol2):
    if mol1 is None or mol2 is None:
        return 0.0
    fp1=AllChem.GetMorganFingerprintAsBitVector(mol1, 2,nBits=2048)
    fp2=AllChem.GetMorganFingerprintAsBitVector(mol2, 2,nBits=2048)
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
    
    return models,data

@st.cache_data
def load_top5_from_db():
    df=pd.read_csv("../data/features/human_trypsin_features.csv",sep=';')
    top5=df.nlargest(5,'pKi')[['Smiles','pKi']].reset_index(drop=True)
    return top5

def predict_pKi(feature_series,model,data,model_name):
    try:
        feat_array=feature_series.values.reshape(1,-1)
        X_var=feat_array[:,data['variance_mask']]
        X_corr=feat_array[:,data['correlation_mask']]
        X_scaled=data['scaler'].transform(X_corr)
        X_selected=data['selector'].transform(X_scaled)
        X_pca=data['pca'].transform(X_selected)
        if model_name=='Random Forest':
            tree_preds=np.array([tree.predict(X_pca)[0] for tree in model.estimators_])
            pKi_mean=tree_preds.mean()
            pKi_std=tree_preds.std()
        else:
            pKi_mean=model.predict(X_pca)[0]
            pKi_std=0.3
        return pKi_mean,pKi_std
    except Exception as e:
        st.error(f"Doslo je do greske pri predikciji: {e}")
        return None,None
    
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

