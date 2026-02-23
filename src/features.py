import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.ML.Descriptors import MoleculeDescriptors
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FeatureGenerator:
    def __init__(self):
        self.input_directory="../data/processed"
        self.output_directory="../data/features"
        os.makedirs(self.output_directory,exist_ok=True)
    
    def load_processed_data(self):
        #ucitavanje procesovanih podataka
        human_df=pd.read_csv(os.path.join(self.input_directory,"human_trypsin_processed.csv"),delimiter=';',encoding='utf-8')
        bovine_df=pd.read_csv(os.path.join(self.input_directory,"bovine_trypsin_processed.csv"),delimiter=';',encoding='utf-8')
        mix_df=pd.read_csv(os.path.join(self.input_directory,"mix_dataset.csv"),delimiter=';',encoding='utf-8')
        
        if 'source' not in human_df.columns:
            human_df['source']='human'
        if 'source' not in bovine_df.columns:
            bovine_df['source']='bovine'
        if 'source' not in mix_df.columns:
            mix_df['source']='mixed'
        
        return human_df,bovine_df,mix_df
    
    #generisanje 2d deskriptora za listu smiles stringova
    def generate_2d_desc(self,smiles_list):
        #lista svih dostupnih 2d deskriptora iz rdkit
        descriptor_names=[desc[0] for desc in Descriptors.descList]
        print(f"Dostupno je {len(descriptor_names)} rdkit deskriptora")

        desc_calculator=MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

        descriptors_list=[]
        valid_indices=[]
        valid_smiles=[]

        for index,smiles in enumerate(tqdm(smiles_list,desc="2D Deskriptori")):
            mol=Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    desc_values=desc_calculator.CalcDescriptors(mol)
                    descriptors_list.append(desc_values)
                    valid_indices.append(index)
                    valid_smiles.append(smiles)
                except:
                    continue #ako neki deskriptor ne radi, preskacem
        
        #kreiranje dataframe-a
        desc_df=pd.DataFrame(descriptors_list,columns=descriptor_names)
        desc_df.insert(0,'Smiles',valid_smiles)
        print(f"Uspesno generisano za {len(desc_df)}/{len(smiles_list)} molekula")
        print(f"Broj 2D deskriptora: {len(descriptor_names)}")
        return desc_df,valid_indices
    
    def generate_morgan_fingerprints(self,smiles_list,radius=2,n_bits=2048):
        fingerprints=[]
        valid_smiles=[]
        for smiles in tqdm(smiles_list,desc="Morganovi fingerprintovi"):
            mol=Chem.MolFromSmiles(smiles)
            if mol:
                try:
                    #generisanje morganovih fingerprintova
                    mfp=AllChem.GetMorganFingerprintAsBitVect(
                        mol,
                        radius=radius,
                        nBits=n_bits
                    )
                    #konverzija u listu bitova
                    mfp_list=list(mfp)
                    fingerprints.append(mfp_list)
                    valid_smiles.append(smiles)
                except:
                    continue
        #Kreiranje dataframe-a
        mfp_columns=[f'MFP_{i}' for i in range(n_bits)]
        mfp_df=pd.DataFrame(fingerprints,columns=mfp_columns)
        mfp_df.insert(0,'Smiles',valid_smiles)

        print(f"Uspesno generisano za {len(mfp_df)} molekula")
        print(f"Dimenzije morganovih fingerprintova: {n_bits} bita")
        return mfp_df
    
    #kombinovanje 2D deskriptora i fingerprintova u jednu feature matricu
    def combine_features(self,desc_df,mfp_df,original_df):
        features_df=pd.merge(desc_df,mfp_df,on='Smiles',how='inner')
        #dodavanje pKi vrednosti iz originalnog DataFrame-a
        result_df=pd.merge(features_df,original_df[['Smiles','pKi','source']],on='Smiles',how='left')
        #provera dimenzija
        n_samples=len(result_df)
        n_features=len(features_df.columns)-1 #minus smiles kolona

        print(f"Finalna feature matrica: {n_samples} x {n_features}")
        print(f"Format: N x {n_features}")
        return result_df
    
    #cuvanje generisanih features
    def save_features(self,human_features,bovine_features,mix_features):
        human_features.to_csv(f"{self.output_directory}/human_trypsin_features.csv",index=False)
        bovine_features.to_csv(f"{self.output_directory}/bovine_trypsin_features.csv",index=False)
        mix_features.to_csv(f"{self.output_directory}/mix_features.csv",index=False)

        print("Uspesno sacuvani features")
        print(f"humani: {len(human_features)} x {len(human_features.columns)-3}")
        print(f"kravlji: {len(bovine_features)} x {len(bovine_features.columns)-3}")
        print(f"mix: {len(mix_features)} x {len(mix_features.columns)-3}")

    def run(self):
        human_df,bovine_df,mix_df=self.load_processed_data()
        print("Obrada humanog tripsina: ")
        human_desc,human_index=self.generate_2d_desc(human_df['Smiles'].tolist())
        human_mfp=self.generate_morgan_fingerprints(human_df['Smiles'].tolist())
        human_features=self.combine_features(human_desc,human_mfp,human_df.iloc[human_index])

        print("*"*60)
        print("Obrada kravljeg tripsina: ")
        bovine_desc,bovine_index=self.generate_2d_desc(bovine_df['Smiles'].tolist())
        bovine_mfp=self.generate_morgan_fingerprints(bovine_df['Smiles'].tolist())
        bovine_features=self.combine_features(bovine_desc,bovine_mfp,bovine_df.iloc[bovine_index])

        print("*"*60)
        print("Obrada kombinvanog skupa: ")
        mix_desc,mix_index=self.generate_2d_desc(mix_df['Smiles'].tolist())
        mix_mfp=self.generate_morgan_fingerprints(mix_df['Smiles'].tolist())
        mix_features=self.combine_features(mix_desc,mix_mfp,mix_df.iloc[mix_index])

        self.save_features(human_features,bovine_features,mix_features)

        return human_features,bovine_features,mix_features
    def generate_single_molecule_features(smiles):
        mol=Chem.MolFromSmiles(smiles)
        if mol is None: #validacija 
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
        except Exception as e:
            print(f"Doslo je do greske pri generisanju features-a: {e}")
            return None
    def validate_smiles(smiles):
        if not smiles or not isinstance(smiles,str):
            return None,False,"SMILES string je prazan ili nije validan"
        try:
            mol=Chem.MolFromSmiles(smiles)
            if mol is None:
                return None,False,"RDKit ne moze parsirati uneti SMILES format"
            return mol,True,""
        except Exception as e:
            return None,False,f"Doslo je do greske: {e}"
                
if __name__=="__main__":
    try:
        feat_generator=FeatureGenerator()
        human_feat,bovine_feat,mix_feat=feat_generator.run()
    except Exception as e:
        print(f"Doslo je do greske {e}")
