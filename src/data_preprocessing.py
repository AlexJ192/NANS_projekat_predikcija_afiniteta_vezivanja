import pandas as pd
import numpy as np
import os

class DataPreparation:
    def __init__(self):
        self.human_path="../data/raw/human_trypsin.csv"
        self.bovine_path="../data/raw/bovine_trypsin.csv"
        self.output_directory="../data/processed"

    def load_data(self): #ucitavanje raw csv fajlova
        if not os.path.exists(self.human_path):
            raise FileNotFoundError(f"Podaci za humani tripsin nisu pronadjeni: {self.human_path}")
        if not os.path.exists(self.bovine_path):
            raise FileNotFoundError(f"Podaci za kravlji tripsin nisu pronadjeni: {self.bovine_path}")

        human_df=pd.read_csv(self.human_path,delimiter=';',encoding='utf-8')
        bovine_df=pd.read_csv(self.bovine_path,delimiter=';',encoding='utf-8')
        return human_df,bovine_df

    def process_data(self,df,dataset_name):
        if 'Standard Type' not in df.columns:
            raise ValueError(f"Fali glavna Ki kolona u {dataset_name}")
        ki_df=df[df['Standard Type']=='Ki'].copy()
        #ukalnjanje redova koji ne sadrze SMILES format ili vrednost merenja
        starting_len=len(ki_df)
        ki_df=ki_df.dropna(subset=['Smiles','Standard Value'])
        removed=starting_len-len(ki_df)
        if removed > 0:
            print(f'Uspesno je uklonjeno {removed} redova')

        ki_df['Standard Value']=pd.to_numeric(ki_df['Standard Value'],errors='coerce')
        ki_df=ki_df.dropna(subset=['Standard Value'])

        ki_df=ki_df[ki_df['Standard Value']>0.000000001] #uklanjanje svih Ki=0

        #filtriranje realnih vrednosti merenja, sve van unetog opsega predstavlja zasigurnu gresku u merenju
        ki_df=ki_df[(ki_df['Standard Value'] > 0.001) & (ki_df['Standard Value']<1e9)]

        #koverzija Ki u pKi u nm
        #pKi = -log10(Ki*1e-9)
        ki_df['pKi']=-np.log10(ki_df['Standard Value']*1e-9)

        if 'pChEMBL Value' in ki_df.columns:
            ki_df['pChEMBL Value']=pd.to_numeric(ki_df['pChEMBL Value'],errors='coerce')
            has_pchembl=~ki_df['pChEMBL Value'].isna()
            ki_df.loc[has_pchembl,'pKi']=ki_df.loc[has_pchembl,'pChEMBL Value']
        
        ki_df=ki_df[ki_df['pKi']<=12] #ako je pKi vece od 12 to se smatra greskom
        #uklanjaje SMILES duplikata
        start_count=len(ki_df)
        ki_df=ki_df.drop_duplicates(subset=['Smiles'],keep='first')
        removed=start_count-len(ki_df)
        if removed>0:
           print(f"Uspesno uklonjeno {removed} duplikata")
        
        print(f"Finalni dataset: {len(ki_df)}")
        print(f"Ki opseg: {ki_df['Standard Value'].min():.10f} - {ki_df['Standard Value'].max():.10f} nM")
        print(f"pKi opseg: {ki_df['pKi'].min():.2f} - {ki_df['pKi'].max():.2f}")
        return ki_df
    
    def save_data(self,human_processed,bovine_processed):
        human_data_processed=human_processed.copy()
        bovine_data_processed=bovine_processed.copy()
        
        #kombinovani skup za transfer learning
        human_data_processed['source']='human'
        bovine_data_processed['source']='bovine'
        human_data_processed.to_csv(f"{self.output_directory}/human_trypsin_processed.csv",index=False,sep=';')
        bovine_data_processed.to_csv(f"{self.output_directory}/bovine_trypsin_processed.csv",index=False,sep=';')

        mix=pd.concat([human_data_processed,bovine_data_processed],ignore_index=True)
        mix.to_csv(f"{self.output_directory}/mix_dataset.csv",index=False,sep=';')

        print("Obradjeni podaci su uspesno sacuvani")
        print(f"Humani: {len(human_processed)}")
        print(f"Kravlji: {len(bovine_processed)}")
        print(f"Kombinovani: {len(mix)}")

        return mix
    
    def run(self):
        human_data,bovine_data=self.load_data()
        human_processed=self.process_data(human_data,"Humani tripsin")
        bovine_processed=self.process_data(bovine_data,"Kravlji tripsin")
        mix=self.save_data(human_processed,bovine_processed)
        return human_processed,bovine_processed
    
if __name__=="__main__":
        processor=DataPreparation()
        human_data,bovine_data=processor.run()
