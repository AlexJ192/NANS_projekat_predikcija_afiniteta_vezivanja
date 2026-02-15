# analyze_mfp_bits.py
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from pathlib import Path
import os

class MFPAnalyzer:
    def __init__(self):
        """
        Inicijalizacija analizatora za Morgan fingerprint bitove
        """
        self.results_dir = "results/mfp_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Bitovi koje analiziramo (iz tvoje redukcije)
        self.target_bits = {
            145: "MFP_145 (najvažniji po F-testu)",
            656: "MFP_656 (drugi najvažniji MFP)",
            826: "MFP_826 (takođe važan)",
            1360: "MFP_1360 (četvrti MFP po važnosti)",
            82: "MFP_82 (peti MFP po važnosti)",
            597: "MFP_597",
            1952: "MFP_1952",
            736:"MFP_736"

        }
        
    def load_data(self):
        """
        Učitavanje podataka - koristimo ORIGINALNI feature fajl
        jer tu imamo sve bitove i SMILES
        """
        
        # Proveri da li fajlovi postoje
        human_path = "data/features/human_trypsin_features.csv"
        bovine_path = "data/features/bovine_trypsin_features.csv"
        
        if not os.path.exists(human_path):
            print(f"Fajl ne postoji: {human_path}")
            # Pokušaj sa relativnom putanjom
            human_path = "../data/features/human_trypsin_features.csv"
            bovine_path = "../data/features/bovine_trypsin_features.csv"
        
        human_features = pd.read_csv(human_path)
        bovine_features = pd.read_csv(bovine_path)
        
        print(f"  Human features: {human_features.shape}")
        print(f"  Bovine features: {bovine_features.shape}")
        
        return human_features, bovine_features
    
    def find_molecules_with_bit(self, df, bit_index):
        """
        Pronalazi molekule u dataframe-u gde je specificirani bit aktivan (vrednost = 1)
        """
        bit_column = f"MFP_{bit_index}"
        
        if bit_column not in df.columns:
            print(f"Kolona {bit_column} ne postoji!")
            return pd.DataFrame()
        
        # Pronađi redove gde je bit = 1
        active_df = df[df[bit_column] == 1].copy()
        
        print(f" Molekula sa aktivnim {bit_column}: {len(active_df)}")
        print(f"     Procentualno: {len(active_df)/len(df)*100:.1f}%")
        
        return active_df
    
    def visualize_molecules_with_bit(self, df, bit_index, n_examples=8):
        """
        Vizuelizuje N primera molekula koji imaju aktiviran dati bit
        """
        print(f"\n{'='*60}")
        print(f"ANALIZA ZA BIT MFP_{bit_index}")
        print(f"{'='*60}")
        
        # Pronađi molekule sa ovim bitom
        active_df = self.find_molecules_with_bit(df, bit_index)
        
        if len(active_df) == 0:
            print(f" Nema molekula sa aktivnim bitom MFP_{bit_index}")
            return None
        
        # Uzmi prvih N (ili manje ako nema dovoljno)
        n_to_show = min(n_examples, len(active_df))
        examples = active_df.head(n_to_show)
        
        # Pripremi molekule i legende
        molecules = []
        legends = []
        
        print(f"\n Prikazujem {n_to_show} primera:")
        
        for idx, row in examples.iterrows():
            smiles = row['Smiles']
            pKi = row['pKi']
            
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                molecules.append(mol)
                legends.append(f"pKi={pKi:.2f}")
                print(f"    {idx}: pKi={pKi:.2f}")
        
        # Kreiraj sliku sa molekulima
        if molecules:
            # Kreiraj grid sliku - returnPNG=True vraća bytes objekat
            img_bytes = Draw.MolsToGridImage(
                molecules,
                molsPerRow=4,
                subImgSize=(300, 300),
                legends=legends,
                returnPNG=True
            )
            
            # Sačuvaj sliku - img_bytes je direktno bytes objekat
            with open(f"{self.results_dir}/mfp_{bit_index}_molecules.png", 'wb') as f:
                f.write(img_bytes)  # Ne koristi .data, direktno img_bytes
            print(f"  Sačuvano: mfp_{bit_index}_molecules.png")
        
        # Statistička analiza
        self.analyze_bit_statistics(active_df, df, bit_index)
        
        return active_df
    
    def analyze_bit_statistics(self, active_df, full_df, bit_index):
        """
        Statistička analiza molekula sa i bez aktivnog bita
        """
        print(f"\n  STATISTIČKA ANALIZA ZA MFP_{bit_index}:")
        
        # pKi distribucija
        active_pki = active_df['pKi']
        inactive_df = full_df[full_df[f'MFP_{bit_index}'] == 0]
        inactive_pki = inactive_df['pKi']
        
        print(f"    pKi sa bitom:    mean={active_pki.mean():.2f}, std={active_pki.std():.2f}")
        print(f"    pKi bez bita:     mean={inactive_pki.mean():.2f}, std={inactive_pki.std():.2f}")
        
        # Da li bit pozitivno ili negativno utiče?
        diff = active_pki.mean() - inactive_pki.mean()
        if diff > 0.2:
            effect = "POZITIVAN (+)"
        elif diff < -0.2:
            effect = "NEGATIVAN (-)"
        else:
            effect = "NEUTRALAN"
        
        print(f"    Razlika u pKi: {diff:+.2f} ({effect})")
        
        # Kreiraj histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(inactive_pki, bins=30, alpha=0.5, label='Bez bita', color='gray', edgecolor='black')
        ax.hist(active_pki, bins=30, alpha=0.7, label='Sa bitom', color='purple', edgecolor='black')
        ax.set_xlabel('pKi')
        ax.set_ylabel('Broj molekula')
        ax.set_title(f'Distribucija pKi za MFP_{bit_index} - {self.target_bits[bit_index]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/mfp_{bit_index}_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f" Sačuvano: mfp_{bit_index}_distribution.png")
    
    def analyze_chemical_properties(self, df):
        """
        Analizira koja hemijska svojstva koreliraju sa važnim MFP bitovima
        """
        print(f"\n{'='*60}")
        print("KORELACIJA MFP BITOVA SA HEMIJSKIM DESKRIPTORIMA")
        print(f"{'='*60}")
        
        # Uzmi sve MFP kolone
        mfp_cols = [col for col in df.columns if col.startswith('MFP_')]
        
        # Uzmi hemijske deskriptore (sve osim MFP, Smiles, pKi, source)
        desc_cols = [col for col in df.columns 
                    if not col.startswith('MFP_') 
                    and col not in ['Smiles', 'pKi', 'source']]
        
        # Za naše bitove od interesa
        for bit in self.target_bits.keys():
            bit_col = f'MFP_{bit}'
            if bit_col not in df.columns:
                continue
                
            print(f"\n  {self.target_bits[bit]}:")
            
            # Izračunaj korelaciju sa svim deskriptorima
            correlations = []
            for desc in desc_cols[:20]:  # Prvih 20 deskriptora
                corr = df[bit_col].corr(df[desc])
                if abs(corr) > 0.2:  # Samo značajnije korelacije
                    correlations.append((desc, corr))
            
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            
            if correlations:
                print(f"    Najviše korelira sa:")
                for desc, corr in correlations[:5]:
                    print(f"      {desc}: {corr:.3f}")
            else:
                print(f"    Nema značajnih korelacija")
    
    def run(self):
        """
        Pokreće kompletnu analizu
        """
        print("="*60)
        print("ANALIZA MORGAN FINGERPRINT BITOVA")
        print("="*60)
        
        # Učitaj podatke
        human_df, bovine_df = self.load_data()
        
        # Analiziraj za human dataset
        print(f"\n{'='*60}")
        print(" HUMAN DATASET")
        print(f"{'='*60}")
        
        for bit in self.target_bits.keys():
            self.visualize_molecules_with_bit(human_df, bit, n_examples=8)
        
        # Analiziraj za bovine dataset
        print(f"\n{'='*60}")
        print(" BOVINE DATASET")
        print(f"{'='*60}")
        
        for bit in self.target_bits.keys():
            self.visualize_molecules_with_bit(bovine_df, bit, n_examples=8)
        
        # Hemijska analiza
        self.analyze_chemical_properties(human_df)
        
        print(f"\n{'='*60}")
        print(f" Analiza završena! Rezultati sačuvani u: {self.results_dir}")
        print(f"{'='*60}")


def main():
    """
    Glavna funkcija
    """
    analyzer = MFPAnalyzer()
    analyzer.run()
    
    print("\nGenerisani fajlovi:")
    print(f"  • Slike molekula: mfp_*_molecules.png")
    print(f"  • pKi distribucije: mfp_*_distribution.png")
    print(f"\n Otvori ove slike da vidiš šta MFP_145 i MFP_656 predstavljaju!")


if __name__ == "__main__":
    main()