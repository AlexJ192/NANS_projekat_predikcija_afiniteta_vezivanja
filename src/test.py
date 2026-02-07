import pandas as pd

# Učitajte bilo koji od Vaših datasetova
df = pd.read_csv("../data/reduced/pca_reduced.csv", delimiter=';')
print("Dimenzije:", df.shape)
print("\nPrvih 2 redova:")
print(df.head(2).to_string())
print("\nNazivi kolona:")
print(df.columns.tolist())