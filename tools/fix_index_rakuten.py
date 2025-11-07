#!/usr/bin/env python3
"""
Script pour corriger l'index du fichier de prédictions Rakuten.

L'index doit commencer à 0 et se terminer à 13811.
"""

import pandas as pd

# Charger le fichier
df = pd.read_csv("test_predictions.csv", index_col=0)

print(f"Index actuel : de {df.index.min()} à {df.index.max()}")
print(f"Nombre de prédictions : {len(df)}")

# Réinitialiser l'index à 0-13811
df_corrected = df.reset_index(drop=True)

print(f"\nIndex corrigé : de {df_corrected.index.min()} à {df_corrected.index.max()}")

# Sauvegarder
df_corrected.to_csv("submission_rakuten.csv")

print(f"\n✓ Fichier corrigé sauvegardé : submission_rakuten.csv")
print(f"  Format : {df_corrected.shape}")
print(f"  Colonnes : {list(df_corrected.columns)}")

# Afficher un aperçu
print("\nAperçu du fichier corrigé :")
print(df_corrected.head(10))
print("...")
print(df_corrected.tail(5))

# Vérifier les classes
print(f"\nClasses prédites : {sorted(df_corrected['prdtypecode'].unique())}")
print(f"Nombre de classes uniques : {df_corrected['prdtypecode'].nunique()}")