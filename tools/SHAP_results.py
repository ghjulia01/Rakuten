import json
import pandas as pd
import matplotlib.pyplot as plt

# Charger les resultats
with open('results/metrics/shap_decomposed.json') as f:
    shap_results = json.load(f)

# 1. Importance globale
global_imp = pd.Series(shap_results['global_importance']).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
global_imp.plot(kind='barh')
plt.title('Importance globale par composante')
plt.xlabel('Importance SHAP moyenne')
plt.tight_layout()
plt.savefig('results/shap_global_importance.png')

# 2. Importance par classe
per_class = pd.DataFrame(shap_results['per_class_importance']).T

# Top 3 composantes pour chaque classe
for class_id in per_class.index:
    top3 = per_class.loc[class_id].sort_values(ascending=False).head(3)
    print(f"\nClasse {class_id}:")
    for comp, val in top3.items():
        print(f"  {comp:20s}: {val:.4f}")

# 3. Heatmap
plt.figure(figsize=(12, 10))
plt.imshow(per_class.values, aspect='auto', cmap='YlOrRd')
plt.colorbar(label='Importance SHAP')
plt.yticks(range(len(per_class)), per_class.index)
plt.xticks(range(len(per_class.columns)), per_class.columns, rotation=45, ha='right')
plt.title('Importance SHAP par classe et composante')
plt.tight_layout()
plt.savefig('results/shap_heatmap.png')