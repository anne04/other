import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import scanpy as sc
from collections import defaultdict
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

cluster_type_info = pd.read_csv('GSE206325_full_HCC_cluster_annotation.csv')
id_to_type = dict()

# "type"

for i in range (0, len(cluster_type_info)):
    id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]

triad_types = ['CD4', 'CD8', 'DC']
for id in id_to_type.keys():
    if id_to_type[id] not in triad_types:
        id_to_type[id] = 'other'

'''
## "subgroup"
for i in range (0, len(cluster_type_info)):
    id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['subgroup'][i]
    
triad_types = ['1-Tfh-like', '2-Dysfunctional-progenitor', 'mregDC']
for id in id_to_type.keys():
    if id_to_type[id] not in triad_types:
        id_to_type[id] = 'other'
'''
'''
1-Tfh-like --- CD4 (out of these: 4-Effector-Memory, 5-Memory, 1-Tfh-like, 2-Th1 GZMK, 3-Th1/Th17)
2-Dysfunctional-progenitor --- CD8 (out of these: 7-Cytotoxic, 3-Dysfunctional-Proliferating, 4-Dysfunctional-effector, 1-Dysfunctional-terminal, 2-Dysfunctional-progenitor, 5-Effector, 6-Memory)
'''

filter_gene = 1

adata = sc.read_h5ad("GSE206325_samples20.h5ad")
gene_list_adata = list(adata.var_names)
if filter_gene == 1:
    # filter for cell surface protein
    uniprot = pd.read_csv('uniprotkb_cell_surface_AND_reviewed_tru_2025_04_28.tsv', sep='\t')
    gene_list = []
    for i in range(0, len(uniprot)):
        if isinstance(uniprot['Gene Names'][i], str) == False:
            continue
        items = uniprot['Gene Names'][i].split(' ')
        for gene in items:
            if gene in gene_list_adata:
                gene_list.append(gene)
    
    # keep only those genes
    adata = adata[:, gene_list]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    X = adata.X  
else:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    # Step 1: Use the raw gene expression matrix
    X = adata[:, adata.var['highly_variable']].X  # Use only highly variable genes


y = list(adata.obs["cell_type"]) #adata.obs['leiden']                       # Your target labels

for i in range(0, len(y)):
    y[i] = id_to_type[y[i]]

#########################################################################
# Assume X = gene expression matrix (cells x genes), y = class labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Predict
y_pred = rf_clf.predict(X_test)

# Confusion Matrix
classes = rf_clf.classes_
cm = confusion_matrix(y_test, y_pred, labels=classes)
print("Confusion Matrix:\n", cm)

# Sensitivity (Recall) per class
sensitivity_per_class = recall_score(y_test, y_pred, average=None, labels=classes)

# Specificity per class
specificity_per_class = []
for i in range(len(classes)):
    # For class i:
    # TP = cm[i, i]
    # FN = sum of row i excluding TP
    # FP = sum of column i excluding TP
    # TN = all else

    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # True Negatives
    fp = np.sum(np.delete(cm[:, i], i))                          # False Positives
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    specificity_per_class.append(specificity)

# Create DataFrame
metrics_df = pd.DataFrame({
    'Class': classes,
    'Sensitivity (Recall)': sensitivity_per_class,
    'Specificity': specificity_per_class
})

print("\nRandom Forest Performance Metrics:")
print(metrics_df)




# Step 4: Get feature (gene) importance
importances = clf.feature_importances_
genes = adata.var_names #adata[:, adata.var['highly_variable']].var_names

# Step 5: Sort top marker genes
gene_importance = dict()
marker_df = pd.DataFrame({'gene': genes, 'importance': importances})
top_markers = marker_df.sort_values('importance', ascending=False).head(20)
print(top_markers)

# iterate over marker_df to get the importance of each feature
for i in range(0, len(marker_df)):
    gene_importance[marker_df["gene"][i]] = marker_df["importance"][i]
##################
used_features = clf.tree_.feature
# Remove -2 entries (they mean "leaf node", no split)
used_features = used_features[used_features != -2]
# Get unique features
unique_features = np.unique(used_features)
print(f"Number of unique genes used: {len(unique_features)}")
#print(f"Indices of features used: {unique_features}")

gene_names = list(genes)
selected_genes = []
for i in unique_features:
    selected_genes.append([gene_names[i], gene_importance[gene_names[i]]])

selected_genes = sorted(selected_genes, key = lambda x: x[1], reverse=True)
print(f"Selected genes used in the tree:")
for gene in selected_genes:
    print("%s: %g"%(gene[0], gene[1]))

###############

plt.clf()
plt.figure(figsize=(20,20))
tree.plot_tree(clf, feature_names=genes, class_names=clf.classes_, filled=True, max_depth=10)
plt.savefig('tree_subgroup_filtered.svg')
plt.savefig('tree_subgroup_nofilter.svg')

# Train Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Get feature importances directly
importances = rf_clf.feature_importances_  # This is an array with importance for each feature (gene)

# Map back to gene names
gene_importance = {gene_names[i]: importance for i, importance in enumerate(importances)}

# Sort genes by importance
sorted_genes = sorted(gene_importance.items(), key=lambda x: x[1], reverse=True)

# Display
print("Genes sorted by Random Forest importance:")
for gene, importance in sorted_genes[:20]:  # Top 20 genes
    print(f"{gene}: {importance:.4f}")


# Initialize set to collect unique feature indices
unique_feature_indices = set()

# Loop through each decision tree in the forest
for estimator in rf_clf.estimators_:
    features = estimator.tree_.feature
    split_features = features[features != -2]  # -2 means it's a leaf
    unique_feature_indices.update(split_features)

# Convert to sorted list
unique_feature_indices = sorted(unique_feature_indices)

# Map to gene names
unique_genes = [gene_names[i] for i in unique_feature_indices]

# Output
print(f"Number of unique genes used in splits: {len(unique_genes)}")
print("Unique genes used:", unique_genes)

