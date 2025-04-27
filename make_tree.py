import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import scanpy as sc
from collections import defaultdict
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np

cluster_type_info = pd.read_csv('GSE206325_full_HCC_cluster_annotation.csv')
id_to_type = dict()
for i in range (0, len(cluster_type_info)):
    id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]
    


adata = sc.read_h5ad("GSE206325_samples20.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)


# Step 1: Use the raw gene expression matrix
X = adata[:, adata.var['highly_variable']].X  # Use only highly variable genes
y = list(adata.obs["cell_type"]) #adata.obs['leiden']                       # Your target labels



for i in range(0, len(y)):
    y[i] = id_to_type[y[i]]

triad_types = ['CD4', 'CD8', 'DC']
for i in range(0, len(y)):
    if y[i] not in triad_types: 
        y[i] = 'other'


# Step 2: Train-test split (optional, for evaluation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# Step 3: Train a Decision Tree
clf = DecisionTreeClassifier(max_depth=10, random_state=100)
clf.fit(X_train, y_train)

# Step 4: Get feature (gene) importance
importances = clf.feature_importances_
genes = adata[:, adata.var['highly_variable']].var_names

# Step 5: Sort top marker genes
marker_df = pd.DataFrame({'gene': genes, 'importance': importances})
top_markers = marker_df.sort_values('importance', ascending=False).head(20)

print(top_markers)

##################
y_pred = clf.predict(X_test)
sensitivity_per_class = recall_score(y_test, y_pred, average=None)
print("Sensitivity (Recall) for each class:")
for label, sens in zip(clf.classes_, sensitivity_per_class):
    print(f"Class {label}: {sens:.2f}")

# Step 2: Calculate specificity for each class
specificity_per_class = []

for i in range(len(clf.classes_)):
    # True negatives: sum all cells except row i and column i
    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
    # False positives: sum column i, except the diagonal element
    fp = np.sum(np.delete(cm[:, i], i))
    
    specificity = tn / (tn + fp)
    specificity_per_class.append(specificity)

# Step 3: Display
print("\nSpecificity for each class:")
for label, spec in zip(clf.classes_, specificity_per_class):
    print(f"Class {label}: {spec:.2f}")



###############
from sklearn import tree
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=genes, class_names=clf.classes_, filled=True, max_depth=5)
plt.savefig('tree.png')
   
