import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import scanpy as sc
from collections import defaultdict
 
cluster_type_info = pd.read_csv('GSE206325_full_HCC_cluster_annotation.csv')
id_to_type = dict()
for i in range (0, len(cluster_type_info)):
    id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]
    


adata = sc.read_h5ad("sample824.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)


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
clf = DecisionTreeClassifier(max_depth=5, random_state=100)
clf.fit(X_train, y_train)

# Step 4: Get feature (gene) importance
importances = clf.feature_importances_
genes = adata[:, adata.var['highly_variable']].var_names

# Step 5: Sort top marker genes
marker_df = pd.DataFrame({'gene': genes, 'importance': importances})
top_markers = marker_df.sort_values('importance', ascending=False).head(20)

print(top_markers)

###############
from sklearn import tree
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=genes, class_names=clf.classes_, filled=True, max_depth=5)
plt.savefig('tree.png')
   
