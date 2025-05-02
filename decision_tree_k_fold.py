import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import scanpy as sc
from collections import defaultdict
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

filter_gene = 1

cluster_type_info = pd.read_csv('GSE206325_full_HCC_cluster_annotation.csv')
id_to_type = dict()
## "subgroup" and "type" combined
for i in range (0, len(cluster_type_info)):
    if cluster_type_info['type'][i] == 'DC':
        if cluster_type_info['subgroup'][i] == 'mregDC':
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['subgroup'][i]
        else:
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]
            
    elif cluster_type_info['type'][i] == 'CD8':
        if cluster_type_info['subgroup'][i] == '2-Dysfunctional-progenitor':
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['subgroup'][i]
        else:
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]
            
    elif cluster_type_info['type'][i] == 'CD4':
        if cluster_type_info['subgroup'][i] == '1-Tfh-like':
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['subgroup'][i]
        else:
            id_to_type[cluster_type_info['cluster_ID'][i]] = cluster_type_info['type'][i]
    else:
        id_to_type[cluster_type_info['cluster_ID'][i]] = 'other'


''' 
1-Tfh-like --- CD4 (out of these: 4-Effector-Memory, 5-Memory, 1-Tfh-like, 2-Th1 GZMK, 3-Th1/Th17)
2-Dysfunctional-progenitor --- CD8 (out of these: 7-Cytotoxic, 3-Dysfunctional-Proliferating, 4-Dysfunctional-effector, 1-Dysfunctional-terminal, 2-Dysfunctional-progenitor, 5-Effector, 6-Memory)
'''

adata = sc.read_h5ad("GSE206325_cell_vs_gene.h5ad")
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
    cell_barcodes = adata.obs_names
    gene_names = adata.var_names #adata[:, adata.var['highly_variable']].var_names
    y = list(adata.obs["cell_type"]) #adata.obs['leiden']                       # Your target labels
    
else:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)
    # Step 1: Use the raw gene expression matrix
    X = adata[:, adata.var['highly_variable']].X  # Use only highly variable genes
    cell_barcodes = adata.obs_names
    gene_names = adata[:, adata.var['highly_variable']].var_names
    y = list(adata.obs["cell_type"]) #adata.obs['leiden']                       # Your target labels


# X has all cells and y has all the labels
for i in range(0, len(y)):
    y[i] = id_to_type[y[i]]


# Read the metadata for K-fold cross-validation
with gzip.open('HCC_metadata.pkl', 'rb') as fp:  
  eligible_test_patient_ids,  patient_vs_cells, patient_vs_samples_dict, sample_to_rows = pickle.load(fp)

print("Prepare K folds")
K_folds = []
K_folds_size = []
i = 0
while i < len(eligible_test_patient_ids):
    count = 0
    subset = []
    while i < len(eligible_test_patient_ids) and count<=120000:
        patient = eligible_test_patient_ids[i]
        count = count + patient_vs_cells[patient]
        subset.append(patient)
        i = i + 1
        
    K_folds.append(subset)
    K_folds_size.append(count)
            
    
print("Run K=%d fold cross-validation"%len(K_folds))
k=0
for fold in K_folds:
    print("\n \n############## Fold K=%d #############"%k)
    test_rows = []
    for test_patient in fold:
        for sample in patient_vs_samples_dict[test_patient]:
            sample_id = sample[0]
            row_idx = sample_to_rows[sample_id][0]
            end_inx = sample_to_rows[sample_id][1]
            test_rows.append(row_idx)
            while row_idx < end_inx:
                row_idx = row_idx + 1
                test_rows.append(row_idx)
    
    # test_rows has the list of row entries from count matrix 'X' that has the test data
    X_test = X[test_rows, :]
    y_test = list(np.array(y)[test_rows])
    print("testing cells count %d"%len(test_rows))
    
    ## The rest of the patients (having cell count>0 in count matrix) will be used for training
    train_rows = []
    for patient in patient_vs_cells:
        if patient in fold or patient_vs_cells[patient]==0:
            continue
        for sample in patient_vs_samples_dict[patient]:
            sample_id = sample[0]
            row_idx = sample_to_rows[sample_id][0]
            end_inx = sample_to_rows[sample_id][1]
            train_rows.append(row_idx)
            while row_idx < end_inx:
                row_idx = row_idx + 1
                train_rows.append(row_idx)    

    X_train = X[train_rows, :]
    y_train = list(np.array(y)[train_rows])
    print("training cells count %d"%len(train_rows))
    
    


    ## Step 2: Train-test split (optional, for evaluation)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    
    # Step 3: Train a Decision Tree
    clf = DecisionTreeClassifier(max_depth=5, random_state=100) #, class_weight='balanced'
    clf.fit(X_train, y_train)
    ##
    ##################
    y_pred = clf.predict(X_test)
    
    # Get confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    
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
    
    
    
    
    
    # Step 4: Get feature (gene) importance
    importances = clf.feature_importances_
    # Step 5: Sort top marker genes
    gene_importance = dict()
    marker_df = pd.DataFrame({'gene': gene_names, 'importance': importances})
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
   
    gene_names = list(gene_names)
    selected_genes = []
    for i in unique_features:
        selected_genes.append([gene_names[i], gene_importance[gene_names[i]]])
    
    selected_genes = sorted(selected_genes, key = lambda x: x[1], reverse=True)
    print(f"Selected genes used in the tree:")
    for gene in selected_genes:
        print("%s: %g"%(gene[0], gene[1]))
    '''
    plt.clf()
    plt.figure(figsize=(20,20))
    tree.plot_tree(clf, feature_names=gene_names, class_names=clf.classes_, filled=True, max_depth=5)
    plt.savefig('tree_type_n_subgroup_filtered_fold'+ str(k) +'.svg')  
    '''
    k = k+1
###############

plt.clf()
plt.figure(figsize=(20,20))
tree.plot_tree(clf, feature_names=gene_names, class_names=clf.classes_, filled=True, max_depth=10)
#plt.savefig('tree_subgroup_filtered.svg')
#plt.savefig('tree_subgroup_nofilter.svg')


#plt.savefig('tree_type_n_subgroup_filtered.svg')   
plt.savefig('tree_type_n_subgroup_nofilter.svg')   
# patient_vs_samples = defaultdict(list) # patient_vs_samples[patient_id] = [[sample_id, type], [sample_id, type], [sample_id, type]] 
# Just check if each patient has at least one "tumor" sample. Also how many cells per patient. 
# for K fold: keep two patients for test, and all other for training
# for each fold, 
    #get the list of important markers then combine it - intersection
# intersect the set of imp genes, and report the average sensitivity and specificity
