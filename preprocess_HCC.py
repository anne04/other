import pandas as pd
import scanpy as sc
from collections import defaultdict
import numpy as np
import pickle
import gzip


patient_vs_samples = pd.read_csv('GSE206325_sample_annots_Liver_Treated_patients.csv')
patient_vs_samples_dict = defaultdict(list)
for i in range (0, len(patient_vs_samples)):
    patient_vs_samples_dict[patient_vs_samples['patient_ID'][i]].append([patient_vs_samples['sample_ID'][i], patient_vs_samples['tissue'][i]])

########################################################
adata = sc.read_h5ad("GSE206325_cell_vs_gene.h5ad")
cell_barcodes = adata.obs_names
# Map sample ids to matrix row
sample_to_rows = defaultdict(list)
i = 0
while i<len(cell_barcodes):
    sample_id = cell_barcodes[i].split('_')[0]
    print('sample_ID %d'%int(sample_id))
    count = 0
    start_row = i
    while i<len(cell_barcodes) and sample_id in cell_barcodes[i]:
        count = count + 1
        i = i+1
    
    end_row = i-1
    sample_to_rows[int(sample_id)] = [start_row, end_row, count]
#####################################################################

# See how many cells each patient has. Discard the patients with 0 cells in count matrix
# Also discard the samples with 0 cells in count matrix
patient_vs_cells = dict()
patient_vs_samples_dict_temp = defaultdict(list)
for patient in patient_vs_samples_dict:
    print('Patient %d'%patient)
    count = 0
    
    for sample in patient_vs_samples_dict[patient]:    
        sample_id = sample[0]
        if sample_id in sample_to_rows:
            count = count + sample_to_rows[sample_id][2]
            patient_vs_samples_dict_temp[patient].append(sample)

    patient_vs_cells[patient] = count

patient_vs_samples_dict = patient_vs_samples_dict_temp
#######################################################################
# Find the K folds
eligible_test_patient_ids = []
for patient in patient_vs_cells:
    if patient_vs_cells[patient] > 10000:
        eligible_test_patient_ids.append(patient)
        
# K = len(eligible_test_patient_ids)

#######################################################################
with gzip.open('HCC_metadata.pkl', 'wb') as fp:  
  pickle.dump([eligible_test_patient_ids,  patient_vs_cells, patient_vs_samples_dict, sample_to_rows], fp)


