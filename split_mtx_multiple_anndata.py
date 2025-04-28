import scipy.io
import pandas as pd
import anndata
import numpy as np


temp = scipy.io.mmread('GSE206325_gene_vs_cell.mtx')
cell_barcodes = pd.read_csv('cell_barcodes.csv', header=None) 
cell_barcodes = list(cell_barcodes[0])
gene_names = pd.read_csv('gene_names.csv', header=None) 
gene_names = list(gene_names[0])
cell_metadata = pd.read_csv('cell_metadata.csv') 

cell_barcode_dict = dict()
for barcode in cell_barcodes:
    cell_barcode_dict[int(barcode.split('_')[0])] = 1


patient_metadata = pd.read_csv('GSE206325_sample_annots_Liver_Treated_patients.csv') 
tumor_R = []
tumor_NR = []
for i in range(0, len(patient_metadata)):
    if patient_metadata['sample_ID'][i] not in cell_barcode_dict:
        continue
    if patient_metadata['tissue'][i] == 'Tumor':# and patient_metadata['Immune_Infiltration'][i] == 'High':
        if patient_metadata['treatment_Resp'][i] == 'anti-PD1_NR':
            tumor_NR.append(patient_metadata['sample_ID'][i])
        elif patient_metadata['treatment_Resp'][i] == 'anti-PD1_R': 
            tumor_R.append(patient_metadata['sample_ID'][i])



tumor_R = tumor_R[0:10]
tumor_NR = tumor_NR[0:10]


temp = temp.tocsc()
count_matrix = temp
count_matrix = np.transpose(count_matrix)
adata = anndata.AnnData(count_matrix)
adata.obs_names = cell_barcodes 
adata.var_names = gene_names
adata.obs["cell_type"] = list(cell_metadata['cell_to_cluster'])
adata.write('GSE206325_cell_vs_gene.h5ad', compression="gzip")

#sample_id_list = [824, 839]

anndata_list = []
sample_id_list = tumor_R + tumor_NR
total_cells = 0
for sample_id in sample_id_list:
    start_column = 0
    while str(sample_id) not in cell_barcodes[start_column]:
        start_column = start_column + 1
    
    end_column = start_column
    while str(sample_id) in cell_barcodes[end_column]:
        end_column = end_column + 1
    
    end_column = end_column - 1
    count_matrix = temp[:,start_column:end_column+1]
    count_matrix = np.transpose(count_matrix)
    adata = anndata.AnnData(count_matrix)
    adata.obs_names = cell_barcodes[start_column:end_column+1]
    adata.var_names = gene_names
    adata.obs["cell_type"] = list(cell_metadata['cell_to_cluster'])[start_column:end_column+1]
    anndata_list.append(adata)
    total_cells = total_cells + (end_column-start_column+1)
    #adata.write('sample'+ str(sample_id) +'.h5ad', compression="gzip")
    print('total cells %d'%total_cells)

print('total cells %d'%total_cells)
adata = anndata.concat(anndata_list)
adata.write('samples20.h5ad', compression="gzip")

# https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html
