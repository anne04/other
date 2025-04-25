import scipy.io
import pandas
import anndata


temp = scipy.io.mmread('GSE206325_gene_vs_cell.mtx')
temp = temp.tocsc()
cell_barcodes = pd.read_csv('cell_barcodes.csv', header=None) 
cell_barcodes = list(cell_barcodes[0])

gene_names = pd.read_csv('gene_names.csv', header=None) 
gene_names = list(gene_names[0])

cell_metadata = pd.read_csv('cell_metadata.csv') 

sample_id_list = [824, 839]
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
    adata.write('sample'+ str(sample_id) +'.h5ad', compression="gzip")


# https://anndata.readthedocs.io/en/latest/tutorials/notebooks/getting-started.html
