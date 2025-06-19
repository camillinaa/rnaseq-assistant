import os
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("sqlite:///rnaseq.db")

base_dir = "data"

# Get file paths

# Initialize a dictionary to hold file paths
files = {
            "dea": {},  # Differential expression analysis
            "dim_reduction": {},  # PCA/MDS
            "normalization": None,  # CPM matrix
            "metadata": None,  # Sample metadata
            "correlation": None  # Sample correlation
        }

# Discover DEA files
for root, dirs, _ in os.walk(base_dir):
    if root.startswith(os.path.join(base_dir, "dea_")):
        if os.path.dirname(root) == base_dir: # only create subset if it is a top-level DEA directory
            subset = os.path.basename(root).replace("dea_", "")
            files["dea"][subset] = {}
        for dir_name in dirs:
            if dir_name.startswith("dea_"):
                comparison = dir_name.replace("dea_", "")
                dea_path = os.path.join(root, dir_name, f"deseq2_toptable.{comparison}.txt")
                gsea_path = os.path.join(root, dir_name, f"gsea.{comparison}.xlsx")
                ora_path = os.path.join(root, dir_name, f"ora_CP.{comparison}.all.xlsx")
                
                if os.path.exists(dea_path):
                    files["dea"][subset][comparison] = {
                        "deseq2": dea_path,
                        "gsea": gsea_path if os.path.exists(gsea_path) else None,
                        "ora": ora_path if os.path.exists(ora_path) else None
                    }

# Discover dim_reduction files
dim_reduction_dir = os.path.join(base_dir, "dim_reduction")
if os.path.exists(dim_reduction_dir):
    pca_path = os.path.join(dim_reduction_dir, "PCA_scores.txt")
    mds_path = os.path.join(dim_reduction_dir, "MDS_scores.txt")
    files["dim_reduction"] = {
        "pca": pca_path if os.path.exists(pca_path) else None,
        "mds": mds_path if os.path.exists(mds_path) else None
    }
            
# Discover normalization files
normalization_dir = os.path.join(base_dir, "normalization")
if os.path.exists(normalization_dir):
    cpm_path = os.path.join(normalization_dir, "cpm.txt")
    files["normalization"] = cpm_path if os.path.exists(cpm_path) else None

# Discover metadata and correlation files
metadata_path = os.path.join(base_dir, "metadata.csv")
correlation_path = os.path.join(base_dir, "samples_correlation_table.txt")
files["metadata"] = metadata_path if os.path.exists(metadata_path) else None
files["correlation"] = correlation_path if os.path.exists(correlation_path) else None

# Upload files in SQL database

def read_and_upload(path, table_name):
    if path is None:
        return
    try:
        if path.endswith(".txt") or path.endswith(".csv"):
            df = pd.read_csv(path, sep=None, engine='python')  # auto-detects separator
        elif path.endswith(".xlsx"):
            df = pd.read_excel(path)
        else:
            print(f"Skipping unsupported file format: {path}")
            return
        print(f"Uploading {table_name} from {path}")
        df.to_sql(table_name, engine, if_exists="replace", index=False)
    except Exception as e:
        print(f"Error processing {path}: {e}")

def traverse_and_upload(d, prefix=""):
    for key, value in d.items():
        new_prefix = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            traverse_and_upload(value, new_prefix)
        elif isinstance(value, str):
            table_name = new_prefix.replace("-", "_").replace(".", "_")
            read_and_upload(value, table_name)

traverse_and_upload(files)

# DB info

from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///rnaseq.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM correlation LIMIT 10;")