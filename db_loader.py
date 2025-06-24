import os
import pandas as pd
import re
from sqlalchemy import create_engine


class RNASeqDBLoader:
    def __init__(self, db_path="sqlite:///rnaseq.db", base_dir="data"):
        self.engine = create_engine(db_path)
        self.base_dir = base_dir
        self.files = {
            "dea": {},  # Differential expression analysis
            "dim_reduction": {},  # PCA/MDS
            "normalization": None,  # CPM matrix
            "metadata": None,  # Sample metadata
            "correlation": None  # Sample correlation
        }
        self._discover_files()

    def _sanitize_table_name(self, name):
        """
        Clean table names to ensure they're valid SQL identifiers.
        
        Args:
            name: Original table name
            
        Returns:
            Sanitized table name safe for SQL
        """
        # Replace problematic characters with underscores
        sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        # Ensure it doesn't start with a number
        if sanitized[0].isdigit():
            sanitized = f"table_{sanitized}"
        return sanitized

    def _discover_files(self):
        # Discover DEA files
        for root, dirs, _ in os.walk(self.base_dir):
            if root.startswith(os.path.join(self.base_dir, "dea_")):
                if os.path.dirname(root) == self.base_dir:  # only create subset if it is a top-level DEA directory
                    subset = os.path.basename(root).replace("dea_", "")
                    self.files["dea"][subset] = {}
                for dir_name in dirs:
                    if dir_name.startswith("dea_"):
                        comparison = dir_name.replace("dea_", "")
                        dea_path = os.path.join(root, dir_name, f"deseq2_toptable.{comparison}.txt")
                        gsea_path = os.path.join(root, dir_name, f"gsea.{comparison}.xlsx")
                        ora_path = os.path.join(root, dir_name, f"ora_CP.{comparison}.all.xlsx")

                        if os.path.exists(dea_path):
                            self.files["dea"][subset][comparison] = {
                                "deseq2": dea_path,
                                "gsea": gsea_path if os.path.exists(gsea_path) else None,
                                "ora": ora_path if os.path.exists(ora_path) else None
                            }

        # Discover dim_reduction files
        dim_reduction_dir = os.path.join(self.base_dir, "dim_reduction")
        if os.path.exists(dim_reduction_dir):
            pca_path = os.path.join(dim_reduction_dir, "PCA_scores.txt")
            mds_path = os.path.join(dim_reduction_dir, "MDS_scores.txt")
            self.files["dim_reduction"] = {
                "pca": pca_path if os.path.exists(pca_path) else None,
                "mds": mds_path if os.path.exists(mds_path) else None
            }

        # Discover normalization files
        normalization_dir = os.path.join(self.base_dir, "normalization")
        if os.path.exists(normalization_dir):
            cpm_path = os.path.join(normalization_dir, "cpm.txt")
            self.files["normalization"] = cpm_path if os.path.exists(cpm_path) else None

        # Discover metadata and correlation files
        metadata_path = os.path.join(self.base_dir, "metadata.csv")
        correlation_path = os.path.join(self.base_dir, "samples_correlation_table.txt")
        self.files["metadata"] = metadata_path if os.path.exists(metadata_path) else None
        self.files["correlation"] = correlation_path if os.path.exists(correlation_path) else None

    def _get_friendly_gsea_name(self, sheet_name):
    # Mapping raw GSEA sheet names to descriptive names
        gsea_name_map = {
            "c2.all.v2024.1.Hs.symbols": "curated_gene_sets",
            "c5.all.v2024.1.Hs.symbols": "go_gene_sets",
            "h.all.v2024.1.Hs.symbols": "hallmark_gene_sets"
        }
        return gsea_name_map.get(sheet_name, self._sanitize_table_name(sheet_name))

    def _read_and_upload(self, path, table_name):
            """
            Read a data file and upload it to the database.
            
            Args:
                path: File path to read
                table_name: Name of the database table to create
            """
            if path is None:
                return   
            try:
                # Handle different file formats
                if path.endswith((".txt", ".csv")):
                    df = pd.read_csv(path, sep=None, engine='python')  # Auto-detect separator
                    clean_table_name = self._sanitize_table_name(table_name)
                    print(f"Uploading {clean_table_name} from {path} ({len(df)} rows)")
                    df.to_sql(clean_table_name, self.engine, if_exists="replace", index=False)
                elif path.endswith(".xlsx"):
                    xls = pd.ExcelFile(path)
                    for sheet_name in xls.sheet_names:
                        df = xls.parse(sheet_name)
                        if "gsea" in path.lower() or "ora" in path.lower():
                            sheet_suffix = self._get_friendly_gsea_name(sheet_name)
                        else:
                            sheet_suffix = self._sanitize_table_name(sheet_name)
                        full_table_name = f"{self._sanitize_table_name(table_name)}_{sheet_suffix}"
                        print(f"Uploading {full_table_name} from sheet '{sheet_name}' in {path} ({len(df)} rows)")
                        df.to_sql(full_table_name, self.engine, if_exists="replace", index=False)
                else:
                    print(f"Skipping unsupported file format: {path}")
                    return
            except Exception as e:
                print(f"Error processing {path}: {e}")

    def _traverse_and_upload(self, data_dict, prefix=""):
        """
        Recursively traverse the file structure and upload files to database.
        
        Args:
            data_dict: Dictionary containing file paths
            prefix: Current prefix for table naming
        """
        for key, value in data_dict.items():
            new_prefix = f"{prefix}_{key}" if prefix else key
            
            if isinstance(value, dict):
                self._traverse_and_upload(value, new_prefix)
            elif isinstance(value, str):
                self._read_and_upload(value, new_prefix)

    def upload_files_to_db(self): 
        """
        Upload all discovered files to the database.
        This is the main method to call for database loading.
        """
        print("Starting database upload process...")
        self._traverse_and_upload(self.files)
        print("Database upload complete!")

    def get_db_info(self):
        from langchain_community.utilities import SQLDatabase
        db = SQLDatabase.from_uri("sqlite:///rnaseq.db")
        print(db.dialect)
        print(db.get_usable_table_names())
        try:
            return db.run("SELECT * FROM correlation LIMIT 10;")
        except Exception as e:
            print(f"Could not query correlation table: {e}")
            return None
    
    def run_all(self):
        self._discover_files()
        self.upload_files_to_db()

if __name__ == "__main__":
    loader = RNASeqDBLoader()
    loader.run_all()
