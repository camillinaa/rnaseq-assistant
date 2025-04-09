import spacy
import scispacy
from scispacy.abbreviation import AbbreviationDetector

# Step 1: Load the metadata CSV
# metadata_path = "/Users/camilla.callierotti/omics-agent/data/bulk_rna_seq/glio/metadata.csv"
# metadata_df = pd.read_csv(metadata_path, sep=",", index_col=0)

# Load the pre-trained SciSpacy model for biomedical NER
nlp = spacy.load("en_core_sci_sm")

# Example text
text = "what CNVs are identified in the cancer samples?"

# Process text
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)


# query = nlp("How many samples are in the differentiated condition?")

