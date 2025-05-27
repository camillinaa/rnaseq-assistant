import json
from sentence_transformers import SentenceTransformer, util

# INTENT CLASSIFIER

model = SentenceTransformer('all-MiniLM-L6-v2')

with open("/Users/camilla.callierotti/omics-agent-rnaseq/config/intents.json", "r") as f:
    intents = json.load(f)

def classify_query(user_query):
    query_embedding = model.encode(user_query, convert_to_tensor=True)
    intent_embeddings = model.encode(list(intents.values()), convert_to_tensor=True)
    scores = util.pytorch_cos_sim(query_embedding, intent_embeddings)
    best_match = list(intents.keys())[scores.argmax()]
    return best_match

q = "plot pvalue distribution of pathway enrichment results"

resp = classify_query(q)
print(f"Intent: {resp}")

# NAMED ENTITY RECOGNITION

import spacy
import scispacy

# Load the pre-trained SciSpacy model for biomedical NER
nlp = spacy.load("en_core_sci_sm")

# Example text
text = "what CNVs are identified in the cancer samples?"

# Process text
doc = nlp(q)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)


# DISPATCHER

if any("counts" in ent.text.lower() for ent in doc.ents):
    # Dispatch to counts analysis script
    print("Dispatching to counts analysis script...")
    # Example: call a function or script for counts analysis
    # counts_analysis.run()

elif any("deseq" in ent.text.lower() for ent in doc.ents):
    # Dispatch to DESeq analysis script
    print("Dispatching to DESeq analysis script...")
    # Example: call a function or script for DESeq analysis
    # deseq_analysis.run()

elif any("enrichr" in ent.text.lower() for ent in doc.ents):
    # Dispatch to Enrichr analysis script
    print("Dispatching to Enrichr analysis script...")
    # Example: call a function or script for Enrichr analysis
    # enrichr_analysis.run()

elif any("metadata" in ent.text.lower() for ent in doc.ents):
    # Dispatch to metadata analysis script
    print("Dispatching to metadata analysis script...")
    # Example: call a function or script for metadata analysis
    # metadata_analysis.run()

elif any("plot" in ent.text.lower() or "plots" in ent.text.lower() for ent in doc.ents):
    # Dispatch to plotting script
    print("Dispatching to plotting script...")
    # Example: call a function or script for plotting
    # plotting.run()

else:
    print("No matching workflow found. Please refine your query.")