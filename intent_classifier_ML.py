import json
import re
from sentence_transformers import SentenceTransformer, util
import torch
from typing import Dict, List, Tuple, Optional

class RNASeqQueryClassifier:
    def __init__(self, config_path: str, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.config_path = config_path
        self.intents = self._load_intents()
        self.intent_embeddings = None
        self._precompute_embeddings()
        
    def _load_intents(self) -> Dict:
        """Load intents configuration from JSON file"""
        with open(self.config_path, "r") as f:
            return json.load(f)
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all intent descriptions"""
        descriptions = []
        for intent_data in self.intents.values():
            if isinstance(intent_data, dict):
                # Use description field if available, otherwise use examples
                desc = intent_data.get('description', '')
                examples = intent_data.get('examples', [])
                if desc:
                    descriptions.append(desc)
                else:
                    descriptions.append(' '.join(examples))
            else:
                # Legacy format - just string descriptions
                descriptions.append(intent_data)
        
        self.intent_embeddings = self.model.encode(descriptions, convert_to_tensor=True)
    
    def classify_query(self, user_query: str, threshold: float = 0.5) -> Dict:
        """
        Classify user query to determine appropriate table type and components
        
        Args:
            user_query: The user's question
            threshold: Minimum similarity score for classification
            
        Returns:
            Dictionary containing classification results
        """
        query_embedding = self.model.encode(user_query, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(query_embedding, self.intent_embeddings)
        
        # Get best matches
        best_score, best_idx = torch.max(scores, dim=1)
        best_match = list(self.intents.keys())[best_idx.item()]
        
        # Get top 3 matches for context
        top_scores, top_indices = torch.topk(scores, k=min(3, len(self.intents)), dim=1)
        top_matches = [(list(self.intents.keys())[idx.item()], score.item()) 
                      for idx, score in zip(top_indices[0], top_scores[0])]
        
        # Extract query components
        components = self._extract_query_components(user_query)
        
        result = {
            'primary_intent': best_match,
            'confidence': best_score.item(),
            'top_matches': top_matches,
            'components': components,
            'table_type': self._determine_table_type(best_match),
            'requires_metadata': self._requires_metadata(best_match, user_query)
        }
        
        return result
    
    def _extract_query_components(self, query: str) -> Dict:
        """Extract key components from the query text"""
        components = {
            'sample_subset': None,
            'comparison_variable': None,
            'comparison_levels': None,
            'analysis_type': None,
            'gene_set': None,
            'specific_genes': [],
            'statistical_terms': []
        }
        
        # Extract analysis types
        analysis_patterns = {
            'deseq2': r'\b(deseq2?|differential\s+expression|de\s+analysis|deg)\b',
            'gsea': r'\b(gsea|gene\s+set\s+enrichment|pathway\s+analysis)\b',
            'ora': r'\b(ora|over\s*representation|enrichment\s+analysis)\b'
        }
        
        for analysis, pattern in analysis_patterns.items():
            if re.search(pattern, query.lower()):
                components['analysis_type'] = analysis
                break
        
        # Extract gene set references
        gene_set_patterns = {
            'hallmark': r'\b(hallmark|msigdb\s+hallmark)\b',
            'curated': r'\b(curated|c2|kegg|reactome|biocarta)\b',
            'go': r'\b(go|gene\s+ontology|biological\s+process|molecular\s+function)\b'
        }
        
        for gene_set, pattern in gene_set_patterns.items():
            if re.search(pattern, query.lower()):
                components['gene_set'] = gene_set
                break
        
        # Extract statistical terms
        stat_terms = re.findall(r'\b(p-?value|padj|fdr|fold\s+change|log2fc|significant)\b', query.lower())
        components['statistical_terms'] = list(set(stat_terms))
        
        # Extract potential gene names (capitalized words that could be genes)
        potential_genes = re.findall(r'\b[A-Z][A-Z0-9-]*\b', query)
        components['specific_genes'] = [gene for gene in potential_genes if len(gene) > 1]
        
        return components
    
    def _determine_table_type(self, intent: str) -> str:
        """Determine the primary table type based on intent"""
        table_mapping = {
            'differential_expression': 'dea',
            'gene_set_enrichment': 'dea',
            'pathway_analysis': 'dea',
            'sample_metadata': 'metadata',
            'gene_counts': 'normalization',
            'correlation_analysis': 'correlation',
            'sample_comparison': 'dea'
        }
        return table_mapping.get(intent, 'dea')
    
    def _requires_metadata(self, intent: str, query: str) -> bool:
        """Determine if metadata table is needed"""
        metadata_indicators = [
            'sample', 'condition', 'treatment', 'group', 'metadata',
            'patient', 'subject', 'clinical', 'phenotype'
        ]
        return any(indicator in query.lower() for indicator in metadata_indicators)
    
    def build_table_name(self, classification_result: Dict, available_tables: List[str]) -> Optional[str]:
        """
        Build the most likely table name based on classification and available tables
        
        Args:
            classification_result: Result from classify_query
            available_tables: List of available table names in the database
            
        Returns:
            Most likely table name or None if no good match
        """
        if classification_result['table_type'] != 'dea':
            return classification_result['table_type']
        
        # For DEA tables, we need to match the modular naming pattern
        components = classification_result['components']
        
        # Score each available DEA table
        dea_tables = [table for table in available_tables if table.startswith('dea_')]
        
        if not dea_tables:
            return None
        
        best_table = None
        best_score = 0
        
        for table in dea_tables:
            score = self._score_table_match(table, components, classification_result)
            if score > best_score:
                best_score = score
                best_table = table
        
        return best_table if best_score > 0.3 else None
    
    def _score_table_match(self, table_name: str, components: Dict, classification: Dict) -> float:
        """Score how well a table matches the query components"""
        parts = table_name.split('_')
        if len(parts) < 4:
            return 0
        
        score = 0
        total_weight = 0
        
        # Analysis type matching (high weight)
        if len(parts) >= 5:
            table_analysis = parts[4]
            if components['analysis_type'] == table_analysis:
                score += 0.4
            total_weight += 0.4
        
        # Gene set matching (medium weight)
        if len(parts) >= 6:
            table_gene_set = '_'.join(parts[5:])
            if components['gene_set'] and components['gene_set'] in table_gene_set:
                score += 0.3
            total_weight += 0.3
        
        # Sample subset matching (lower weight, harder to detect)
        if len(parts) >= 2:
            table_subset = parts[1]
            # This would require more sophisticated matching based on your metadata
            total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0


# Example usage and configuration
def create_example_config():
    """Create an example configuration file"""
    config = {
        "differential_expression": {
            "description": "Questions about differentially expressed genes, DESeq2 results, fold changes, and statistical significance",
            "examples": [
                "Which genes are differentially expressed between conditions?",
                "Show me genes with significant fold changes",
                "What are the top upregulated genes?",
                "Find genes with padj < 0.05"
            ]
        },
        "gene_set_enrichment": {
            "description": "Questions about pathway enrichment, GSEA results, and gene set analysis",
            "examples": [
                "What pathways are enriched?",
                "Show me GSEA results",
                "Which gene sets are significantly enriched?",
                "What biological processes are affected?"
            ]
        },
        "sample_metadata": {
            "description": "Questions about sample information, conditions, treatments, and metadata",
            "examples": [
                "What samples do we have?",
                "Show me sample conditions",
                "What treatments were used?",
                "Which samples are controls?"
            ]
        },
        "gene_counts": {
            "description": "Questions about gene expression counts, normalization, and raw data",
            "examples": [
                "Show me expression counts for a gene",
                "What is the expression level of GENE1?",
                "Get normalized counts",
                "Show raw expression data"
            ]
        },
        "correlation_analysis": {
            "description": "Questions about sample correlations and relationships between samples",
            "examples": [
                "How similar are the samples?",
                "Show me sample correlations",
                "Which samples cluster together?",
                "Are samples correlated?"
            ]
        }
    }
    
    return config

# Example usage
if __name__ == "__main__":
    # Create example config
    config = create_example_config()
    
    # Save config to file
    with open("bulk_rna_seq_intents.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize classifier
    classifier = RNASeqQueryClassifier("bulk_rna_seq_intents.json")
    
    # Example queries
    test_queries = [
        "Which genes are significantly upregulated in treatment vs control?",
        "Show me GSEA results for hallmark gene sets",
        "What samples do we have in the dataset?",
        "Get expression counts for TP53",
        "How correlated are the samples?"
    ]
    
    # Example available tables
    available_tables = [
        "dea_all_samples_treatment_control_deseq2",
        "dea_early_condition_yes_no_gsea_hallmark",
        "dea_late_flattening_yes_no_ora_curated_gene_sets",
        "metadata",
        "normalization",
        "correlation"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = classifier.classify_query(query)
        table_name = classifier.build_table_name(result, available_tables)
        
        print(f"Intent: {result['primary_intent']} (confidence: {result['confidence']:.3f})")
        print(f"Table type: {result['table_type']}")
        print(f"Suggested table: {table_name}")
        print(f"Components: {result['components']}")