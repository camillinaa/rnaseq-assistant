import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Intent:
    analysis_type: str
    confidence: float
    biological_terms: List[str]
    metadata_column: Optional[str] = None
    sample_subset: Optional[str] = None

class RNASeqIntentClassifier:
    def __init__(self, metadata_values: Dict[str, List[str]] = None):
        """
        Initialize the classifier with metadata values.
        
        Args:
            metadata_values: Dictionary mapping column names to their unique values
                           e.g., {'Type': ['E-GSC', 'L-GSC', 'NS', 'differentiated'],
                                  'Patient': ['R008', 'R012', 'R015'],
                                  'Flattening': ['Yes', 'No'],
                                  'Batch': ['b1', 'b2']}
        """
        self.metadata_values = metadata_values or {}
        
        # Define patterns for each analysis type
        self.analysis_patterns = {
            'gsea': [
                r'\bpathways?\b',
                r'\benrich\w*\b',
                r'\bNES\b',
                r'\bgene.sets?\b',
                r'\bgsea\b',
                r'\bfunctional.analysis\b',
                r'\bover.?representation\b'
            ],
            'deseq2': [
                r'\bDEGs?\b',
                r'\bdifferential\w*\s+express\w*\b',
                r'\bfold.change\b',
                r'\blog2FC\b',
                r'\bdeseq2?\b',
                r'\bup.?regulat\w*\b',
                r'\bdown.?regulat\w*\b',
                r'\bsignificant\w*\s+genes?\b'
            ],
            'ora': [
                r'\bover.?representation\b',
                r'\bORA\b',
                r'\bhypergeometric\b',
                r'\bfisher.?exact\b'
            ],
            'correlation': [
                r'\bcorrelat\w*\b',
                r'\bpearson\b',
                r'\bspearman\b',
                r'\bco.?express\w*\b'
            ],
            'counts': [
                r'\bcounts?\b',
                r'\bnormali[zs]\w*\b',
                r'\bexpression.levels?\b',
                r'\bTPM\b',
                r'\bFPKM\b',
                r'\bCPM\b'
            ],
            'metadata': [
                r'\bsample\w*\b',
                r'\bmetadata\b',
                r'\bpatient\w*\b',
                r'\bbatch\w*\b',
                r'\bdemographic\w*\b'
            ]
        }
        
        # Sample subset patterns
        self.subset_patterns = [
            r'\bin\s+(\w+)\s+sample\w*\b',
            r'\b(\w+)\s+sample\w*\s+only\b',
            r'\bsubset\s+to\s+(\w+)\b'
        ]

    def classify_intent(self, question: str) -> Intent:
        """Main classification function"""
        question_lower = question.lower()
        
        # 1. Classify analysis type
        analysis_type, confidence = self._classify_analysis_type(question_lower)
        
        # 2. Extract biological terms
        biological_terms = self._extract_biological_terms(question)
        
        # 3. Extract sample subset if mentioned
        sample_subset = self._extract_sample_subset(question)
        
        # 4. Infer metadata column based on biological terms
        metadata_column = self._infer_metadata_column(biological_terms)
        
        return Intent(
            analysis_type=analysis_type,
            confidence=confidence,
            biological_terms=biological_terms,
            metadata_column=metadata_column,
            sample_subset=sample_subset
        )
    
    def _classify_analysis_type(self, question: str) -> Tuple[str, float]:
        """Classify the analysis type with confidence score"""
        scores = {}
        
        for analysis_type, patterns in self.analysis_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, question, re.IGNORECASE))
                score += matches
            scores[analysis_type] = score
        
        if not any(scores.values()):
            return 'unknown', 0.0
        
        # Get best match
        best_analysis = max(scores, key=scores.get)
        max_score = scores[best_analysis]
        total_matches = sum(scores.values())
        
        confidence = max_score / total_matches if total_matches > 0 else 0.0
        
        return best_analysis, confidence
    
    def _extract_biological_terms(self, question: str) -> List[str]:
        """Extract biological terms from the question by matching against metadata values"""
        terms = []
        
        if not self.metadata_values:
            return terms
        
        question_lower = question.lower()
        
        # Check each metadata column's values
        for column, values in self.metadata_values.items():
            for value in values:
                # Create case-insensitive regex pattern for the value
                # Handle special characters in the value
                escaped_value = re.escape(value)
                pattern = r'\b' + escaped_value + r'\b'
                
                # Check if this value appears in the question
                if re.search(pattern, question, re.IGNORECASE):
                    terms.append(value)
        
        # Also look for comparison patterns to catch terms that might be mentioned together
        comparison_patterns = [
            r'\bbetween\s+([^,\s]+(?:\s+[^,\s]+)*)\s+and\s+([^,\s]+(?:\s+[^,\s]+)*)\b',
            r'\b([^,\s]+(?:\s+[^,\s]+)*)\s+vs\s+([^,\s]+(?:\s+[^,\s]+)*)\b',
            r'\b([^,\s]+(?:\s+[^,\s]+)*)\s+versus\s+([^,\s]+(?:\s+[^,\s]+)*)\b'
        ]
        
        for pattern in comparison_patterns:
            matches = re.findall(pattern, question, re.IGNORECASE)
            for match in matches:
                for term in match:
                    # Check if this term matches any metadata value
                    term_clean = term.strip()
                    if self._is_metadata_value(term_clean):
                        terms.append(term_clean)
        
        return list(set(terms))  # Remove duplicates
    
    def _is_metadata_value(self, term: str) -> bool:
        """Check if a term is a valid metadata value"""
        if not self.metadata_values:
            return False
        
        for values in self.metadata_values.values():
            if term in values or term.lower() in [v.lower() for v in values]:
                return True
        return False
    
    def _extract_sample_subset(self, question: str) -> Optional[str]:
        """Extract sample subset if mentioned"""
        for pattern in self.subset_patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _infer_metadata_column(self, biological_terms: List[str]) -> Optional[str]:
        """Infer which metadata column based on biological terms found in metadata"""
        if not biological_terms or not self.metadata_values:
            return None
        
        # Check which metadata column contains the most biological terms
        column_scores = {}
        
        for column, values in self.metadata_values.items():
            score = 0
            for term in biological_terms:
                # Case-insensitive check
                if term in values or term.lower() in [v.lower() for v in values]:
                    score += 1
            column_scores[column] = score
        
        # Return the column with the highest score
        if column_scores and max(column_scores.values()) > 0:
            return max(column_scores, key=column_scores.get)
        
        return None

    def generate_table_name(self, intent: Intent) -> str:
        """Generate the expected table name based on intent"""
        if intent.analysis_type in ['gsea', 'deseq2', 'ora']:
            # Sample subset
            subset = intent.sample_subset if intent.sample_subset else 'all_samples'
            
            # Metadata column and values
            if intent.metadata_column and len(intent.biological_terms) >= 2:
                # Convert biological terms to table format
                converted_terms = [term.replace('-', '_').replace(' ', '_') 
                                 for term in intent.biological_terms[:2]]
                
                # Construct table name
                table_name = f"dea_{subset}_{intent.metadata_column}_{converted_terms[0]}_{converted_terms[1]}_{intent.analysis_type}"
                
                # Add gene set suffix for GSEA/ORA
                if intent.analysis_type in ['gsea', 'ora']:
                    table_name += "_curated_gene_sets"
                
                return table_name
        
        return f"{intent.analysis_type}_table"

# Usage example
def main():
    # Example metadata values (you would get these from your actual metadata table)
    metadata_values = {
        'Type': ['E-GSC', 'L-GSC', 'NS', 'differentiated'],
        'Patient': ['R008', 'R012', 'R015'],
        'Flattening': ['Yes', 'No'],
        'Batch': ['b1', 'b2', 'b3']
    }
    
    classifier = RNASeqIntentClassifier(metadata_values)
    
    # Test questions
    test_questions = [
        "What are the top enriched pathways (NES > 1) from GSEA between E-GSC and L-GSC?",
        "Show me differentially expressed genes between E-GSC and L-GSC with fold change > 2",
        "What are the correlation values between samples s17 and s20?",
        "Get normalized counts for GAPDH gene",
        "Compare flattening Yes vs No samples",
        "Show me DEGs in differentiated samples"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        intent = classifier.classify_intent(question)
        print(f"Intent: {intent}")
        print(f"Expected table: {classifier.generate_table_name(intent)}")
        print("-" * 50)

# Helper function to get metadata values from your database
def get_metadata_values_from_db(db_connection) -> Dict[str, List[str]]:
    """
    Extract unique values from each metadata column
    
    Args:
        db_connection: Your database connection
        
    Returns:
        Dictionary mapping column names to their unique values
    """
    metadata_values = {}
    
    # Get all columns from metadata table (excluding Sample column)
    columns_query = "SELECT column_name FROM information_schema.columns WHERE table_name = 'metadata' AND column_name != 'Sample'"
    columns = [row[0] for row in db_connection.execute(columns_query).fetchall()]
    
    # Get unique values for each column
    for column in columns:
        values_query = f"SELECT DISTINCT {column} FROM metadata WHERE {column} IS NOT NULL"
        values = [row[0] for row in db_connection.execute(values_query).fetchall()]
        metadata_values[column] = values
    
    return metadata_values

if __name__ == "__main__":
    main()