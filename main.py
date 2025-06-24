import os
from db_loader import RNASeqDBLoader
from chatbot import RNASeqChatbot


def demonstrate_plotting_capabilities():
    """
    Demonstrate the plotting capabilities with example queries that showcase
    different types of visualizations commonly used in RNA-seq analysis.
    """
    
    print("🧬 RNA-seq Chatbot with Intelligent Plotting - Demo")
    print("=" * 60)
    
    # Initialize chatbot
    bot = RNASeqChatbot(
        model_name="mistral-large-latest", 
        api_key=os.getenv("MISTRAL_API_KEY"), 
        db_uri="sqlite:///rnaseq.db", 
        dialect="sqlite", 
        top_k=10,
    )
    
    # Example queries that demonstrate different plot types
    example_queries = [
        {
            "question": "Show me the most significantly differentially expressed genes",
            "expected_plot": "volcano plot",
            "description": "This should generate a volcano plot showing differential expression results"
        },
        {
            "question": "What does the PCA analysis show about sample clustering?",
            "expected_plot": "scatter plot",
            "description": "This should create a PCA scatter plot with sample groupings"
        },
        {
            "question": "Show me the correlation between samples",
            "expected_plot": "heatmap",
            "description": "This should generate a correlation heatmap"
        },
        {
            "question": "How many genes are upregulated vs downregulated?",
            "expected_plot": "bar chart",
            "description": "This should create a bar chart showing count comparisons"
        }
    ]
    
    print("\n🚀 Running demonstration queries...")
    print("Each query will show both textual analysis and appropriate visualizations\n")
    
    for i, example in enumerate(example_queries, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: {example['description']}")
        print(f"Expected visualization: {example['expected_plot']}")
        print(f"{'='*60}")
        
        # Run the query with the enhanced chatbot
        bot.chat(example["question"])
        
        # Option to save plots
        save_choice = input(f"\n💾 Save this plot? (y/n): ").lower().strip()
        if save_choice == 'y':
            filename = f"demo_plot_{i}_{example['expected_plot'].replace(' ', '_')}"
            bot.save_plot(example["question"], filename)
        
        # Pause between examples
        if i < len(example_queries):
            input("\n⏸️  Press Enter to continue to next example...")


def interactive_mode():
    """
    Interactive mode where users can ask their own questions and see
    both textual responses and visualizations.
    """
    
    print("\n🔬 Interactive RNA-seq Analysis Mode")
    print("Ask questions about your RNA-seq data and get intelligent visualizations!")
    print("Type 'quit' to exit, 'demo' to run demonstrations, or 'help' for tips.\n")
    
    bot = RNASeqChatbot(
        model_name="mistral-large-latest", 
        api_key=os.getenv("MISTRAL_API_KEY"), 
        db_uri="sqlite:///rnaseq.db", 
        dialect="sqlite", 
        top_k=10,
    )
    
    while True:
        try:
            user_input = input("\n🧬 Your question: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for using the RNA-seq chatbot!")
                break
            
            elif user_input.lower() == 'demo':
                demonstrate_plotting_capabilities()
                continue
            
            elif user_input.lower() == 'help':
                print_help_message()
                continue
            
            elif not user_input:
                continue
            
            # Process the user's question
            bot.display_results(user_input)
            
            # Ask if they want to save any plots
            save_choice = input("\n💾 Save plot to file? (y/n): ").lower().strip()
            if save_choice == 'y':
                format_choice = input("📊 Format (html/png/pdf/svg) [default: html]: ").lower().strip()
                if not format_choice:
                    format_choice = "html"
                
                filename = bot.save_plot(user_input, format=format_choice)
                if filename:
                    print(f"✅ Plot saved as: {filename}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question or check your database connection.")


def print_help_message():
    """Display helpful tips for using the RNA-seq chatbot effectively."""
    
    help_text = """
    📚 RNA-seq Chatbot Help & Tips
    
    🎯 Effective Question Patterns:
    ├── Differential Expression: "Show me significantly upregulated genes"
    ├── Data Exploration: "What does the PCA tell us about sample similarity?"
    ├── Specific Comparisons: "Compare treatment vs control in liver samples"
    ├── Statistical Summaries: "How many genes have p-adj < 0.05?"
    └── Pathway Analysis: "Show me enriched pathways in upregulated genes"
    
    📊 Visualization Types:
    ├── Volcano Plot: For differential expression results
    ├── Scatter Plot: For PCA/MDS and correlation analyses  
    ├── Heatmap: For expression matrices and correlation data
    ├── Bar Chart: For counts and categorical comparisons
    └── Table View: For detailed data inspection
    
    💡 Pro Tips:
    ├── Be specific about sample groups or conditions
    ├── Ask for significance thresholds (e.g., "p-adj < 0.01")
    ├── Mention fold change criteria for better filtering
    ├── Use biological context in your questions
    └── Try both broad and specific questions for different insights
    
    🔧 Commands:
    ├── 'demo' - Run demonstration queries
    ├── 'help' - Show this help message
    └── 'quit' - Exit the chatbot
    """
    print(help_text)


def main():
    """
    Main function that sets up the database and provides options for
    different modes of interaction with the enhanced chatbot.
    """
    
    print("🧬 Enhanced RNA-seq Chatbot - Setup")
    print("=" * 50)
    
    # Check if database exists, if not, load it
    db_path = "rnaseq.db"
    if not os.path.exists(db_path):
        print("📊 Database not found. Loading RNA-seq data...")
        try:
            loader = RNASeqDBLoader(db_path=f"sqlite:///{db_path}", base_dir="data")
            loader.upload_files_to_db()
            print("✅ Database loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading database: {e}")
            print("Please ensure your 'data' directory contains RNA-seq analysis files.")
            return
    else:
        print("✅ Database found and ready!")
    
    # Provide options for different interaction modes
    print("\n🚀 Choose interaction mode:")
    print("1. Interactive Q&A mode (recommended)")
    print("2. Run plotting demonstrations")
    print("3. Single query mode")
    
    try:
        choice = input("\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            interactive_mode()
        
        elif choice == "2":
            demonstrate_plotting_capabilities()
        
        elif choice == "3":
            # Single query mode - like your original script
            user_query = input("\n🧬 Enter your query: ")
            bot = RNASeqChatbot(
                model_name="mistral-large-latest", 
                api_key=os.getenv("MISTRAL_API_KEY"), 
                db_uri="sqlite:///rnaseq.db", 
                dialect="sqlite", 
                top_k=10,
            )
            
            results = bot.chat(user_query)
            print(f"\n📝 Response: {results['answer']}")
            
            if results.get("plot_available"):
                print(f"\n📊 Plot generated: {results.get('plot_type')}")
                save_choice = input("💾 Save plot? (y/n): ").lower().strip()
                if save_choice == 'y':
                    bot.save_plot(user_query)
        
        else:
            print("❌ Invalid choice. Please run the script again.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()