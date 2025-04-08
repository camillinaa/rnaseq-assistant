# main.py
from nlu.parser import classify_intent, classify_query_type

def run_agent():
    print("🧬 OMICS Agent ready. Ask me anything about your processed data.\n")
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        domain, intent = classify_intent(query)
        qtype = classify_query_type(query)

        print(f"\nIntent: {intent} | Domain: {domain} | Query Type: {qtype}")
        print(f"🔍 [Placeholder] Responding about {intent} in {domain} using template for {qtype}-type questions.\n")

if __name__ == "__main__":
    run_agent()
