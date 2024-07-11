from transformers import BertTokenizer, BertModel

if __name__ == "__main__":
    # Ensure you import the necessary modules at the beginning of your script
    from transformers import BertTokenizer, BertModel

    # Define the tokenizer and model paths
    tokenizer_path = 'bert-base-uncased'
    model_path = 'bert-base-uncased'

    # Attempt to load the tokenizer and model
    try:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        model = BertModel.from_pretrained(model_path)
        print("Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
