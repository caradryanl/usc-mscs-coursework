import random
from datasets import load_dataset
from utils import custom_transform

def compare_original_and_transformed(num_examples=5):
    # Load the IMDB dataset
    dataset = load_dataset("imdb")

    # Get the test split
    test_data = dataset["test"]

    # Randomly select indices
    random_indices = random.sample(range(len(test_data)), num_examples)

    # Select examples and apply transformations
    for idx in random_indices:
        original_text = test_data[idx]["text"]
        original_label = "Positive" if test_data[idx]["label"] == 1 else "Negative"

        # Apply transformation
        transformed = custom_transform({"text": [original_text]})
        transformed_text = transformed["text"][0]

        # Print comparison
        print(f"Example {idx}")
        print(f"Original (Label: {original_label}):")
        print(original_text[:1000] + "..." if len(original_text) > 1000 else original_text)
        print("\nTransformed:")
        print(transformed_text[:1000] + "..." if len(transformed_text) > 1000 else transformed_text)
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    compare_original_and_transformed()