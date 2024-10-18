import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import random
import torch
import argparse
import numpy as np
from datasets import Dataset, concatenate_datasets


# custom transformations you can import
from utils import *

# Set seed (you can turn this off)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

# # prevent annoying warnings -- I think it's only needed for notebooks
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# # Tokenize the input
def tokenize_function(tokenizer, examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Evaluation code
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Core training function -- you can modify this signature as needed if you want to add hyperparams
def do_train(model_name, train_set, eval_set, batch_size, epochs, lr, save_dir):

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Implement training; use the jupyter notebook we went over in class as a guide
    # You will have to tokenize data, create a model, and run training; it should be fairly short code

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenized_train = train_set.map(lambda x: tokenize_function(tokenizer, x), batched=True)
    tokenized_eval = eval_set.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    ##### YOUR CODE ENDS HERE ######

    print("Training completed...")
    trainer.save_model(save_dir+"/final")
    return


# Core evaluation function
# model_dir is the on-disk model you want to evaluate
# out_file is the file you will write results to
# you shouldn't have to change this
def do_eval(eval_set, model_dir, batch_size, out_file):

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)


    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    tokenized_eval = eval_set.map(lambda x: tokenize_function(tokenizer, x),  batched=True,load_from_cache_file=False)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    trainer.args.per_device_eval_batch_size=batch_size
    logits = trainer.predict(tokenized_eval)
    predictions = torch.argmax(torch.tensor(logits.predictions), dim=-1)
    labels = logits.label_ids
    score = logits.metrics["test_accuracy"]
    # write to output file
    for i in range(predictions.shape[0]):
        out_file.write(f"{predictions[i].item()} {labels[i].item()}\n")
    return score

# Created an augmented training dataset
def create_augmented_data(orig_dataset):
    aug_dataset = orig_dataset.copy()

    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Here, 'orig_dataset' is the original dataset.
    # You should return a dataset called 'aug_dataset' -- this
    # will be for the original training split augmented with 5k random transformed examples from the training set.

     # Get the size of the original training set
    train_size = len(orig_dataset['train'])

    # Generate 5000 random indices
    random_indices = random.sample(range(train_size), 5000)

    # Create lists to store augmented examples
    augmented_texts = []
    augmented_labels = []

    # Apply transformations to the selected examples
    for idx in random_indices:
        original_example = orig_dataset['train'][idx]
        transformed_example = custom_transform({'text': [original_example['text']]})
        
        # Check if the transformation was successful
        if transformed_example is not None and 'text' in transformed_example:
            augmented_texts.append(transformed_example['text'][0])
            augmented_labels.append(original_example['label'])

    # Get the original features
    original_features = orig_dataset['train'].features

    # Create a new dataset with the augmented examples, using the original features
    augmented_dataset = Dataset.from_dict(
        {'text': augmented_texts, 'label': augmented_labels},
        features=original_features
    )

    # Concatenate the original training set with the augmented dataset
    aug_dataset = concatenate_datasets([orig_dataset['train'], augmented_dataset])


    # Create a new dataset dictionary with the augmented training set
    return {
        'train': aug_dataset,
        'test': orig_dataset['test']
    }

    ##### YOUR CODE ENDS HERE ######

    return aug_dataset

    

# Create a dataset for the transformed test set
# you shouldn't have to change this but you can probably apply approaches from it elsewhere in your code
def create_transformed_dataset(dataset, debug_transformation):

    # Print 5 random transformed examples
    if debug_transformation:
        small_dataset = dataset.shuffle(seed=42).select(range(5))
        small_transformed_dataset = small_dataset.map(custom_transform, load_from_cache_file=False, batched=True)
        for k in range(5):
            print("Original Example ", str(k))
            print(small_dataset[k])
            print("\n")
            print("Transformed Example ", str(k))
            print(small_transformed_dataset[k])
            print('='*30)

        exit()

    transformed_dataset = dataset.map(custom_transform, load_from_cache_file=False, batched=True)
    return transformed_dataset

# you shouldn't have to change the main function unless you change other function signatures.

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--train", action="store_true", help="train a model on the training data")
    parser.add_argument("--dataset", type=str, default="imdb", help="huggingface dataset to use")
    parser.add_argument("--train_augmented", action="store_true", help="train a model on the augmented training data")
    parser.add_argument("--eval", action="store_true", help="evaluate model on the test set")
    parser.add_argument("--eval_transformed", action="store_true", help="evaluate model on the transformed test set")
    parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-cased", help="huggingface base model to use")
    parser.add_argument("--model_dir", type=str, default="./out")
    parser.add_argument("--eval_outfile", type=str, default="./outfile.txt")
    parser.add_argument("--debug_transformation", action="store_true", help="print a few transformed examples for debugging")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--small", action="store_true", help="use small dataset")

    args = parser.parse_args()


    # Load the dataset
    dataset = load_dataset(args.dataset)
    if args.small:
      print("Using small dataset")
      dataset["train"] = dataset["train"].shuffle(seed=42).select(range(4000))
      dataset["test"] = dataset["test"].shuffle(seed=42).select(range(1000))


    # Train model on the augmented training dataset
    if args.train_augmented:
        dataset = create_augmented_data(dataset)

    if args.train or args.train_augmented:
        do_train(args.model_name, dataset["train"], dataset["test"],
                 args.batch_size, args.num_epochs, args.learning_rate, args.model_dir)

    if args.eval_transformed:
        test_data = create_transformed_dataset(dataset["test"], args.debug_transformation)
    else:
        test_data = dataset["test"]
    # Evaluate the trained model on the original test dataset
    if args.eval or args.eval_transformed:
        out_file = open(args.eval_outfile, "w")
        score = do_eval(test_data, args.model_dir+"/final", args.batch_size, out_file)
        print("Score: ", score)
        out_file.close()


