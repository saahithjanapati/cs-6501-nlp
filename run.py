import random
from transformers import pipeline
import pandas as pd
from tqdm import tqdm


# Load the dataset
data = pd.read_csv("trim.csv")  # Replace with your file name
assert "title" in data.columns and "label" in data.columns

# Shuffle and split the dataset
random.seed(42)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
train_data = data.iloc[:3500]
test_data = data.iloc[4000:]


# Randomly select 5 examples for ICL
five_shot_examples = train_data.sample(n=5, random_state=40)

# Mapping function
def map_label_to_text(label):
    return "Fake" if label == 1 else "Real"

# Format the 5-shot prompt
def format_icl_prompt(five_shot, test_title):
    prompt = ""
    for _, row in five_shot.iterrows():
        label_text = map_label_to_text(row['label'])
        prompt += f"Title: {row['title']}\nLabel: {label_text}\n\n"
    prompt += f"Title: {test_title}\nLabel:"
    return prompt



# Run ICL on the test set
correct = 0
total = 0



for _, test_row in tqdm(test_data.iterrows()):
    test_prompt = format_icl_prompt(five_shot_examples, test_row['title'])
    # input_tokens = tokenizer.tokenizer(test_prompt)
    
    result = icl_model(test_prompt, max_new_tokens=4, do_sample=False)[0]['generated_text']
    
    
    # Extract the prediction and map it back to label
    predicted_label_text = result.split("Label:")[-1].strip().split()[0]
    predicted_label = 1 if predicted_label_text == "Fake" else 0
    if predicted_label == test_row['label']:
        correct += 1
    total += 1
    
    print(result, correct/total)

# Calculate accuracy
accuracy = correct / total
print(f"ICL Test Set Accuracy: {accuracy * 100:.2f}%")


with open('icl_acc_results.txt', 'w') as f:
    f.write(f"Accuracy: {correct/total}")

# format_icl_prompt(five_shot_examples, "")