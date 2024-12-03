import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import Dataset

# Load the fine-tuned model and tokenizer
model_dir = "./results"  # Path to the saved model directory
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.eval()

# Load the dataset (test set should be a CSV file)
test_file = "test_data.csv"  # Path to the test data CSV
test_data = pd.read_csv(test_file)
test_dataset = Dataset.from_pandas(test_data)

# Preprocess the test data
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Extract [CLS] representations
def extract_cls_representations(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    cls_representations = []
    labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: batch[key].to(model.device) for key in ["input_ids", "attention_mask"]}
            outputs = model(**inputs, output_hidden_states=True)
            cls_tokens = outputs.hidden_states[-1][:, 0, :]  # Extract [CLS] representation
            cls_representations.append(cls_tokens.cpu().numpy())
            labels.extend(batch["label"].numpy())

    cls_representations = np.concatenate(cls_representations, axis=0)
    return cls_representations, labels

# Reduce dimensions and plot
def plot_pca(embeddings, labels):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.title("PCA of [CLS] Representations")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(loc="best", title="Label")
    plt.show()

def plot_tsne(embeddings, labels, perplexity=30):
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=reduced[:, 0], y=reduced[:, 1], hue=labels, palette="tab10", alpha=0.7)
    plt.title("t-SNE of [CLS] Representations")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc="best", title="Label")
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Extracting [CLS] representations...")
    cls_representations, cls_labels = extract_cls_representations(model, test_dataset)
    
    print("Visualizing with PCA...")
    plot_pca(cls_representations, cls_labels)
    
    print("Visualizing with t-SNE...")
    plot_tsne(cls_representations, cls_labels)