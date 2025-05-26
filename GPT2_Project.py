```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
import seaborn as sns
from collections import Counter
import os

# Set random seed for reproducibility
torch.manual_seed(42)

# 1. Model Definition
class GPT2Model(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072, dropout=0.1):
        super(GPT2Model, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self.create_positional_encoding(max_len=128, d_model=d_model)
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, mask=None):
        # [Placeholder: Implement forward pass with embedding, positional encoding, transformer layers, and output]
        batch_size, seq_len = x.shape
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :].to(x.device)
        for layer in self.transformer_layers:
            x = layer(x, mask)
        return self.output_layer(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # [Placeholder: Implement attention, residual connections, and feed-forward]
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        # [Placeholder: Implement scaled dot-product attention]
        batch_size = q.size(0)
        q = self.q_linear(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.out_linear(context)

# 2. Dataset Preprocessing
class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoding = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        return input_ids[:-1], input_ids[1:]

def load_dataset_tinystories():
    dataset = load_dataset('roneneldan/TinyStories')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    train_data = dataset['train']
    val_data = dataset['validation']
    return TinyStoriesDataset(train_data, tokenizer), TinyStoriesDataset(val_data, tokenizer)

# 3. Visualization Functions
def plot_word_cloud(data):
    # [Placeholder: Implement word cloud generation]
    # Example: Use Counter to count tokens and generate word cloud
    texts = [item['text'] for item in data]
    tokens = ' '.join(texts).split()
    word_counts = Counter(tokens)
    # Generate word cloud using wordcloud package and save as 'word_cloud.png'
    plt.figure(figsize=(10, 6))
    # [Add wordcloud code here]
    plt.savefig('word_cloud.png')
    plt.close()

def plot_perplexity(train_perplexities, val_perplexities):
    # [Placeholder: Implement perplexity plot]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_perplexities) + 1), train_perplexities, label='Training Perplexity')
    plt.plot(range(1, len(val_perplexities) + 1), val_perplexities, label='Validation Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity over Epochs')
    plt.legend()
    plt.savefig('perplexity_plot.png')
    plt.close()

def plot_attention_heatmap(model, input_ids):
    # [Placeholder: Implement attention weights heatmap]
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, mask=None)
        # Extract attention weights from model (modify based on your implementation)
        attn_weights = torch.rand(12, input_ids.size(1), input_ids.size(1))  # Dummy data
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights[0], cmap='viridis')
        plt.title('Attention Weights Heatmap (Layer 1)')
        plt.savefig('attention_heatmap.png')
        plt.close()

# 4. Training Function
def train_model(model, train_loader, val_loader, epochs=5, lr=0.0001, device='cuda'):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_perplexities, val_perplexities = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        
        train_perplexity = np.exp(total_loss / len(train_loader))
        train_perplexities.append(train_perplexity)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1)).item()
        
        val_perplexity = np.exp(val_loss / len(val_loader))
        val_perplexities.append(val_perplexity)
        print(f'Epoch {epoch+1}: Train Perplexity = {train_perplexity:.4f}, Val Perplexity = {val_perplexity:.4f}')

    return train_perplexities, val_perplexities

# 5. Text Generation
def generate_text(model, tokenizer, prompt="Once upon a time", max_len=50, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated = input_ids
    with torch.no_grad():
        for _ in range(max_len):
            outputs = model(generated)
            next_token = torch.argmax(outputs[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)
    return tokenizer.decode(generated[0])

# 6. Main Execution
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataset, val_dataset = load_dataset_tinystories()
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = GPT2Model(vocab_size=50257, d_model=768, n_layers=12, n_heads=12, d_ff=3072)
    train_perplexities, val_perplexities = train_model(model, train_loader, val_loader, epochs=5)

    # Generate visualizations
    plot_word_cloud(train_dataset.data)
    plot_perplexity(train_perplexities, val_perplexities)
    sample_input = train_dataset[0][0].unsqueeze(0).to(device)
    plot_attention_heatmap(model, sample_input)

    # Generate text
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    generated_text = generate_text(model, tokenizer)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()
```

### Instructions for Conversion and Integration
1. **Convert Your .ipynb File**:
   - **Jupyter Notebook**: Open your .ipynb file, go to `File > Download as > Python (.py)`, and save it (e.g., `gpt2_project.py`).
   - **Command Line**: Run `jupyter nbconvert --to python your_notebook.ipynb` to generate `your_notebook.py`.
   - **Manual Extraction**: If you can’t use the above, open the .ipynb file in a text editor, extract code cells (JSON `cell_type: code`), and copy them into a .py file.

2. **Integrate Your Code**:
   - The sample script above includes placeholders for key functions (`GPT2Model`, `TransformerBlock`, `MultiHeadAttention`, `plot_word_cloud`, `plot_attention_heatmap`). Replace these placeholders with the actual code from your converted .py file.
   - For example, copy your implementation of multi-head self-attention into the `MultiHeadAttention.forward` method or your word cloud logic into `plot_word_cloud`.

3. **Charts**:
   - Your notebook likely includes `plot_word_cloud`, `plot_perplexity`, and `plot_attention_heatmap`. Ensure these functions save images (e.g., `plt.savefig('word_cloud.png')`) in your .ipynb file.
   - After conversion, check that the .py file includes these save commands. If not, add them as shown in the sample script (e.g., `plt.savefig('perplexity_plot.png')`).
   - The charts should be saved as `word_cloud.png`, `perplexity_plot.png`, and `attention_heatmap.png` for use in the Word report.

4. **Update the Report**:
   - Use the generated text from `generate_text` (run the script with `python gpt2_project.py`) and insert it into the report’s “Generated Text” section.
   - Insert the chart images into the Word document at the placeholders (e.g., `Insert > Pictures > word_cloud.png`).

5. **Run the Script**:
   - Save the script as `gpt2_project.py`.
   - Install dependencies: `pip install torch transformers datasets matplotlib seaborn wordcloud`.
   - Run: `python gpt2_project.py` to train the model, generate visualizations, and produce text output.

6. **Word Report**:
   - Copy the Markdown report from the previous response into a Word document.
   - Insert the chart images (`word_cloud.png`, `perplexity_plot.png`, `attention_heatmap.png`) at the placeholders.
   - Update the “Generated Text” section with the output from `generate_text`.
   - Replace `[Insert GitHub repository link]` and `[Your Name], [Team Member Names]` as needed.

### Notes
- **Missing .ipynb Content**: Since I don’t have your notebook, the script includes placeholders for critical components. Please share your .ipynb file or specific code snippets (e.g., model definition, visualization functions) for a more precise conversion.
- **Chart Generation**: The script assumes your visualization functions use Matplotlib/Seaborn and save images. If they use different libraries (e.g., Plotly), let me know, and I can adjust the script.
- **Dependencies**: Ensure your environment has the required libraries. The script uses PyTorch, Hugging Face (transformers, datasets), Matplotlib, Seaborn, and WordCloud.
- **Execution Time**: Training on TinyStories with a GPU may take ~10 hours for 5 epochs, as noted in the report. Ensure your setup (e.g., Google Colab) has sufficient resources.

If you can share your .ipynb file or specific code, I can provide a fully tailored .py file. Let me know if you need help with the conversion process, running the script, or updating the Word report!