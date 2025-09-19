# Transformer Implementation from Scratch

A PyTorch implementation of the Transformer architecture from the paper "Attention is All You Need" by Vaswani et al. This project implements all core components of the Transformer model including multi-head attention, encoder-decoder architecture, and positional encoding.

## 🤯 The Beautiful Simplicity of Transformers

What makes the Transformer architecture truly remarkable is its **elegant simplicity**. Despite revolutionizing the field of NLP and becoming the foundation for models like GPT, BERT, and ChatGPT, the core concept is surprisingly straightforward:

### 💡 **The "Aha!" Moment**

The entire Transformer boils down to one brilliant insight: **"What if we just let every word talk to every other word directly?"**

- **No complex loops** like RNNs that process words one by one
- **No convoluted hierarchies** like traditional CNNs
- Just **pure attention** - letting each position look at all other positions and decide what's important

### 🎨 **Architectural Elegance**

```
Input → Embeddings → Attention → Feed Forward → Output
```

That's it! The magic happens in just **~300 lines of PyTorch code** across 6 files:

- `multihead_attention.py` (49 lines) - The heart of it all
- `transformer_block.py` (24 lines) - Just attention + FFN + residuals
- `en_block.py` (25 lines) - Stack identical blocks
- `decoder.py` (45 lines) - Add masked attention
- `transformer.py` (88 lines) - Wire everything together

### 🚀 **Why It Works So Well**

1. **Parallelization**: Unlike RNNs, all positions process simultaneously
2. **Long-range dependencies**: Direct connections between any two positions
3. **Inductive bias**: Minimal assumptions, let the model learn patterns
4. **Scalability**: Same architecture works from 100M to 175B+ parameters

### 🧩 **The Building Blocks**

- **Attention**: `Q × K^T × V` - literally just matrix multiplication!
- **Layer Norm**: Normalize and stabilize training
- **Residual Connections**: Skip connections prevent vanishing gradients
- **Position Encoding**: Tell the model about word order

The genius is in the **composition**, not complexity. Each component is simple, but together they create something that can understand language, write code, and even generate images!

## 🏗️ Project Structure

```
transformer_my_way/
├── transformer/
│   ├── __init__.py
│   ├── multihead_attention.py    # Multi-head attention mechanism
│   ├── transformer_block.py      # Transformer block (encoder layer)
│   ├── en_block.py              # Encoder implementation
│   ├── decoder.py               # Decoder implementation
│   └── transformer.py           # Main Transformer model
├── myvenv/                      # Virtual environment
├── train.py                     # Training script (if available)
└── README.md                    # This file
```

## 🚀 Features

- **Multi-Head Attention**: Scaled dot-product attention with multiple attention heads
- **Encoder**: Stack of transformer blocks with self-attention
- **Decoder**: Stack of decoder blocks with self-attention and cross-attention
- **Positional Encoding**: Learned positional embeddings
- **Layer Normalization**: Applied after each sub-layer
- **Feed-Forward Networks**: Position-wise fully connected layers
- **Masking**: Support for padding masks and causal masks

## 📋 Requirements

- Python 3.7+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/anuz505/attention-is-all-you-need-implementation.git
cd attention-is-all-you-need-implementation
```

2. Create and activate a virtual environment:

```bash
python -m venv myvenv
# On Windows
myvenv\Scripts\activate
# On macOS/Linux
source myvenv/bin/activate
```

3. Install dependencies:

```bash
pip install torch torchvision numpy
```

## 🎯 Usage

### Basic Example

```python
import torch
from transformer.transformer import Transformer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define vocabulary sizes and special tokens
src_vocab_size = 10000  # Source vocabulary size
trg_vocab_size = 10000  # Target vocabulary size
src_pad_idx = 0         # Source padding token index
trg_pad_idx = 0         # Target padding token index

# Create model
model = Transformer(
    src_vocab_size=src_vocab_size,
    trg_vocab_size=trg_vocab_size,
    src_pad_idx=src_pad_idx,
    trg_pad_idx=trg_pad_idx,
    embed_size=512,         # Embedding dimension
    num_layers=6,           # Number of encoder/decoder layers
    forward_expansion=4,    # FFN expansion factor
    heads=8,               # Number of attention heads
    dropout=0.1,           # Dropout rate
    device=device,
    max_length=100         # Maximum sequence length
).to(device)

# Example input sequences
src = torch.randint(1, src_vocab_size, (2, 10)).to(device)  # Batch size 2, seq len 10
trg = torch.randint(1, trg_vocab_size, (2, 8)).to(device)   # Batch size 2, seq len 8

# Forward pass
output = model(src, trg[:, :-1])  # Teacher forcing: exclude last token
print(f"Output shape: {output.shape}")  # [batch_size, trg_seq_len-1, trg_vocab_size]
```

### Running the Demo

```bash
cd transformer
python transformer.py
```

This will run a simple demo with toy data and print the output shape.

## 🛠️ Building This Implementation: A Journey

### 🔥 **The Development Story**

This implementation started as a learning exercise but became something beautiful in its simplicity. Here's what makes it special:

#### **From Chaos to Clarity** 🌟

- **Started with import errors** - Classic Python module confusion!
- **Fixed circular dependencies** - `transformer.py` was importing from itself 🤦‍♂️
- **Debugged tensor shapes** - The eternal struggle of `einsum` operations
- **Achieved working code** - That magical moment when `torch.Size([2, 7, 10])` finally printed!

#### **The "Simple but Not Easy" Principle** 🎯

```python
# This looks simple...
attention = torch.einsum("nqhd,nkhd->nhqk", queries, keys)

# But getting the dimensions right? That's the art! 🎨
# queries: [batch, query_len, heads, head_dim]
# keys:    [batch, key_len, heads, head_dim]
# output:  [batch, heads, query_len, key_len]
```

#### **What We Learned Building This** 📚

1. **Einstein Notation is Magic**: `einsum` makes tensor operations readable
2. **Masking is Crucial**: Without proper masks, attention goes haywire
3. **Residual Connections Save Lives**: Gradients flow, training succeeds
4. **Parameter Order Matters**: `(query, key, value)` vs `(value, key, query)` - one character, big difference!

#### **The Debugging War Stories** 🐛

```bash
# The classic error we all know and love:
RuntimeError: einsum(): subscript l has size 7 for operand 1
which does not broadcast with previously seen size 9

# Translation: "Your tensor shapes don't match, fix your life!" 😅
```

### 🎭 **Fun Implementation Facts**

- **Total Lines**: ~231 lines of actual code (excluding comments)
- **Files Created**: 6 Python modules working in harmony
- **Import Fixes**: 3 circular import issues resolved
- **Tensor Shape Bugs**: Lost count, but each one taught us something!
- **"It Works!" Moments**: Priceless 🎉

### 🧠 **Why Hand-Code Instead of Using Libraries?**

- **Deep Understanding**: You truly get how attention works
- **Debugging Skills**: When shapes don't match, you know exactly where to look
- **Customization Freedom**: Want to modify attention? You own every line!
- **Interview Prep**: Can you implement a Transformer? Now you can! 💪

## 🏛️ Architecture Details

### Multi-Head Attention (`multihead_attention.py`)

Implements the scaled dot-product attention mechanism:

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

- **Input**: Query, Key, Value tensors and optional mask
- **Output**: Context-aware representations
- **Features**:
  - Configurable number of attention heads
  - Supports different sequence lengths for Q, K, V
  - Masking support for padding and causal attention

### Transformer Block (`transformer_block.py`)

A single transformer layer consisting of:

1. Multi-head self-attention
2. Add & Norm (residual connection + layer normalization)
3. Position-wise feed-forward network
4. Add & Norm

### Encoder (`en_block.py`)

- Stack of identical transformer blocks
- Includes input embeddings and positional encoding
- Processes the source sequence

### Decoder (`decoder.py`)

- **DecoderBlock**: Contains masked self-attention, cross-attention, and FFN
- **MainDecoder**: Stack of decoder blocks with output projection
- Includes target embeddings and positional encoding

### Main Transformer (`transformer.py`)

Combines encoder and decoder with:

- Source and target padding masks
- Target causal mask (for autoregressive generation)
- Complete forward pass implementation

## � The Mathematical Beauty

### 🧮 **Attention: Just 3 Matrix Multiplications!**

```python
# The entire attention mechanism in 3 lines:
scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(d_k)  # Similarity
attention = torch.softmax(scores, dim=-1)                   # Probabilities
output = torch.matmul(attention, V)                         # Weighted sum
```

That's it! The mechanism that powers ChatGPT, GPT-4, and every modern language model.

### 🎯 **Why This Formula is Genius**

- **Q (Queries)**: "What am I looking for?"
- **K (Keys)**: "What do I have to offer?"
- **V (Values)**: "Here's my actual content"
- **Attention Matrix**: "How much should I care about each word?"

### ⚡ **Computational Complexity Magic**

```
RNN: O(n × d²) sequential steps (slow!)
CNN: O(n × d² × k) with limited receptive field
Transformer: O(n² × d) fully parallel (fast!)
```

For typical sequences (n < 1000), the parallelization wins big time!

### 🎨 **Multi-Head: Multiple Perspectives**

Instead of one attention mechanism, we have 8 parallel "heads":

- Head 1 might focus on syntax
- Head 2 might capture semantics
- Head 3 might track long-range dependencies
- Each head learns different patterns automatically!

```python
# Split into heads: [batch, seq, embed] → [batch, seq, heads, head_dim]
queries = queries.view(batch, seq, heads, head_dim)
# Process each head independently, then concatenate
```

## �🔧 Model Configuration

| Parameter           | Default | Description                      |
| ------------------- | ------- | -------------------------------- |
| `embed_size`        | 512     | Embedding dimension              |
| `num_layers`        | 6       | Number of encoder/decoder layers |
| `heads`             | 8       | Number of attention heads        |
| `forward_expansion` | 4       | FFN hidden size multiplier       |
| `dropout`           | 0.1     | Dropout probability              |
| `max_length`        | 100     | Maximum sequence length          |

## 📊 Example Output

```bash
cuda
torch.Size([2, 7, 10])
```

- Uses CUDA if available
- Output tensor: [batch_size=2, sequence_length=7, vocab_size=10]

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the correct directory and the virtual environment is activated
2. **CUDA Out of Memory**: Reduce batch size or sequence length
3. **Dimension Mismatch**: Ensure `embed_size` is divisible by `heads`

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Code walkthrough

**Happy coding! 🎉**
