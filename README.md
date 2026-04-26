# Java Transformer Language Model

A from-scratch implementation of a GPT-like transformer model in pure Java.

## What This Is

This is a decoder-only transformer architecture implemented entirely in Java without external ML frameworks. It includes:

- **Multi-head self-attention** with proper scaling
- **Layer normalization** and residual connections  
- **Position-independent embeddings**
- **Adam optimizer** with gradient clipping
- **JSON serialization** for model checkpointing

## Architecture

- **Model size**: ~9M parameters
- **Layers**: 8 transformer blocks
- **Dimensions**: 256 (dModel), 1024 (FFN)
- **Heads**: 8 multi-head attention
- **Context**: 24 token sequence length
- **Training**: Batch size 16, LR 5e-6

## Components

- `Transformer.java` - Main transformer model
- `TransformerBlock.java` - Single transformer layer
- `MultiHeadAttention.java` - Attention mechanism
- `Linear.java` - Linear layer with Adam optimizer
- `Embedding.java` - Token embedding layer
- `LayerNorm.java` - Layer normalization
- `MatrixOps.java` - Matrix operations with parallelization
- `Tokenizer.java` - Simple tokenizer
- `Main.java` - Training and generation loop

## Usage

### Build
```bash
mvn clean compile
```

### Train
```bash
mvn exec:java -Dexec.mainClass="Main"
```

Place your training text in `dataset.txt`. The model will:
- Train for 3000 batches
- Auto-save checkpoints every 50 batches
- Display loss, speed, and ETA

### Generate Text
After training completes, enter prompts at the interactive prompt:
```
Input: Once upon a time
Output: [generated continuation]
```

## Model Files

- `transformer.json` - Trained model weights
- `tokenizer.bin` - Fitted tokenizer vocabulary
- `batch_state.bin` - Training progress checkpoint

## Performance

Designed for CPU training on multi-core systems. Expected training time: 8-10 hours for full 3000 batches on modern CPU.

## What This Is NOT

- This is not a production-ready language model
- It does not include advanced features like:
  - Positional encodings
  - Advanced tokenization (BPE)
  - Beam search
  - Temperature scheduling
  - Extensive hyperparameter tuning

This is an educational implementation to understand transformer architecture from first principles.

## License

Open source - feel free to learn from and modify.
