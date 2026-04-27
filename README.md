# Dream Engine V3: 77M Parameter Java Transformer

A high-performance, from-scratch implementation of a Generative Pre-trained Transformer (GPT) architecture in pure Java. No external ML frameworks, no Python dependencies—just raw computational logic.

## 🚀 The V3 Upgrade
Unlike previous iterations, V3 transitions from a basic educational tool to a production-grade architecture capable of handling complex linguistic patterns.

- **Byte Pair Encoding (BPE):** Custom 50k+ vocabulary tokenizer for efficient subword processing.
- **Advanced Activations:** Swapped standard ReLU for **GELU (Gaussian Error Linear Units)** for smoother gradient flow.
- **Scaling:** Scaled from 9M to **77 Million Parameters**.
- **Memory Optimization:** Engineered for high-volume CPU training with manual garbage collection management and parallelized matrix operations.

## 🏗️ Architecture Specs
- **Model Size:** 77,142,848 parameters
- **Layers:** 12 Transformer Blocks
- **Hidden Dimensions:** 768 (d_model)
- **Attention Heads:** 12
- **Context Window:** 128 - 256 tokens (Dynamic)
- **Optimizer:** Adam with Gradient Clipping and Warmup Scheduling

## 📂 Core Components
- `Transformer.java`: Core orchestration of the 77M parameter stack.
- `BPETokenizer.java`: Advanced subword tokenization logic.
- `MatrixOps.java`: SIMD-inspired parallel matrix math for multi-core CPUs.
- `GELU.java`: High-performance approximation of Gaussian Error Linear Units.
- `CheckpointManager.java`: Robust JSON-based weight serialization.

## 🛠️ Performance & Training
Designed specifically for high-efficiency CPU training. 
- **Current Training Status:** - **Batch Size:** 2 (Accumulation: 4 | Effective: 8)
  - **Loss Target:** < 4.0
  - **Speed:** ~0.010 b/s on consumer-grade hardware.

## 🚦 Usage

### Build
```bash
mvn clean compile
Train
Bash
mvn exec:java -Dexec.mainClass="Main"
Inference
Once trained, the engine supports interactive text generation. Note: 77M parameters require significant RAM (~2GB - 4GB) for stable inference.
```

📜 Roadmap
[x] BPE Tokenization

[x] 77M Parameter Scaling

[ ] KV Caching for 10x faster inference

[ ] 16-bit Quantization for lower memory footprint

⚖️ License
MIT - Build, break, and scale.


---

