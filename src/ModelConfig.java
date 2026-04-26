/**
 * Centralized model configuration for Dream Engine v3.
 * All hyperparameters in one place — no more magic numbers scattered across Main.java.
 */
public class ModelConfig {
    // --- Architecture ---
    public int dModel = 768;
    public int numHeads = 12;
    public int numLayers = 10;
    public int ffnDim = 3072;
    public int vocabSize = 8192;   // BPE tokenizer
    // PERF: seq=128 is 4x faster than 256 (attention is O(n^2))
    public int maxSeqLen = 128;
    public float dropoutRate = 0.1f;
    public boolean weightTying = true; // Share embedding & LM head weights

    // --- Training ---
    // PERF: batchSize=2, accum=4 → effective batch=8. Enough for stable gradients,
    //       low enough to not OOM or freeze for minutes per step.
    public int batchSize = 2;
    public int accumulationSteps = 4;  // effective batch = 8
    public int totalBatches = 20000;
    public float maxLR = 3e-4f;
    public float minLR = 3e-5f;
    public int warmupBatches = 500;
    public int checkpointInterval = 100;
    public int sampleInterval = 500;
    // Log EVERY batch so you can see it's alive
    public int logInterval = 1;

    // --- Serialization ---
    public static final int MAGIC = 0xDE3A1024;
    public static final int VERSION = 3;

    /**
     * Estimate parameter count for this configuration.
     */
    public long estimateParamCount() {
        // Embedding: vocabSize × dModel
        long embed = (long) vocabSize * dModel;

        // Per block: 4 attention projections + 2 FFN layers + 2 LayerNorms
        long attnPerBlock = 4L * ((long) dModel * dModel + dModel); // Q, K, V, O
        long ffnPerBlock = ((long) dModel * ffnDim + ffnDim)        // FFN1
                         + ((long) ffnDim * dModel + dModel);       // FFN2
        long lnPerBlock = 2L * (2L * dModel);                      // LN1 + LN2
        long perBlock = attnPerBlock + ffnPerBlock + lnPerBlock;

        // All blocks
        long allBlocks = perBlock * numLayers;

        // Final LayerNorm
        long finalLN = 2L * dModel;

        // LM Head (0 if weight-tied with embedding)
        long lmHead = weightTying ? 0 : ((long) dModel * vocabSize + vocabSize);

        return embed + allBlocks + finalLN + lmHead;
    }

    @Override
    public String toString() {
        return String.format(
            "ModelConfig{d=%d, heads=%d, layers=%d, ffn=%d, vocab=%d, seq=%d, tying=%s, params=~%,.0fM}",
            dModel, numHeads, numLayers, ffnDim, vocabSize, maxSeqLen, 
            weightTying, estimateParamCount() / 1_000_000.0
        );
    }

    /** 
     * Create a small config for testing/debugging.
     */
    public static ModelConfig small() {
        ModelConfig c = new ModelConfig();
        c.dModel = 384;
        c.numHeads = 8;
        c.numLayers = 8;
        c.ffnDim = 1536;
        c.vocabSize = 260;
        c.maxSeqLen = 128;
        c.batchSize = 12;
        c.accumulationSteps = 2;
        return c;
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(dModel);
        out.writeInt(numHeads);
        out.writeInt(numLayers);
        out.writeInt(ffnDim);
        out.writeInt(vocabSize);
        out.writeInt(maxSeqLen);
        out.writeBoolean(weightTying);
    }

    public static ModelConfig loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        int magic = in.readInt();
        if (magic != MAGIC) {
            throw new java.io.IOException("Invalid model file (bad magic number: 0x" + Integer.toHexString(magic) + ")");
        }
        int version = in.readInt();
        if (version != VERSION) {
            throw new java.io.IOException("Unsupported model version: " + version + " (expected " + VERSION + ")");
        }
        ModelConfig config = new ModelConfig();
        config.dModel = in.readInt();
        config.numHeads = in.readInt();
        config.numLayers = in.readInt();
        config.ffnDim = in.readInt();
        config.vocabSize = in.readInt();
        config.maxSeqLen = in.readInt();
        config.weightTying = in.readBoolean();
        return config;
    }
}
