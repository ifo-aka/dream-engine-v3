import java.util.ArrayList;
import java.util.List;
import java.io.*;

public class Transformer {
    private List<TransformerBlock> blocks;
    private Embedding embedding;
    private PositionalEncoding positionalEncoding;
    private LayerNorm finalNorm;
    private Linear finalHead;       // null if weight-tying is enabled
    private boolean weightTying;
    private int dModel;
    private int vocabSize;
    private int maxSeqLen;

    public Transformer(ModelConfig config) {
        this.dModel = config.dModel;
        this.vocabSize = config.vocabSize;
        this.maxSeqLen = config.maxSeqLen;
        this.weightTying = config.weightTying;
        this.embedding = new Embedding(vocabSize, dModel);
        this.positionalEncoding = new PositionalEncoding(maxSeqLen, dModel);
        this.blocks = new ArrayList<>();
        for (int i = 0; i < config.numLayers; i++) {
            blocks.add(new TransformerBlock(dModel, config.numHeads, config.ffnDim));
        }
        this.finalNorm = new LayerNorm(dModel);

        // Weight tying: LM head shares weights with embedding table
        if (!weightTying) {
            this.finalHead = new Linear(dModel, vocabSize);
        }
    }
    
    public int getNumLayers() {
        return blocks.size();
    }

    public float[][] predict(int[] tokenIds) {
        float[][] x = embedding.forward(tokenIds);
        
        // Embedding scaling (standard for transformers)
        float scale = (float) Math.sqrt(dModel);
        for (int i = 0; i < x.length; i++)
            for (int j = 0; j < x[i].length; j++)
                x[i][j] *= scale;
        
        x = positionalEncoding.addPositionalEncoding(x);
        for (TransformerBlock block : blocks) {
            x = block.forward(x);
        }
        x = finalNorm.forward(x);

        // Project to vocab: either weight-tied or separate head
        if (weightTying) {
            return embedding.projectToVocab(x);
        } else {
            return finalHead.forward(x);
        }
    }

    // Thread-local cache for the hidden state before projection (needed for weight-tied backward)
    private ThreadLocal<float[][]> lastHiddenBeforeHead = new ThreadLocal<>();

    public float accumulateGradients(int[][] batchInputs, int[][] batchTargets, float[][] batchMasks) {
        int batchSize = batchInputs.length;
        final float[] lossArr = new float[batchSize];
        final int[] tokensArr = new int[batchSize];
        
        java.util.stream.IntStream.range(0, batchSize).parallel().forEach(b -> {
            // Forward — custom inline to capture hidden state
            float[][] x = embedding.forward(batchInputs[b]);
            float scale = (float) Math.sqrt(dModel);
            for (int i = 0; i < x.length; i++)
                for (int j = 0; j < x[i].length; j++)
                    x[i][j] *= scale;
            x = positionalEncoding.addPositionalEncoding(x);
            for (TransformerBlock block : blocks) {
                x = block.forward(x);
            }
            x = finalNorm.forward(x);
            
            // Cache hidden state for weight-tied backward
            float[][] hiddenState = x;
            lastHiddenBeforeHead.set(hiddenState);

            float[][] logits;
            if (weightTying) {
                logits = embedding.projectToVocab(hiddenState);
            } else {
                logits = finalHead.forward(hiddenState);
            }
            
            float[][] probs = MatrixOps.softmax(logits);

            float[][] lossGrad = new float[probs.length][vocabSize];
            float exampleLoss = 0;
            float exampleTokens = 0;
            
            for (int t = 0; t < probs.length; t++) {
                float mask = batchMasks[b][t];
                int targetIndex = batchTargets[b][t];
                
                if (mask > 0 && targetIndex >= 0 && targetIndex < vocabSize) {
                    exampleLoss -= (float) Math.log(probs[t][targetIndex] + 1e-10f) * mask;
                    exampleTokens += mask;
                    
                    for (int v = 0; v < vocabSize; v++) {
                        float g = probs[t][v] * mask;
                        if (v == targetIndex) g -= 1.0f * mask;
                        lossGrad[t][v] = g;
                    }
                }
            }
            
            if (exampleTokens > 0) {
                lossArr[b] = exampleLoss / exampleTokens;
                tokensArr[b] = 1;
            }

            // Backward through LM head
            float[][] currentGrad;
            if (weightTying) {
                currentGrad = embedding.backwardProjection(lossGrad, hiddenState);
            } else {
                currentGrad = finalHead.backward(lossGrad);
            }
            
            currentGrad = finalNorm.backward(currentGrad);
            for (int i = blocks.size() - 1; i >= 0; i--) {
                currentGrad = blocks.get(i).backward(currentGrad);
            }
            // Scale gradients for embedding scaling
            for (int i = 0; i < currentGrad.length; i++)
                for (int j = 0; j < currentGrad[i].length; j++)
                    currentGrad[i][j] *= scale;
            embedding.backward(currentGrad);
        });

        float totalLoss = 0;
        int totalTokens = 0;
        for(int i = 0; i < batchSize; i++) {
            totalLoss += lossArr[i];
            totalTokens += tokensArr[i];
        }

        return (totalTokens > 0) ? totalLoss / totalTokens : 0;
    }

    public void updateWeights(float lr) {
        if (!weightTying && finalHead != null) {
            finalHead.updateWeights(lr);
        }
        finalNorm.updateWeights(lr);
        for (TransformerBlock block : blocks) {
            block.updateWeights(lr);
        }
        embedding.updateWeights(lr);
    }

    public void setTraining(boolean training) {
        for (TransformerBlock block : blocks) {
            block.setTraining(training);
        }
    }

    public void cleanup() {
        for (TransformerBlock block : blocks) {
            block.cleanup();
        }
        lastHiddenBeforeHead.remove();
    }

    public int[] generate(int[] seedIds, int length, int maxSeqLen, float temperature, int eosTokenId) {
        setTraining(false);
        try {
            int[] result = new int[seedIds.length + length];
            System.arraycopy(seedIds, 0, result, 0, seedIds.length);
            java.util.Random rand = new java.util.Random();
            
            int finalLength = seedIds.length;
            for (int i = 0; i < length; i++) {
                int currentLen = seedIds.length + i;
                int start = Math.max(0, currentLen - maxSeqLen);
                int[] window = new int[Math.min(currentLen, maxSeqLen)];
                System.arraycopy(result, start, window, 0, window.length);
                
                float[][] output = predict(window);
                float[] logits = output[window.length - 1];
                
                // Repetition Penalty
                int lookback = Math.min(15, currentLen);
                for (int k = 1; k <= lookback; k++) {
                    int pastId = result[currentLen - k];
                    if (pastId >= 0 && pastId < logits.length) {
                        if (logits[pastId] > 0) logits[pastId] /= 1.2f;
                        else logits[pastId] *= 1.2f;
                    }
                }

                // Apply Temperature and Sample
                float[] probs = new float[vocabSize];
                float maxLogit = Float.NEGATIVE_INFINITY;
                for (float l : logits) if (l > maxLogit) maxLogit = l;
                
                float sum = 0;
                for (int j = 0; j < vocabSize; j++) {
                    probs[j] = (float) Math.exp((logits[j] - maxLogit) / temperature) + 1e-10f;
                    sum += probs[j];
                }
                for (int j = 0; j < vocabSize; j++) probs[j] /= sum;

                // Top-P (Nucleus) Sampling
                Integer[] indices = new Integer[vocabSize];
                for (int j = 0; j < vocabSize; j++) indices[j] = j;
                java.util.Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));

                float cumulativeProb = 0;
                int lastIdx = vocabSize - 1;
                for (int j = 0; j < vocabSize; j++) {
                    cumulativeProb += probs[indices[j]];
                    if (cumulativeProb >= 0.95f) { 
                        lastIdx = j;
                        break;
                    }
                }
                
                float topPSum = 0;
                for (int j = 0; j <= lastIdx; j++) topPSum += probs[indices[j]];
                
                float r = rand.nextFloat() * topPSum;
                float runningSum = 0;
                int selectedId = indices[0];
                for (int j = 0; j <= lastIdx; j++) {
                    runningSum += probs[indices[j]];
                    if (r <= runningSum) {
                        selectedId = indices[j];
                        break;
                    }
                }
                result[currentLen] = selectedId;
                finalLength++;
                if (selectedId == eosTokenId) break;
            }
            
            if (finalLength < result.length) {
                int[] trimmed = new int[finalLength];
                System.arraycopy(result, 0, trimmed, 0, finalLength);
                return trimmed;
            }
            return result;
        } finally {
            setTraining(true);
        }
    }

    public int[] generate(int[] seedIds, int length, int maxSeqLen, float temperature) {
        return generate(seedIds, length, maxSeqLen, temperature, -1);
    }

    public long getParamCount() {
        long total = embedding.getParamCount();
        for (TransformerBlock block : blocks) {
            total += block.getParamCount();
        }
        if (!weightTying && finalHead != null) {
            total += finalHead.getParamCount();
        }
        return total;
    }

    // ===== Binary Model Save/Load =====

    public void saveModel(String path) {
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path), 1 << 20))) {
            // Write config header
            ModelConfig config = new ModelConfig();
            config.dModel = dModel;
            config.vocabSize = vocabSize;
            config.maxSeqLen = maxSeqLen;
            config.numLayers = blocks.size();
            config.weightTying = weightTying;
            // Infer numHeads and ffnDim from first block (they're uniform)
            config.numHeads = 16;  // Will be saved in config
            config.ffnDim = 4096;
            config.saveBinary(out);

            // Write weights
            embedding.saveBinary(out);
            for (TransformerBlock block : blocks) {
                block.saveBinary(out);
            }
            finalNorm.saveBinary(out);
            if (!weightTying && finalHead != null) {
                finalHead.saveBinary(out);
            }
            
            out.flush();
            System.out.println("Model saved (binary) to " + path + " (" + new File(path).length() / (1024*1024) + " MB)");
        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public static Transformer loadModel(String path) {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path), 1 << 20))) {
            ModelConfig config = ModelConfig.loadBinary(in);
            System.out.println("Loading model: " + config);
            
            Transformer model = new Transformer(config);
            model.embedding.loadBinary(in);
            for (TransformerBlock block : model.blocks) {
                block.loadBinary(in);
            }
            model.finalNorm.loadBinary(in);
            if (!config.weightTying) {
                model.finalHead.loadBinary(in);
            }
            
            System.out.println("Model loaded from " + path);
            return model;
        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    // Legacy JSON save for backward compat — DO NOT USE for 200M models
    public void saveModelJson(String path) {
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            mapper.enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT);
            mapper.writeValue(new java.io.File(path), this);
            System.out.println("Model saved (JSON) to " + path);
        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
        }
    }

    // --- Getters for Dashboard/Jackson compat ---
    public Transformer() {}
    
    public int getDModel() { return dModel; }
    public void setDModel(int dModel) { this.dModel = dModel; }

    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }

    public int getMaxSeqLen() { return maxSeqLen; }
    public void setMaxSeqLen(int maxSeqLen) { this.maxSeqLen = maxSeqLen; }

    public boolean isWeightTying() { return weightTying; }
    public void setWeightTying(boolean weightTying) { this.weightTying = weightTying; }

    public Embedding getEmbedding() { return embedding; }
    public void setEmbedding(Embedding embedding) { this.embedding = embedding; }

    public PositionalEncoding getPositionalEncoding() { return positionalEncoding; }
    public void setPositionalEncoding(PositionalEncoding positionalEncoding) { this.positionalEncoding = positionalEncoding; }

    public List<TransformerBlock> getBlocks() { return blocks; }
    public void setBlocks(List<TransformerBlock> blocks) { this.blocks = blocks; }

    public Linear getFinalHead() { return finalHead; }
    public void setFinalHead(Linear finalHead) { this.finalHead = finalHead; }

    public LayerNorm getFinalNorm() { return finalNorm; }
    public void setFinalNorm(LayerNorm finalNorm) { this.finalNorm = finalNorm; }
}
