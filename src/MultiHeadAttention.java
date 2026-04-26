public class MultiHeadAttention {
    private int numHeads;
    private int dModel;
    private int headDim;
    private Linear qProjection;
    private Linear kProjection;
    private Linear vProjection;
    private Linear outProjection;

    // Cache for backward
    private ThreadLocal<float[][]> lastInput = new ThreadLocal<>();
    private ThreadLocal<float[][][]> lastHeadWeights = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastQ = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastK = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastV = new ThreadLocal<>();

    public MultiHeadAttention(int dModel, int numHeads) {
        this.dModel = dModel;
        this.numHeads = numHeads;
        this.headDim = dModel / numHeads;
        
        this.qProjection = new Linear(dModel, dModel);
        this.kProjection = new Linear(dModel, dModel);
        this.vProjection = new Linear(dModel, dModel);
        this.outProjection = new Linear(dModel, dModel);
    }

    public float[][] forward(float[][] input) {
        lastInput.set(input);
        int seqLen = input.length;
        float[][] lQ = qProjection.forward(input);
        float[][] lK = kProjection.forward(input);
        float[][] lV = vProjection.forward(input);
        lastQ.set(lQ);
        lastK.set(lK);
        lastV.set(lV);

        float[][] combinedContext = new float[seqLen][dModel];
        float[][][] lHeadWeights = new float[numHeads][seqLen][seqLen];

        for (int h = 0; h < numHeads; h++) {
            int start = h * headDim;
            float[][] qH = extractHead(lQ, start, seqLen);
            float[][] kH = extractHead(lK, start, seqLen);
            float[][] vH = extractHead(lV, start, seqLen);

            // Attention scores: (Q_h @ K_h^T) / sqrt(headDim)
            float[][] scores = MatrixOps.multiplyWithTransposeB(qH, kH);
            float scale = (float) Math.sqrt(headDim);
            for(int i = 0; i < seqLen; i++) {
                for(int j = 0; j < seqLen; j++) {
                    scores[i][j] /= scale;
                    if (j > i) scores[i][j] = Float.NEGATIVE_INFINITY;
                }
            }
            float[][] weights = MatrixOps.softmax(scores);
            lHeadWeights[h] = weights;

            float[][] headContext = MatrixOps.multiply(weights, vH);
            
            for(int i = 0; i < seqLen; i++) {
                System.arraycopy(headContext[i], 0, combinedContext[i], start, headDim);
            }
        }
        lastHeadWeights.set(lHeadWeights);

        return outProjection.forward(combinedContext);
    }

    private float[][] extractHead(float[][] full, int start, int seqLen) {
        float[][] head = new float[seqLen][headDim];
        for(int i = 0; i < seqLen; i++) {
            System.arraycopy(full[i], start, head[i], 0, headDim);
        }
        return head;
    }

    public float[][] backward(float[][] gradOutput) {
        float[][] gradCombined = outProjection.backward(gradOutput);
        int seqLen = gradCombined.length;
        
        float[][] gradQ = new float[seqLen][dModel];
        float[][] gradK = new float[seqLen][dModel];
        float[][] gradV = new float[seqLen][dModel];

        float[][][] lHeadWeights = lastHeadWeights.get();
        float[][] lQ = lastQ.get();
        float[][] lK = lastK.get();
        float[][] lV = lastV.get();

        for (int h = 0; h < numHeads; h++) {
            int start = h * headDim;
            float[][] gradHead = extractHead(gradCombined, start, seqLen);
            float[][] qH = extractHead(lQ, start, seqLen);
            float[][] kH = extractHead(lK, start, seqLen);
            float[][] vH = extractHead(lV, start, seqLen);
            float[][] weights = lHeadWeights[h];

            float[][] gV = MatrixOps.multiplyTransposeA(weights, gradHead);
            float[][] gW = MatrixOps.multiplyWithTransposeB(gradHead, vH);
            
            float[][] gS = new float[seqLen][seqLen];
            float invSqrt = 1.0f / (float) Math.sqrt(headDim);
            for(int i = 0; i < seqLen; i++) {
                float dot = 0;
                for(int k = 0; k < seqLen; k++) dot += gW[i][k] * weights[i][k];
                for(int j = 0; j < seqLen; j++) gS[i][j] = weights[i][j] * (gW[i][j] - dot) * invSqrt;
            }

            float[][] gQ = MatrixOps.multiply(gS, kH);
            float[][] gK = MatrixOps.multiplyTransposeA(gS, qH);

            for(int i = 0; i < seqLen; i++) {
                System.arraycopy(gQ[i], 0, gradQ[i], start, headDim);
                System.arraycopy(gK[i], 0, gradK[i], start, headDim);
                System.arraycopy(gV[i], 0, gradV[i], start, headDim);
            }
        }

        float[][] gInQ = qProjection.backward(gradQ);
        float[][] gInK = kProjection.backward(gradK);
        float[][] gInV = vProjection.backward(gradV);

        float[][] gradInput = new float[seqLen][dModel];
        for(int i = 0; i < seqLen; i++) {
            for(int j = 0; j < dModel; j++) gradInput[i][j] = gInQ[i][j] + gInK[i][j] + gInV[i][j];
        }
        return gradInput;
    }

    public void updateWeights(float lr) {
        qProjection.updateWeights(lr);
        kProjection.updateWeights(lr);
        vProjection.updateWeights(lr);
        outProjection.updateWeights(lr);
    }

    public long getParamCount() {
        return qProjection.getParamCount() + kProjection.getParamCount() + vProjection.getParamCount() + outProjection.getParamCount();
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        qProjection.saveBinary(out);
        kProjection.saveBinary(out);
        vProjection.saveBinary(out);
        outProjection.saveBinary(out);
    }

    public void loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        qProjection.loadBinary(in);
        kProjection.loadBinary(in);
        vProjection.loadBinary(in);
        outProjection.loadBinary(in);
    }

    /** Release ThreadLocal caches */
    public void cleanup() {
        lastInput.remove();
        lastHeadWeights.remove();
        lastQ.remove();
        lastK.remove();
        lastV.remove();
    }

    // Jackson serialization
    public MultiHeadAttention() {}

    public int getNumHeads() { return numHeads; }
    public void setNumHeads(int numHeads) { this.numHeads = numHeads; }

    public int getDModel() { return dModel; }
    public void setDModel(int dModel) { 
        this.dModel = dModel; 
        if (numHeads > 0) this.headDim = dModel / numHeads;
    }

    public Linear getQProjection() { return qProjection; }
    public void setQProjection(Linear qProjection) { this.qProjection = qProjection; }
    
    public Linear getKProjection() { return kProjection; }
    public void setKProjection(Linear kProjection) { this.kProjection = kProjection; }

    public Linear getVProjection() { return vProjection; }
    public void setVProjection(Linear vProjection) { this.vProjection = vProjection; }

    public Linear getOutProjection() { return outProjection; }
    public void setOutProjection(Linear outProjection) { this.outProjection = outProjection; }
}
