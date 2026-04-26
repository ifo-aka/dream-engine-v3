public class TransformerBlock {
    private MultiHeadAttention attention;
    private Linear ffn1;
    private Linear ffn2;
    private LayerNorm ln1;
    private LayerNorm ln2;
    private Dropout drop1, drop2;
    private ThreadLocal<float[][]> lastLn1Out = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastLn2Out = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastFfn1Pre = new ThreadLocal<>(); // Pre-GELU output for backward
    private ThreadLocal<float[][]> lastInput = new ThreadLocal<>();

    public TransformerBlock(int dModel, int numHeads, int ffnDim) {
        this.attention = new MultiHeadAttention(dModel, numHeads);
        this.ffn1 = new Linear(dModel, ffnDim);
        this.ffn2 = new Linear(ffnDim, dModel);
        this.ln1 = new LayerNorm(dModel);
        this.ln2 = new LayerNorm(dModel);
        this.drop1 = new Dropout(0.1);
        this.drop2 = new Dropout(0.1);
    }

    public float[][] forward(float[][] input) {
        lastInput.set(input);
        
        // Block 1: Residual + Attention(LayerNorm(X))
        float[][] lLn1Out = ln1.forward(input);
        lastLn1Out.set(lLn1Out);
        float[][] attnOut = attention.forward(lLn1Out);
        attnOut = drop1.forward(attnOut);
        float[][] x1 = MatrixOps.add(input, attnOut); // Residual
        
        // Block 2: Residual + FFN(LayerNorm(X1))
        float[][] lLn2Out = ln2.forward(x1);
        lastLn2Out.set(lLn2Out);
        float[][] ffn1Out = ffn1.forward(lLn2Out); // Pre-activation
        lastFfn1Pre.set(ffn1Out); // Cache pre-GELU for backward
        float[][] ffn1Activated = MatrixOps.gelu(ffn1Out); // GELU activation (v3)
        
        float[][] ffnOut = ffn2.forward(ffn1Activated);
        ffnOut = drop2.forward(ffnOut);
        
        return MatrixOps.add(x1, ffnOut); // Residual
    }

    public float[][] backward(float[][] gradOutput) {
        // Backprop through Block 2
        float[][] lFfn1Pre = lastFfn1Pre.get();

        float[][] gradFfnDrop = drop2.backward(gradOutput);
        float[][] gradFfn2 = ffn2.backward(gradFfnDrop);
        // GELU backward: needs pre-activation input
        float[][] gradGelu = MatrixOps.geluBackward(lFfn1Pre, gradFfn2);
        float[][] gradFfn1 = ffn1.backward(gradGelu);
        float[][] gradLn2 = ln2.backward(gradFfn1);
        
        // Sum gradients for Block 1
        float[][] gradX1 = new float[gradOutput.length][gradOutput[0].length];
        for(int i = 0; i < gradX1.length; i++) {
            for(int j = 0; j < gradX1[0].length; j++) gradX1[i][j] = gradOutput[i][j] + gradLn2[i][j];
        }

        // Backprop through Block 1
        float[][] gradAttnDrop = drop1.backward(gradX1);
        float[][] gradAttn = attention.backward(gradAttnDrop);
        float[][] gradLn1 = ln1.backward(gradAttn);
        
        // Sum gradients back to input
        float[][] gradInput = new float[gradX1.length][gradX1[0].length];
        for(int i = 0; i < gradInput.length; i++) {
            for(int j = 0; j < gradInput[0].length; j++) {
                gradInput[i][j] = gradX1[i][j] + gradLn1[i][j];
            }
        }
        
        return gradInput;
    }

    public void updateWeights(float lr) {
        attention.updateWeights(lr);
        ffn1.updateWeights(lr);
        ffn2.updateWeights(lr);
        ln1.updateWeights(lr);
        ln2.updateWeights(lr);
    }

    /** Propagate training/inference mode to dropout layers */
    public void setTraining(boolean training) {
        if (drop1 != null) drop1.setTraining(training);
        if (drop2 != null) drop2.setTraining(training);
    }

    /** Release ThreadLocal caches to prevent memory leaks in thread pools */
    public void cleanup() {
        lastLn1Out.remove();
        lastLn2Out.remove();
        lastFfn1Pre.remove();
        lastInput.remove();
        attention.cleanup();
    }

    public long getParamCount() {
        return attention.getParamCount() + ffn1.getParamCount() + ffn2.getParamCount();
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        attention.saveBinary(out);
        ffn1.saveBinary(out);
        ffn2.saveBinary(out);
        ln1.saveBinary(out);
        ln2.saveBinary(out);
    }

    public void loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        attention.loadBinary(in);
        ffn1.loadBinary(in);
        ffn2.loadBinary(in);
        ln1.loadBinary(in);
        ln2.loadBinary(in);
    }

    // Jackson serialization
    public TransformerBlock() {}

    public MultiHeadAttention getAttention() { return attention; }
    public void setAttention(MultiHeadAttention attention) { this.attention = attention; }

    public Linear getFfn1() { return ffn1; }
    public void setFfn1(Linear ffn1) { this.ffn1 = ffn1; }

    public Linear getFfn2() { return ffn2; }
    public void setFfn2(Linear ffn2) { this.ffn2 = ffn2; }

    public LayerNorm getLn1() { return ln1; }
    public void setLn1(LayerNorm ln1) { this.ln1 = ln1; }

    public LayerNorm getLn2() { return ln2; }
    public void setLn2(LayerNorm ln2) { this.ln2 = ln2; }

    public void initMissingComponents(int dModel) {
        if (ln1 == null) ln1 = new LayerNorm(dModel);
        if (ln2 == null) ln2 = new LayerNorm(dModel);
        if (drop1 == null) drop1 = new Dropout(0.1);
        if (drop2 == null) drop2 = new Dropout(0.1);
        if (attention != null) attention.setDModel(dModel);
    }
}
