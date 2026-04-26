public class Linear {
    private float[][] weights;
    private float[] bias;
    
    // Gradient storage
    private float[][] gradW;
    private float[] gradB;
    private ThreadLocal<float[][]> lastInput = new ThreadLocal<>();
    
    // Adam Optimizer State
    private float[][] mW, vW;
    private float[] mB, vB;
    private int t = 0;

    public Linear(int inputDim, int outputDim) {
        this.weights = MatrixOps.randomMatrix(inputDim, outputDim);
        this.bias = MatrixOps.randomArray(outputDim);
        initOptimizer(inputDim, outputDim);
    }

    // Required for Jackson Deserialization
    public Linear() {}

    private void initOptimizer(int inputDim, int outputDim) {
        this.gradW = new float[inputDim][outputDim];
        this.gradB = new float[outputDim];
        this.mW = new float[inputDim][outputDim];
        this.vW = new float[inputDim][outputDim];
        this.mB = new float[outputDim];
        this.vB = new float[outputDim];
    }

    public float[][] forward(float[][] input) {
        lastInput.set(input);
        float[][] product = MatrixOps.multiply(input, weights);
        return MatrixOps.add(product, bias);
    }

    public float[][] backward(float[][] gradOutput) {
        float[][] lIn = lastInput.get();

        // Use specialized transpose methods to avoid allocating temporary matrices
        float[][] gradW_batch = MatrixOps.multiplyTransposeA(lIn, gradOutput);
        
        synchronized(this) {
            for(int i = 0; i < gradW.length; i++) {
                for(int j = 0; j < gradW[0].length; j++) gradW[i][j] += gradW_batch[i][j];
            }

            int m = gradOutput.length;
            int n = gradOutput[0].length;
            for(int i = 0; i < m; i++) {
                for(int j = 0; j < n; j++) gradB[j] += gradOutput[i][j];
            }
        }

        return MatrixOps.multiplyWithTransposeB(gradOutput, weights);
    }

    public void updateWeights(float lr) {
        t++;
        
        MatrixOps.clipByNorm(gradW, 1.0f);
        MatrixOps.clipByNorm(gradB, 1.0f);
        MatrixOps.adamUpdate(weights, gradW, mW, vW, 0.9f, 0.999f, lr, t);
        MatrixOps.adamUpdate(bias, gradB, mB, vB, 0.9f, 0.999f, lr, t);
        
        // Clear gradients
        for(int i = 0; i < gradW.length; i++) java.util.Arrays.fill(gradW[i], 0);
        java.util.Arrays.fill(gradB, 0);
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(weights.length);
        out.writeInt(weights[0].length);
        for (float[] row : weights)
            for (float w : row) out.writeFloat(w);
        out.writeInt(bias.length);
        for (float b : bias) out.writeFloat(b);
        // Save optimizer state
        out.writeInt(t);
        for (float[] row : mW) for (float v : row) out.writeFloat(v);
        for (float[] row : vW) for (float v : row) out.writeFloat(v);
        for (float v : mB) out.writeFloat(v);
        for (float v : vB) out.writeFloat(v);
    }

    public void loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        int rows = in.readInt();
        int cols = in.readInt();
        this.weights = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) weights[i][j] = in.readFloat();
        int bSize = in.readInt();
        this.bias = new float[bSize];
        for (int i = 0; i < bSize; i++) bias[i] = in.readFloat();
        // Load optimizer state
        this.t = in.readInt();
        this.mW = new float[rows][cols];
        this.vW = new float[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) mW[i][j] = in.readFloat();
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) vW[i][j] = in.readFloat();
        this.mB = new float[bSize];
        this.vB = new float[bSize];
        for (int i = 0; i < bSize; i++) mB[i] = in.readFloat();
        for (int i = 0; i < bSize; i++) vB[i] = in.readFloat();
        this.gradW = new float[rows][cols];
        this.gradB = new float[bSize];
    }

    public long getParamCount() {
        return (long)weights.length * weights[0].length + bias.length;
    }

    // --- Jackson Getters/Setters (used for tokenizer JSON, kept for compat) ---
    public float[][] getWeights() { return weights; }
    public void setWeights(float[][] weights) { 
        this.weights = weights; 
        if (gradW == null) initOptimizer(weights.length, weights[0].length);
    }

    public float[] getBias() { return bias; }
    public void setBias(float[] bias) { 
        this.bias = bias;
        if (mB == null || mB.length != bias.length) {
             this.mB = new float[bias.length];
             this.vB = new float[bias.length];
             this.gradB = new float[bias.length];
        }
    }

    public float[][] getMW() { return mW; }
    public void setMW(float[][] mW) { this.mW = mW; }
    public float[][] getVW() { return vW; }
    public void setVW(float[][] vW) { this.vW = vW; }
    public float[] getMB() { return mB; }
    public void setMB(float[] mB) { this.mB = mB; }
    public float[] getVB() { return vB; }
    public void setVB(float[] vB) { this.vB = vB; }
    public int getT() { return t; }
    public void setT(int t) { this.t = t; }
}
