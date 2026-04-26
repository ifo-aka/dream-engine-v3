//LayerNorm.java
public class LayerNorm {
    private float[] gamma;
    private float[] beta;
    private ThreadLocal<float[]> lastMean = new ThreadLocal<>();
    private ThreadLocal<float[]> lastVar = new ThreadLocal<>();
    private ThreadLocal<float[][]> lastNormalized = new ThreadLocal<>();
    private int size;
    private float[] gradGamma, gradBeta;

    public LayerNorm(int size) {
        this.size = size;
        this.gamma = new float[size];
        this.beta = new float[size];
        for (int i = 0; i < size; i++) {
            gamma[i] = 1.0f;
            beta[i] = 0.0f;
        }
    }

    public float[][] forward(float[][] input) {
        int m = input.length;
        int n = size;
        float[] lMean = new float[m];
        float[] lVar  = new float[m];
        float[][] lNorm = new float[m][n];

        float eps = 1e-5f; // Slightly larger eps for float stability

        for (int i = 0; i < m; i++) {
            float mean = 0;
            for (int j = 0; j < n; j++) mean += input[i][j];
            mean /= n;
            lMean[i] = mean;

            float var = 0;
            for (int j = 0; j < n; j++) {
                float diff = input[i][j] - mean;
                var += diff * diff;
            }
            var /= n;
            lVar[i] = var;

            float stdInv = 1.0f / (float) Math.sqrt(var + eps);
            for (int j = 0; j < n; j++) {
                lNorm[i][j] = (input[i][j] - mean) * stdInv;
            }
        }
        
        lastMean.set(lMean);
        lastVar.set(lVar);
        lastNormalized.set(lNorm);

        float[][] output = new float[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                output[i][j] = lNorm[i][j] * gamma[j] + beta[j];

        return output;
    }

    public float[][] backward(float[][] gradOutput) {
        int m = gradOutput.length;
        int n = size;
        float[][] gradInput = new float[m][n];
        float eps = 1e-5f;

        float[] localGradGamma = new float[n];
        float[] localGradBeta = new float[n];

        float[] lVar = lastVar.get();
        float[][] lNorm = lastNormalized.get();

        for (int i = 0; i < m; i++) {
            float stdInv = 1.0f / (float) Math.sqrt(lVar[i] + eps);
            float sumGrad = 0;
            float sumGradX = 0;
            
            for (int j = 0; j < n; j++) {
                float dy = gradOutput[i][j] * gamma[j];
                sumGrad += dy;
                sumGradX += dy * lNorm[i][j];
                
                localGradGamma[j] += gradOutput[i][j] * lNorm[i][j];
                localGradBeta[j] += gradOutput[i][j];
            }
            
            for (int j = 0; j < n; j++) {
                float dy = gradOutput[i][j] * gamma[j];
                gradInput[i][j] = (dy - sumGrad/n - lNorm[i][j] * sumGradX/n) * stdInv;
            }
        }

        synchronized (this) {
            if (gradGamma == null) {
                gradGamma = new float[n];
                gradBeta = new float[n];
            }
            for (int j = 0; j < n; j++) {
                gradGamma[j] += localGradGamma[j];
                gradBeta[j] += localGradBeta[j];
            }
        }
        return gradInput;
    }

    // Adam state for LayerNorm
    private float[] mGamma, vGamma;
    private float[] mBeta, vBeta;
    private int t = 0;

    public void updateWeights(float lr) {
        if (mGamma == null) {
            mGamma = new float[size];
            vGamma = new float[size];
            mBeta = new float[size];
            vBeta = new float[size];
        }
        if (gradGamma == null) return;

        t++;
        
        MatrixOps.adamUpdate(gamma, gradGamma, mGamma, vGamma, 0.9f, 0.999f, lr, t);
        MatrixOps.adamUpdate(beta, gradBeta, mBeta, vBeta, 0.9f, 0.999f, lr, t);
        
        java.util.Arrays.fill(gradGamma, 0);
        java.util.Arrays.fill(gradBeta, 0);
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(size);
        for (float g : gamma) out.writeFloat(g);
        for (float b : beta) out.writeFloat(b);
        out.writeInt(t);
        if (mGamma != null) {
            out.writeBoolean(true);
            for (float v : mGamma) out.writeFloat(v);
            for (float v : vGamma) out.writeFloat(v);
            for (float v : mBeta) out.writeFloat(v);
            for (float v : vBeta) out.writeFloat(v);
        } else {
            out.writeBoolean(false);
        }
    }

    public void loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        this.size = in.readInt();
        this.gamma = new float[size];
        this.beta = new float[size];
        for (int i = 0; i < size; i++) gamma[i] = in.readFloat();
        for (int i = 0; i < size; i++) beta[i] = in.readFloat();
        this.t = in.readInt();
        boolean hasOptimizer = in.readBoolean();
        if (hasOptimizer) {
            mGamma = new float[size]; vGamma = new float[size];
            mBeta = new float[size]; vBeta = new float[size];
            for (int i = 0; i < size; i++) mGamma[i] = in.readFloat();
            for (int i = 0; i < size; i++) vGamma[i] = in.readFloat();
            for (int i = 0; i < size; i++) mBeta[i] = in.readFloat();
            for (int i = 0; i < size; i++) vBeta[i] = in.readFloat();
        }
    }

    // Jackson serialization (kept for compatibility)
    public LayerNorm() {}

    public float[] getGamma() { return gamma; }
    public void setGamma(float[] gamma) { this.gamma = gamma; }

    public float[] getBeta() { return beta; }
    public void setBeta(float[] beta) { this.beta = beta; }

    public int getSize() { return size; }
    public void setSize(int size) { this.size = size; }
}
