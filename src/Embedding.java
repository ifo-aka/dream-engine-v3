//Embedding.java
public class Embedding {
    private float[][] table;
    private float[][] gradTable;
    private ThreadLocal<int[]> lastIndices = new ThreadLocal<>();
    // Adam State
    private float[][] mTable, vTable;
    private int t = 0;


    public Embedding(int vocabSize, int dModel) {
        this.table = MatrixOps.randomMatrix(vocabSize, dModel);
        this.gradTable = new float[vocabSize][dModel];
        this.mTable = new float[vocabSize][dModel];
        this.vTable = new float[vocabSize][dModel];
    }

    public float[][] forward(int[] indices) {
        lastIndices.set(indices);
        int seqLen = indices.length;
        int dModel = table[0].length;
        float[][] output = new float[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            System.arraycopy(table[indices[i]], 0, output[i], 0, dModel);
        }
        return output;
    }

    public void backward(float[][] gradOutput) {
        int[] lIdx = lastIndices.get();
        synchronized(this) {
            for (int i = 0; i < lIdx.length; i++) {
                int idx = lIdx[i];
                float[] gRow = gradTable[idx];
                float[] oRow = gradOutput[i];
                for (int j = 0; j < gRow.length; j++) {
                    gRow[j] += oRow[j];
                }
            }
        }
    }

    /** 
     * For weight-tied LM head: compute logits = hidden @ table^T 
     * Returns [seqLen][vocabSize] logits
     */
    public float[][] projectToVocab(float[][] hidden) {
        return MatrixOps.multiplyWithTransposeB(hidden, table);
    }

    /**
     * For weight-tied LM head backward: accumulate gradients into embedding table
     * gradLogits: [seqLen][vocabSize], hidden: [seqLen][dModel]
     * Gradient w.r.t. table += gradLogits^T @ hidden (scattered by vocab)
     * Gradient w.r.t. hidden = gradLogits @ table
     */
    public float[][] backwardProjection(float[][] gradLogits, float[][] hidden) {
        // Gradient to embedding table: gradLogits^T @ hidden
        float[][] gradTable_batch = MatrixOps.multiplyTransposeA(gradLogits, hidden);
        synchronized(this) {
            for (int i = 0; i < gradTable.length; i++) {
                for (int j = 0; j < gradTable[0].length; j++) {
                    gradTable[i][j] += gradTable_batch[i][j];
                }
            }
        }
        // Gradient to hidden: gradLogits @ table
        return MatrixOps.multiply(gradLogits, table);
    }

    public void updateWeights(float lr) {
        t++;
        
        MatrixOps.clipByNorm(gradTable, 1.0f);
        MatrixOps.adamUpdate(table, gradTable, mTable, vTable, 0.9f, 0.999f, lr, t);
        
        // Clear gradients
        for(int i = 0; i < gradTable.length; i++) java.util.Arrays.fill(gradTable[i], 0);
    }

    // --- Binary Serialization ---
    public void saveBinary(java.io.DataOutputStream out) throws java.io.IOException {
        out.writeInt(table.length);
        out.writeInt(table[0].length);
        for (float[] row : table)
            for (float v : row) out.writeFloat(v);
        // Save optimizer state
        out.writeInt(t);
        for (float[] row : mTable) for (float v : row) out.writeFloat(v);
        for (float[] row : vTable) for (float v : row) out.writeFloat(v);
    }

    public void loadBinary(java.io.DataInputStream in) throws java.io.IOException {
        int rows = in.readInt();
        int cols = in.readInt();
        this.table = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++) table[i][j] = in.readFloat();
        this.t = in.readInt();
        this.mTable = new float[rows][cols];
        this.vTable = new float[rows][cols];
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) mTable[i][j] = in.readFloat();
        for (int i = 0; i < rows; i++) for (int j = 0; j < cols; j++) vTable[i][j] = in.readFloat();
        this.gradTable = new float[rows][cols];
    }

    public long getParamCount() {
        return (long)table.length * table[0].length;
    }

    // Jackson serialization
    public Embedding() {}

    public float[][] getTable() { return table; }
    public void setTable(float[][] table) { 
        this.table = table;
        if (gradTable == null && mTable == null) {
            int rows = table.length;
            int cols = table[0].length;
            this.gradTable = new float[rows][cols];
            this.mTable = new float[rows][cols];
            this.vTable = new float[rows][cols];
        } else if (gradTable == null) {
            this.gradTable = new float[table.length][table[0].length];
        }
    }

    public float[][] getMTable() { return mTable; }
    public void setMTable(float[][] mTable) { this.mTable = mTable; }
    public float[][] getVTable() { return vTable; }
    public void setVTable(float[][] vTable) { this.vTable = vTable; }
    public int getT() { return t; }
    public void setT(int t) { this.t = t; }
}
