public class PositionalEncoding {
    private float[][] encoding;

    public PositionalEncoding(int maxSeqLen, int dModel) {
        encoding = new float[maxSeqLen][dModel];
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dModel; i++) {
                double angle = pos / Math.pow(10000.0, (2.0 * (i / 2)) / dModel);
                encoding[pos][i] = (i % 2 == 0) ? (float) Math.sin(angle) : (float) Math.cos(angle);
            }
        }
    }

    public float[][] addPositionalEncoding(float[][] input) {
        int seqLen = input.length;
        int dModel = input[0].length;
        float[][] result = new float[seqLen][dModel];
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < dModel; j++) {
                result[i][j] = input[i][j] + encoding[i][j];
            }
        }
        return result;
    }

    // Jackson serialization
    public PositionalEncoding() {}

    public float[][] getEncoding() { return encoding; }
    public void setEncoding(float[][] encoding) { this.encoding = encoding; }
}
