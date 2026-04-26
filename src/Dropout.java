import java.util.concurrent.ThreadLocalRandom;

public class Dropout {
    private double rate;
    private ThreadLocal<float[][]> mask = new ThreadLocal<>();
    private volatile boolean isTraining;

    public Dropout(double rate) {
        this.rate = rate;
        this.isTraining = true;
    }

    // For deserialization
    public Dropout() {
        this.rate = 0.1;
        this.isTraining = true;
    }

    public void setTraining(boolean isTraining) {
        this.isTraining = isTraining;
    }

    public float[][] forward(float[][] input) {
        if (!isTraining || rate == 0.0) {
            return input;
        }

        int rows = input.length;
        int cols = input[0].length;
        float[][] lMask = new float[rows][cols];
        float[][] output = new float[rows][cols];
        float scale = (float) (1.0 / (1.0 - rate));
        ThreadLocalRandom random = ThreadLocalRandom.current();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (random.nextDouble() >= rate) {
                    lMask[i][j] = 1.0f;
                    output[i][j] = input[i][j] * scale;
                } else {
                    lMask[i][j] = 0.0f;
                    output[i][j] = 0.0f;
                }
            }
        }
        mask.set(lMask);
        return output;
    }

    public float[][] backward(float[][] gradOutput) {
        if (!isTraining || rate == 0.0) {
            return gradOutput;
        }

        float[][] gradInput = new float[gradOutput.length][gradOutput[0].length];
        float scale = (float) (1.0 / (1.0 - rate));
        float[][] lMask = mask.get();

        for (int i = 0; i < gradOutput.length; i++) {
            for (int j = 0; j < gradOutput[0].length; j++) {
                gradInput[i][j] = gradOutput[i][j] * lMask[i][j] * scale;
            }
        }
        return gradInput;
    }
}
