import java.util.Random;

public class MatrixOps {
    
    // Tile size tuned for L1 cache. 64 floats = 256 bytes per row slice.
    // A 64x64 tile of floats = 16KB, fitting comfortably in L1 cache.
    private static final int TILE = 64;

    // Cache-blocked (tiled) Multiply: A @ B
    public static float[][] multiply(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        int p = b[0].length;
        float[][] result = new float[m][p];

        for (int ii = 0; ii < m; ii += TILE) {
            int iEnd = Math.min(ii + TILE, m);
            for (int kk = 0; kk < n; kk += TILE) {
                int kEnd = Math.min(kk + TILE, n);
                for (int jj = 0; jj < p; jj += TILE) {
                    int jEnd = Math.min(jj + TILE, p);
                    for (int i = ii; i < iEnd; i++) {
                        float[] aRow = a[i];
                        float[] resRow = result[i];
                        for (int k = kk; k < kEnd; k++) {
                            float aik = aRow[k];
                            float[] bRow = b[k];
                            for (int j = jj; j < jEnd; j++) {
                                resRow[j] += aik * bRow[j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    // Cache-blocked A^T @ B
    public static float[][] multiplyTransposeA(float[][] a, float[][] b) {
        int m = a[0].length;
        int n = a.length;
        int p = b[0].length;
        float[][] result = new float[m][p];

        for (int kk = 0; kk < n; kk += TILE) {
            int kEnd = Math.min(kk + TILE, n);
            for (int ii = 0; ii < m; ii += TILE) {
                int iEnd = Math.min(ii + TILE, m);
                for (int jj = 0; jj < p; jj += TILE) {
                    int jEnd = Math.min(jj + TILE, p);
                    for (int k = kk; k < kEnd; k++) {
                        float[] aRow = a[k];
                        float[] bRow = b[k];
                        for (int i = ii; i < iEnd; i++) {
                            float aik = aRow[i];
                            float[] resRow = result[i];
                            for (int j = jj; j < jEnd; j++) {
                                resRow[j] += aik * bRow[j];
                            }
                        }
                    }
                }
            }
        }
        return result;
    }

    // A @ B^T
    public static float[][] multiplyWithTransposeB(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        int p = b.length;
        float[][] result = new float[m][p];

        for (int ii = 0; ii < m; ii += TILE) {
            int iEnd = Math.min(ii + TILE, m);
            for (int kk = 0; kk < p; kk += TILE) {
                int kEnd = Math.min(kk + TILE, p);
                for (int i = ii; i < iEnd; i++) {
                    float[] aRow = a[i];
                    float[] resRow = result[i];
                    for (int k = kk; k < kEnd; k++) {
                        float[] bRow = b[k];
                        float dot = 0;
                        for (int j = 0; j < n; j++) {
                            dot += aRow[j] * bRow[j];
                        }
                        resRow[k] = dot;
                    }
                }
            }
        }
        return result;
    }

    public static float[][] add(float[][] a, float[] b) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[i][j] + b[j];
            }
        }
        return result;
    }

    public static float[][] add(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[i][j] + b[i][j];
            }
        }
        return result;
    }

    public static float[][] divide(float[][] a, float divisor) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[i][j] / divisor;
            }
        }
        return result;
    }

    public static float[][] softmax(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            float max = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < n; j++) if (a[i][j] > max) max = a[i][j];
            float sum = 0;
            for (int j = 0; j < n; j++) {
                result[i][j] = (float) Math.exp(a[i][j] - max);
                sum += result[i][j];
            }
            if (sum == 0) sum = 1e-10f;
            for (int j = 0; j < n; j++) result[i][j] /= sum;
        }
        return result;
    }

    // --- GELU Activation (replaces LeakyReLU for v3) ---
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    private static final float GELU_COEFF = (float) Math.sqrt(2.0 / Math.PI);

    public static float[][] gelu(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float x = a[i][j];
                float inner = GELU_COEFF * (x + 0.044715f * x * x * x);
                result[i][j] = 0.5f * x * (1.0f + (float) Math.tanh(inner));
            }
        }
        return result;
    }

    public static float[][] geluBackward(float[][] input, float[][] gradOutput) {
        int m = input.length;
        int n = input[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float x = input[i][j];
                float x3 = x * x * x;
                float inner = GELU_COEFF * (x + 0.044715f * x3);
                float tanhVal = (float) Math.tanh(inner);
                float sech2 = 1.0f - tanhVal * tanhVal;
                float dInner = GELU_COEFF * (1.0f + 3.0f * 0.044715f * x * x);
                float dGelu = 0.5f * (1.0f + tanhVal) + 0.5f * x * sech2 * dInner;
                result[i][j] = gradOutput[i][j] * dGelu;
            }
        }
        return result;
    }

    // Keep legacy relu/reluDeriv for backward compatibility if needed
    public static float[][] relu(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[i][j] > 0 ? a[i][j] : 0.01f * a[i][j];
            }
        }
        return result;
    }

    public static float[][] reluDeriv(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[i][j] = a[i][j] > 0 ? 1.0f : 0.01f;
            }
        }
        return result;
    }

    public static float[][] randomMatrix(int rows, int cols) {
        Random rand = new Random();
        float[][] matrix = new float[rows][cols];
        float scale = (float) Math.sqrt(2.0 / cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (float) rand.nextGaussian() * scale;
            }
        }
        return matrix;
    }

    public static float[] randomArray(int size) {
        return new float[size]; 
    }

    public static float[][] transpose(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[n][m];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result[j][i] = a[i][j];
            }
        }
        return result;
    }

    public static float[][] subtract(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) result[i][j] = a[i][j] - b[i][j];
        }
        return result;
    }

    public static float[][] multiplyElementWise(float[][] a, float[][] b) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) result[i][j] = a[i][j] * b[i][j];
        }
        return result;
    }

    public static float[][] scalarMultiply(float[][] a, float scalar) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) result[i][j] = a[i][j] * scalar;
        }
        return result;
    }

    public static void clip(float[][] g, float limit) {
        for (int i = 0; i < g.length; i++) {
            for (int j = 0; j < g[0].length; j++) {
                if (g[i][j] > limit) g[i][j] = limit;
                else if (g[i][j] < -limit) g[i][j] = -limit;
            }
        }
    }

    public static void clip(float[] g, float limit) {
        for (int i = 0; i < g.length; i++) {
            if (g[i] > limit) g[i] = limit;
            else if (g[i] < -limit) g[i] = -limit;
        }
    }

    /** Global Gradient Norm Clipping (2D) — preserves gradient direction */
    public static void clipByNorm(float[][] g, float maxNorm) {
        double norm = 0; // Use double for accumulation precision
        for (float[] row : g)
            for (float val : row)
                norm += (double) val * val;
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            float scale = (float) (maxNorm / norm);
            for (int i = 0; i < g.length; i++)
                for (int j = 0; j < g[0].length; j++)
                    g[i][j] *= scale;
        }
    }

    /** Global Gradient Norm Clipping (1D) — preserves gradient direction */
    public static void clipByNorm(float[] g, float maxNorm) {
        double norm = 0;
        for (float val : g)
            norm += (double) val * val;
        norm = Math.sqrt(norm);
        if (norm > maxNorm) {
            float scale = (float) (maxNorm / norm);
            for (int i = 0; i < g.length; i++)
                g[i] *= scale;
        }
    }

    public static void adamUpdate(float[][] w, float[][] g, float[][] m, float[][] v, 
                                float beta1, float beta2, float lr, int t) {
        float eps = 1e-8f;
        float b1t = 1.0f - (float) Math.pow(beta1, t);
        float b2t = 1.0f - (float) Math.pow(beta2, t);
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                m[i][j] = beta1 * m[i][j] + (1.0f - beta1) * g[i][j];
                v[i][j] = beta2 * v[i][j] + (1.0f - beta2) * (g[i][j] * g[i][j]);
                float mHat = m[i][j] / b1t;
                float vHat = v[i][j] / b2t;
                w[i][j] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
            }
        }
    }

    public static void adamUpdate(float[] w, float[] g, float[] m, float[] v, 
                                float beta1, float beta2, float lr, int t) {
        float eps = 1e-8f;
        float b1t = 1.0f - (float) Math.pow(beta1, t);
        float b2t = 1.0f - (float) Math.pow(beta2, t);
        for (int i = 0; i < w.length; i++) {
            m[i] = beta1 * m[i] + (1.0f - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g[i] * g[i]);
            float mHat = m[i] / b1t;
            float vHat = v[i] / b2t;
            w[i] -= lr * mHat / ((float) Math.sqrt(vHat) + eps);
        }
    }

    public static float[] mean(float[][] a) {
        int m = a.length;
        int n = a[0].length;
        float[] result = new float[m];
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) sum += a[i][j];
            result[i] = sum / n;
        }
        return result;
    }

    public static float[] variance(float[][] a, float[] mean) {
        int m = a.length;
        int n = a[0].length;
        float[] result = new float[m];
        for (int i = 0; i < m; i++) {
            float sum = 0;
            for (int j = 0; j < n; j++) {
                float diff = a[i][j] - mean[i];
                sum += diff * diff;
            }
            result[i] = sum / n;
        }
        return result;
    }

    public static float[][] normalize(float[][] a, float[] mean, float[] var) {
        int m = a.length;
        int n = a[0].length;
        float[][] result = new float[m][n];
        float eps = 1e-8f;
        for (int i = 0; i < m; i++) {
            float std = (float) Math.sqrt(var[i] + eps);
            for (int j = 0; j < n; j++) {
                result[i][j] = (a[i][j] - mean[i]) / std;
            }
        }
        return result;
    }
}
