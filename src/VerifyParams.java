/**
 * Quick verification script — run once to confirm parameter count + memory.
 * Delete after verification.
 */
public class VerifyParams {
    public static void main(String[] args) {
        System.out.println("=== Dream Engine v3 — Parameter Verification ===\n");
        
        ModelConfig config = new ModelConfig();
        System.out.println("Config: " + config);
        System.out.println("Estimated params: " + String.format("%,d", config.estimateParamCount()));
        
        System.out.println("\nInitializing model...");
        long before = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        Transformer model = new Transformer(config);
        
        System.gc();
        long after = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        
        long actualParams = model.getParamCount();
        System.out.println("\n--- Results ---");
        System.out.printf("Actual param count:    %,d (%.1fM)%n", actualParams, actualParams / 1_000_000.0);
        System.out.printf("Estimated param count: %,d (%.1fM)%n", config.estimateParamCount(), config.estimateParamCount() / 1_000_000.0);
        System.out.printf("Memory for weights:    %,d MB%n", (after - before) / (1024 * 1024));
        System.out.printf("JVM total memory:      %,d MB%n", Runtime.getRuntime().totalMemory() / (1024 * 1024));
        System.out.printf("JVM max memory:        %,d MB%n", Runtime.getRuntime().maxMemory() / (1024 * 1024));
        
        // Verify weight tying
        System.out.println("\nWeight tying: " + config.weightTying);
        System.out.println("Architecture: d=" + config.dModel + " h=" + config.numHeads 
            + " L=" + config.numLayers + " ffn=" + config.ffnDim + " vocab=" + config.vocabSize);
        
        System.out.println("\n=== VERIFICATION COMPLETE ===");
    }
}
