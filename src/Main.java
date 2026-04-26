import java.util.*;
import java.io.*;
import java.nio.file.*;
import io.javalin.Javalin;
import io.javalin.plugin.bundled.CorsPluginConfig;

class DataPoint {
    public int x;
    public double y;
    public DataPoint(int x, double y) { this.x = x; this.y = y; }
}

class TrainingStats {
    public int batch;
    public int totalBatches;
    public double loss;
    public double ppl;
    public double lr;
    public int seqLen;
    public int dModel;
    public int layers;
    public int heads;
    public double speed;
    public double tokensPerSec;
    public long paramCount;
    public long usedMemory;
    public long maxMemory;
    public String os;
    public String cpu;
    public String javaVersion;
    public long startTime;
    public String lastSave = "Never";
    public List<DataPoint> lossHistory = new ArrayList<>();
    public List<DataPoint> lrHistory = new ArrayList<>();
    public String status = "Training";
    public String modelVersion = "v3-200M";
    public String tokenizerType = "BPE";
    public int vocabSize;
}

public class Main {
    private static TrainingStats stats = new TrainingStats();
    private static Transformer transformer;
    private static BPETokenizer tokenizer;

    private static final String MODEL_PATH = "transformer_v3.bin";
    private static final String TOKENIZER_PATH = "bpe_tokenizer_v3.json";
    private static final String BATCH_STATE_PATH = "batch_state_v3.json";
    private static final String ENCODED_CACHE_PATH = "dataset_encoded_v3.json";

    // Special token strings
    private static final String BOS = "\u003c|bos|\u003e";
    private static final String EOS = "\u003c|eos|\u003e";
    private static final String USER_TAG = "\u003c|user|\u003e";
    private static final String ASST_TAG = "\u003c|assistant|\u003e";

    public static void main(String[] args) {
        System.out.println("=====================================================");
        System.out.println("   Dream Engine v3 - 200M Parameter Transformer");
        System.out.println("   Float32 | GELU | BPE | Weight-Tied LM Head");
        System.out.println("=====================================================");

        // ===== Configuration =====
        ModelConfig config = new ModelConfig();
        System.out.println("Config: " + config);
        System.out.println("Estimated params: " + String.format("%,d", config.estimateParamCount()));

        // ===== Dataset =====
        ensureDataset();
        
        String rawData = "";
        try {
            rawData = new String(Files.readAllBytes(Paths.get("dataset.txt")), java.nio.charset.StandardCharsets.UTF_8);
        } catch (Exception e) {
            System.out.println("Error: dataset.txt not found.");
            return;
        }
        rawData = cleanText(rawData);
        System.out.println("Raw dataset: " + rawData.length() + " characters");

        // ===== BPE Tokenizer =====
        tokenizer = new BPETokenizer();
        boolean tokenizerExists = Files.exists(Paths.get(TOKENIZER_PATH));

        if (tokenizerExists) {
            System.out.println("Loading existing BPE tokenizer from " + TOKENIZER_PATH + "...");
            tokenizer.load(TOKENIZER_PATH);
            config.vocabSize = tokenizer.getVocabSize();
            System.out.println("BPE vocab size: " + tokenizer.getVocabSize());
        } else {
            System.out.println("Training BPE tokenizer (target vocab: " + config.vocabSize + ")...");
            long tStart = System.currentTimeMillis();
            
            // OPTIMIZATION: Train BPE on a 1MB representative sample instead of all 42MB
            int sampleSize = Math.min(rawData.length(), 1_000_000); // 1 MB is plenty for BPE
            String trainData = rawData.substring(0, sampleSize);
            System.out.println("Using a " + (sampleSize/1024) + "KB sample for BPE training to speed it up.");
            
            tokenizer.train(trainData, config.vocabSize);
            long tElapsed = System.currentTimeMillis() - tStart;
            System.out.println("BPE training complete in " + (tElapsed / 1000) + "s. Vocab: " + tokenizer.getVocabSize());
            tokenizer.save(TOKENIZER_PATH);
            config.vocabSize = tokenizer.getVocabSize();
        }

        // ===== Encode Dataset =====
        com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
        mapper.enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT);

        int[] encoded;
        if (Files.exists(Paths.get(ENCODED_CACHE_PATH))) {
            System.out.println("Loading encoded dataset from cache...");
            try {
                encoded = mapper.readValue(new File(ENCODED_CACHE_PATH), int[].class);
                System.out.println("Loaded " + encoded.length + " BPE tokens from cache.");
            } catch (Exception e) {
                System.out.println("Cache read failed, re-encoding...");
                encoded = encodeAndCacheDataset(rawData, mapper);
            }
        } else {
            encoded = encodeAndCacheDataset(rawData, mapper);
        }
        
        System.out.println("Dataset: " + encoded.length + " BPE tokens");
        if (rawData.length() > 0 && encoded.length > 0) {
            double compressionRatio = (double) rawData.length() / encoded.length;
            System.out.printf("Compression ratio: %.2fx (%.1f chars per token)%n", compressionRatio, compressionRatio);
        }

        // ===== Load Batch State =====
        int currentBatch = 0;
        try {
            if (Files.exists(Paths.get(BATCH_STATE_PATH))) {
                Map<String, Integer> state = mapper.readValue(new File(BATCH_STATE_PATH),
                    new com.fasterxml.jackson.core.type.TypeReference<Map<String, Integer>>() {});
                currentBatch = state.get("batch");
                System.out.println("Resuming from batch " + currentBatch);
            }
        } catch (Exception e) {}

        // ===== Initialize Model =====
        boolean modelExists = Files.exists(Paths.get(MODEL_PATH));
        
        if (modelExists) {
            System.out.println("Loading existing v3 model (" + MODEL_PATH + ")...");
            transformer = Transformer.loadModel(MODEL_PATH);
            if (transformer == null) {
                System.out.println("Failed to load model. Initializing fresh...");
                transformer = new Transformer(config);
            }
        } else {
            System.out.println("Initializing new v3 model...");
            transformer = new Transformer(config);
        }
        
        long totalParams = transformer.getParamCount();
        stats.paramCount = totalParams;
        stats.vocabSize = config.vocabSize;
        System.out.printf("Total Parameters: %,d (%.1fM)%n", totalParams, totalParams / 1_000_000.0);

        // Memory report
        Runtime rt = Runtime.getRuntime();
        System.gc();
        long usedMB = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
        long maxMB = rt.maxMemory() / (1024 * 1024);
        System.out.printf("Memory: %dMB used / %dMB max%n", usedMB, maxMB);

        // ===== Dashboard API =====
        Javalin app = Javalin.create(c -> {
            c.plugins.enableCors(cors -> cors.add(CorsPluginConfig::anyHost));
        }).start(7070);

        stats.lossHistory = DatabaseManager.getHistoricalLoss();
        stats.lrHistory = DatabaseManager.getHistoricalLR();
        if (!stats.lossHistory.isEmpty()) {
            DataPoint last = stats.lossHistory.get(stats.lossHistory.size() - 1);
            stats.batch = last.x;
            stats.loss = last.y;
            System.out.println("Restored " + stats.lossHistory.size() + " datapoints from MySQL.");
        }

        app.get("/api/stats", ctx -> ctx.json(stats));

        app.post("/api/chat", ctx -> {
            Map body = ctx.bodyAsClass(Map.class);
            String prompt = (String) body.get("prompt");
            DatabaseManager.saveChatMessage("user", prompt);

            String chatPrompt = BOS + "\n" + USER_TAG + "\n" + prompt + "\n" + ASST_TAG;
            int[] seedIds = tokenizer.encode(chatPrompt);
            
            // BPE EOS token ID
            int eosId = 257; // EOS special token
            int[] generated = transformer.generate(seedIds, 128, config.maxSeqLen, 0.7f, eosId);
            String fullResponse = tokenizer.decode(generated);
            
            System.out.println("--- Raw Model Output ---");
            System.out.println(fullResponse);
            System.out.println("------------------------");

            String response = "";
            if (fullResponse.contains(ASST_TAG)) {
                int index = fullResponse.lastIndexOf(ASST_TAG);
                response = fullResponse.substring(index + ASST_TAG.length()).trim();
            }
            
            // Clean up trailing special tags
            for (String tag : new String[]{BOS, EOS, USER_TAG, ASST_TAG}) {
                if (response.contains(tag)) {
                    response = response.substring(0, response.indexOf(tag)).trim();
                }
            }
            
            if (response.isEmpty()) response = "... (Training in progress)";
            
            DatabaseManager.saveChatMessage("bot", response);
            ctx.json(Map.of("response", response));
        });

        System.out.println("Dashboard API running at http://localhost:7070");

        // ===== Dataset Structural Check =====
        System.out.println("\n--- DATASET STRUCTURAL CHECK ---");
        int bosId = 256;
        int samplesFound = 0;
        int checkStart = 0;
        while (samplesFound < 3 && checkStart < encoded.length - 200) {
            int tagIdx = -1;
            for (int j = checkStart; j < checkStart + 1000 && j < encoded.length; j++) {
                if (encoded[j] == bosId) {
                    tagIdx = j;
                    break;
                }
            }
            if (tagIdx != -1) {
                int[] slice = Arrays.copyOfRange(encoded, tagIdx, Math.min(encoded.length, tagIdx + 128));
                String decoded = tokenizer.decode(slice);
                if (decoded.contains(ASST_TAG) && decoded.contains(EOS)) {
                    System.out.println("Sample " + (samplesFound + 1) + ": " + decoded.replace("\n", " ").substring(0, Math.min(120, decoded.length())) + "...");
                    samplesFound++;
                }
                checkStart = tagIdx + 128;
            } else {
                checkStart += 1000;
            }
        }

        if (samplesFound == 0) {
            System.err.println("WARNING: No valid structural tags found in encoded dataset.");
            System.err.println("The model will train on raw text without chat structure.");
        }

        // ===== Training Loop =====
        int seqLen = config.maxSeqLen;
        int batchSize = config.batchSize;
        int accumulationSteps = config.accumulationSteps;
        int totalBatches = config.totalBatches;
        float maxLR = config.maxLR;
        float minLR = config.minLR;
        int warmupBatches = config.warmupBatches;

        if (currentBatch < totalBatches) {
            System.out.println("\n--- Training (batch " + currentBatch + " to " + totalBatches + ") ---");
            System.out.printf("Batch size: %d | Accumulation: %d | Effective batch: %d%n",
                batchSize, accumulationSteps, batchSize * accumulationSteps);
            System.out.println("[HEARTBEAT] Starting batch 0... (first batch may take 60-120s on CPU)");
            System.out.flush();
            
            long startTime = System.nanoTime();
            Random rand = new Random();
            double movingAvgLoss = 0;
            double smoothing = 0.98;

            // BPE special token IDs
            int asstTokenId = 259;
            int eosTokenId = 257;
            int bosTokenId = 256;

            for (int i = currentBatch; i < totalBatches; i++) {
                // LR Schedule: Cosine Decay with Warmup
                float learningRate;
                if (i < warmupBatches) {
                    learningRate = minLR + (maxLR - minLR) * ((float) i / warmupBatches);
                } else {
                    float progress = (float)(i - warmupBatches) / (totalBatches - warmupBatches);
                    learningRate = minLR + 0.5f * (maxLR - minLR) * (1.0f + (float) Math.cos(Math.PI * progress));
                }
                
                // --- Gradient Accumulation Loop ---
                float batchLoss = 0;

                for (int acc = 0; acc < accumulationSteps; acc++) {
                    int[][] batchInputs = new int[batchSize][seqLen];
                    int[][] batchTargets = new int[batchSize][seqLen];
                    float[][] batchMasks = new float[batchSize][seqLen];

                    for (int b = 0; b < batchSize; b++) {
                        int startIdx = rand.nextInt(Math.max(1, encoded.length - seqLen - 1));
                        boolean maskOn = false;
                        boolean anyMasked = false;
                        for (int k = 0; k < seqLen; k++) {
                            int currentToken = encoded[startIdx + k];
                            int targetToken = encoded[startIdx + k + 1];

                            batchInputs[b][k] = currentToken;
                            batchTargets[b][k] = targetToken;

                            // Mask: train on assistant response tokens only
                            if (currentToken == asstTokenId) maskOn = true;
                            if (targetToken == eosTokenId || targetToken == bosTokenId) maskOn = false;

                            batchMasks[b][k] = maskOn ? 1.0f : 0.0f;
                            if (maskOn) anyMasked = true;
                        }
                        // CRITICAL: If no assistant tokens in this window, train on ALL tokens
                        // This prevents zero-gradient batches (which look like a "hang")
                        if (!anyMasked) {
                            for (int k = 0; k < seqLen; k++) batchMasks[b][k] = 1.0f;
                        }
                    }

                    // Heartbeat: print progress within the accumulation loop
                    if (i == currentBatch && acc == 0) {
                        System.out.print("  [acc 1/" + accumulationSteps + "] forward+backward... ");
                        System.out.flush();
                    }

                    batchLoss += transformer.accumulateGradients(batchInputs, batchTargets, batchMasks);

                    if (i == currentBatch) {
                        System.out.println("acc " + (acc + 1) + " done.");
                        System.out.flush();
                    }
                }
                
                // Update weights
                int effectiveBatchSize = batchSize * accumulationSteps;
                transformer.updateWeights(learningRate / effectiveBatchSize);
                transformer.cleanup();
                
                float loss = batchLoss / accumulationSteps;
                movingAvgLoss = (movingAvgLoss == 0) ? loss : (smoothing * movingAvgLoss + (1 - smoothing) * loss);
                
                if ((i + 1) % config.logInterval == 0) {
                    long now = System.nanoTime();
                    double elapsedSec = (now - startTime) / 1e9;
                    double batchSpeed = (i + 1 - currentBatch) / elapsedSec;
                    double etaSec = (totalBatches - (i + 1)) / (batchSpeed + 0.00001);
                    
                    int hrs = (int) (etaSec / 3600);
                    int mins = (int) ((etaSec % 3600) / 60);
                    
                    stats.batch = i + 1;
                    stats.totalBatches = totalBatches;
                    stats.loss = movingAvgLoss;
                    stats.ppl = Math.exp(movingAvgLoss);
                    stats.lr = learningRate;
                    stats.speed = batchSpeed;
                    stats.tokensPerSec = batchSpeed * batchSize * seqLen;
                    stats.lossHistory.add(new DataPoint(i + 1, movingAvgLoss));
                    stats.lrHistory.add(new DataPoint(i + 1, learningRate));
                    
                    DatabaseManager.saveMetric(i + 1, movingAvgLoss, Math.exp(movingAvgLoss), learningRate);

                    rt = Runtime.getRuntime();
                    stats.usedMemory = (rt.totalMemory() - rt.freeMemory()) / (1024 * 1024);
                    stats.maxMemory = rt.maxMemory() / (1024 * 1024);
                    
                    if (stats.os == null) {
                        stats.os = System.getProperty("os.name");
                        stats.cpu = Runtime.getRuntime().availableProcessors() + " Cores";
                        stats.javaVersion = System.getProperty("java.version");
                        stats.startTime = startTime / 1000000;
                        stats.dModel = config.dModel;
                        stats.layers = config.numLayers;
                        stats.heads = config.numHeads;
                        stats.seqLen = seqLen;
                    }

                    System.out.printf("Batch %d/%d | Loss: %.4f | PPL: %.1f | LR: %.6f | Speed: %.3f b/s | ETA: %02d:%02d | Mem: %dMB%n",
                        (i + 1), totalBatches, movingAvgLoss, Math.exp(movingAvgLoss), learningRate, batchSpeed, hrs, mins, stats.usedMemory);
                    System.out.flush();
                }

                if ((i + 1) % config.sampleInterval == 0) {
                    System.out.println("\n[Sample Generation]");
                    String samplePrompt = BOS + "\n" + USER_TAG + "\nWhat is a catalytic converter?\n" + ASST_TAG;
                    int[] sampleSeed = tokenizer.encode(samplePrompt);
                    int[] generated = transformer.generate(sampleSeed, 128, seqLen, 0.7f, eosTokenId);
                    String sample = tokenizer.decode(generated);
                    System.out.println(">>> " + sample);
                    System.out.print("Continuing training...\n");
                }

                // Checkpoint
                if ((i + 1) % config.checkpointInterval == 0) {
                    try {
                        transformer.saveModel(MODEL_PATH);
                        stats.lastSave = new java.text.SimpleDateFormat("HH:mm:ss").format(new Date());
                        Map<String, Integer> state = new HashMap<>();
                        state.put("batch", i + 1);
                        mapper.writeValue(new File(BATCH_STATE_PATH), state);
                    } catch (Exception e) {
                        System.err.println("Error saving checkpoint: " + e.getMessage());
                    }
                }
            }
            
            // Final save
            try {
                transformer.saveModel(MODEL_PATH);
                Map<String, Integer> state = new HashMap<>();
                state.put("batch", totalBatches);
                mapper.writeValue(new File(BATCH_STATE_PATH), state);
                stats.status = "Finished";
            } catch (Exception e) {
                System.err.println("Error saving final model: " + e.getMessage());
            }
        }

        // ===== Interactive Generation =====
        Scanner scanner = new Scanner(System.in);
        System.out.println("\n=== Dream Engine v3 Ready (BPE) ===");
        
        while (true) {
            System.out.print("\nInput: ");
            String input = scanner.nextLine().trim();
            if (input.equalsIgnoreCase("exit")) break;
            
            if (input.equalsIgnoreCase("save")) {
                transformer.saveModel(MODEL_PATH);
                continue;
            }
            
            if (input.isEmpty()) continue;

            int eosId = 257;
            String chatInput = BOS + "\n" + USER_TAG + "\n" + input + "\n" + ASST_TAG;
            int[] seedIds = tokenizer.encode(chatInput);
            int[] generatedIds = transformer.generate(seedIds, 256, config.maxSeqLen, 0.7f, eosId);
            String fullText = tokenizer.decode(generatedIds);
            
            System.out.println("Output:\n" + fullText);
        }
    }
    
    private static int[] encodeAndCacheDataset(String rawData, com.fasterxml.jackson.databind.ObjectMapper mapper) {
        System.out.println("Encoding dataset with BPE tokenizer...");
        long tStart = System.currentTimeMillis();
        int[] encoded = tokenizer.encode(rawData);
        System.out.println("Encoding took: " + ((System.currentTimeMillis() - tStart) / 1000) + "s");
        
        System.out.println("Saving encoded dataset to cache...");
        try {
            mapper.writeValue(new File(ENCODED_CACHE_PATH), encoded);
        } catch (Exception e) {
            System.err.println("Failed to save cache: " + e.getMessage());
        }
        return encoded;
    }

    private static void ensureDataset() {
        File datasetFile = new File("dataset.txt");
        if (!datasetFile.exists() || datasetFile.length() < 1000) {
             System.out.println("Warning: Dataset missing or small. Please ensure dataset.txt is present.");
        }
    }

    private static String cleanText(String text) {
        if (text == null) return "";
        
        // Normalize smart quotes and dashes
        text = text.replace("\u201C", "\"").replace("\u201D", "\"");
        text = text.replace("\u2018", "'").replace("\u2019", "'");
        text = text.replace("\u2014", "-").replace("\u2013", "-");
        text = text.replace("\u2026", "...");
        
        // Remove control characters (keep newlines/tabs)
        text = text.replaceAll("[^\\x20-\\x7E\\n\\r\\t]", "");
        
        // Normalize multiple newlines
        text = text.replaceAll("\\n{3,}", "\n\n");
        
        return text;
    }
}