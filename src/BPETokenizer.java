//BPETokenizer.java
import java.util.*;
import java.io.*;
import java.nio.charset.StandardCharsets;

public class BPETokenizer {
    private Map<Integer, String> vocab;
    private Map<String, Integer> reverseVocab;
    private List<Pair> merges;
    private int vocabSize;
    private Set<String> specialTokens;
    private Map<String, Integer> specialToId;
    private Map<Integer, Character> byteToUnicode;
    private Map<Character, Integer> unicodeToByte;

    public static final String BOS = "<|bos|>";
    public static final String EOS = "<|eos|>";
    public static final String USER = "<|user|>";
    public static final String ASST = "<|assistant|>";

    // --- Jackson Serialization ---
    
    // Helper to rebuild transient state after JSON load
    public void rebuildState() {
        this.reverseVocab = new HashMap<>();
        for (Map.Entry<Integer, String> e : vocab.entrySet()) {
            reverseVocab.put(e.getValue(), e.getKey());
        }
        
        this.specialTokens = new HashSet<>(Arrays.asList(BOS, EOS, USER, ASST));
        this.specialToId = new HashMap<>();
        this.byteToUnicode = new HashMap<>();
        this.unicodeToByte = new HashMap<>();
        
        // Re-init mappings (Same as constructor)
        for (int i = 33; i <= 126; i++) addMapping(i, (char)i);
        for (int i = 161; i <= 172; i++) addMapping(i, (char)i);
        for (int i = 174; i <= 255; i++) addMapping(i, (char)i);
        int unicodePtr = 256;
        for (int i = 0; i < 256; i++) {
            if (!byteToUnicode.containsKey(i)) {
                while (isBadUnicode(unicodePtr)) unicodePtr++;
                addMapping(i, (char)unicodePtr);
                unicodePtr++;
            }
        }
        
        // Re-map special tokens
        int sid = 256;
        for (String st : Arrays.asList(BOS, EOS, USER, ASST)) {
            // Check if they exist in loaded vocab, else add? 
            // Usually they should be in vocab if trained.
            // Just ensure specialToId is populated
            if (reverseVocab.containsKey(st)) {
                specialToId.put(st, reverseVocab.get(st));
            }
            sid++;
        }
    }

    public void save(String filename) {
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            mapper.enable(com.fasterxml.jackson.databind.SerializationFeature.INDENT_OUTPUT);
            mapper.writeValue(new File(filename), this);
            System.out.println("Tokenizer (JSON) saved to " + filename);
        } catch (Exception e) {
            System.err.println("Error saving tokenizer: " + e.getMessage());
        }
    }

    public void load(String filename) {
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            BPETokenizer loaded = mapper.readValue(new File(filename), BPETokenizer.class);
            
            this.vocab = loaded.vocab;
            this.merges = loaded.merges;
            this.vocabSize = loaded.vocabSize;
            
            rebuildState();
            System.out.println("Tokenizer loaded from " + filename);
        } catch (Exception e) {
            System.err.println("Error loading tokenizer: " + e.getMessage());
        }
    }

    public void loadLegacyBinary(String filename) {
        try (DataInputStream dis = new DataInputStream(new FileInputStream(filename))) {
            vocabSize = dis.readInt();
            int numMerges = dis.readInt();
            vocab = new HashMap<>();
            reverseVocab = new HashMap<>();
            for (int i = 0; i < vocabSize; i++) {
                int len = dis.readInt();
                byte[] b = new byte[len];
                dis.readFully(b);
                String s = new String(b, StandardCharsets.UTF_8);
                vocab.put(i, s);
                reverseVocab.put(s, i);
            }
            merges = new ArrayList<>();
            for (int i = 0; i < numMerges; i++) {
                merges.add(new Pair(dis.readInt(), dis.readInt()));
            }
            rebuildState();
            System.out.println("Loaded legacy binary tokenizer from " + filename);
        } catch (Exception e) {
            System.err.println("Failed to load legacy binary tokenizer: " + e.getMessage());
        }
    }

    // Getters and Setters for Jackson
    public Map<Integer, String> getVocab() { return vocab; }
    public void setVocab(Map<Integer, String> vocab) { this.vocab = vocab; }

    public List<Pair> getMerges() { return merges; }
    public void setMerges(List<Pair> merges) { this.merges = merges; }

    public int getVocabSize() { return vocabSize; }
    public void setVocabSize(int vocabSize) { this.vocabSize = vocabSize; }

    public static class Pair {
        public int left, right;
        public Pair() {} // Jackson needs this
        public Pair(int l, int r) { this.left = l; this.right = r; }
        
        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (o == null || getClass() != o.getClass()) return false;
            Pair pair = (Pair) o;
            return left == pair.left && right == pair.right;
        }

        @Override
        public int hashCode() { return Objects.hash(left, right); }
        
        @Override
        public String toString() { return left + "+" + right; }
    }

    public BPETokenizer() {
        this.vocab = new HashMap<>();
        this.reverseVocab = new HashMap<>();
        this.merges = new ArrayList<>();
        this.specialTokens = new HashSet<>(Arrays.asList(BOS, EOS, USER, ASST));
        this.specialToId = new HashMap<>();
        this.byteToUnicode = new HashMap<>();
        this.unicodeToByte = new HashMap<>();
        
        // 1. Initialize Byte-to-Unicode mapping
        for (int i = 33; i <= 126; i++) addMapping(i, (char)i);
        for (int i = 161; i <= 172; i++) addMapping(i, (char)i);
        for (int i = 174; i <= 255; i++) addMapping(i, (char)i);
        int unicodePtr = 256;
        for (int i = 0; i < 256; i++) {
            if (!byteToUnicode.containsKey(i)) {
                while (isBadUnicode(unicodePtr)) unicodePtr++;
                addMapping(i, (char)unicodePtr);
                unicodePtr++;
            }
        }
        
        // 2. Initialize Base Vocab
        for (int i = 0; i < 256; i++) {
            String s = String.valueOf(byteToUnicode.get(i));
            vocab.put(i, s);
            reverseVocab.put(s, i);
        }
        
        // 3. Initialize Special Tokens
        int sid = 256;
        for (String st : Arrays.asList(BOS, EOS, USER, ASST)) {
            vocab.put(sid, st);
            reverseVocab.put(st, sid);
            specialToId.put(st, sid);
            sid++;
        }
        
        this.vocabSize = 260; // Base bytes + Specials
    }

    private void addMapping(int b, char c) {
        byteToUnicode.put(b, c);
        unicodeToByte.put(c, b);
    }

    private boolean isBadUnicode(int c) {
        // Simple filter for characters that might cause issues in some terminals or regex
        return Character.isISOControl(c) || Character.isWhitespace(c);
    }

    private List<String> splitBySpecialTokens(String text) {
        List<String> parts = new ArrayList<>();
        if (text == null || text.isEmpty()) return parts;
        
        int i = 0;
        while (i < text.length()) {
            boolean found = false;
            for (String st : specialTokens) {
                if (text.startsWith(st, i)) {
                    parts.add(st);
                    i += st.length();
                    found = true;
                    break;
                }
            }
            if (!found) {
                int start = i;
                while (i < text.length()) {
                    boolean hitSpecial = false;
                    for (String st : specialTokens) {
                        if (text.startsWith(st, i)) { hitSpecial = true; break; }
                    }
                    if (hitSpecial) break;
                    i++;
                }
                parts.add(text.substring(start, i));
            }
        }
        return parts;
    }

    public void train(String text, int targetVocabSize) {
        System.out.println("Processing text for BPE training...");
        List<String> chunks = splitBySpecialTokens(text);
        
        List<List<Integer>> tokenizedChunks = new ArrayList<>();
        for (String chunk : chunks) {
            if (specialTokens.contains(chunk)) continue;
            
            // Map raw bytes to Unicode-string base tokens
            byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
            List<Integer> ids = new ArrayList<>();
            for (byte b : bytes) {
                ids.add(Byte.toUnsignedInt(b));
            }
            tokenizedChunks.add(ids);
        }

        int currentMaxId = 259; // Last special token ID
        int mergesToPerform = targetVocabSize - vocab.size();
        
        System.out.println("Starting merges. Initial vocab: " + vocab.size());
        
        for (int m = 0; m < mergesToPerform; m++) {
            Map<Pair, Integer> stats = new HashMap<>();
            for (List<Integer> chunk : tokenizedChunks) {
                for (int i = 0; i < chunk.size() - 1; i++) {
                    Pair p = new Pair(chunk.get(i), chunk.get(i+1));
                    stats.put(p, stats.getOrDefault(p, 0) + 1);
                }
            }
            
            if (stats.isEmpty()) break;
            
            Pair bestPair = Collections.max(stats.entrySet(), Map.Entry.comparingByValue()).getKey();
            int newTokenId = ++currentMaxId;
            
            String s1 = vocab.get(bestPair.left);
            String s2 = vocab.get(bestPair.right);
            String combined = s1 + s2;
            
            vocab.put(newTokenId, combined);
            reverseVocab.put(combined, newTokenId);
            merges.add(bestPair);
            
            for (int i = 0; i < tokenizedChunks.size(); i++) {
                tokenizedChunks.set(i, applyMerge(tokenizedChunks.get(i), bestPair, newTokenId));
            }
            
            if ((m + 1) % 100 == 0) System.out.println("Merge " + (m+1) + ": " + combined + " (ID: " + newTokenId + ")");
        }
        this.vocabSize = vocab.size();
    }

    private List<Integer> applyMerge(List<Integer> ids, Pair pair, int idx) {
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < ids.size(); i++) {
            if (i < ids.size() - 1 && ids.get(i) == pair.left && ids.get(i + 1) == pair.right) {
                result.add(idx);
                i++;
            } else {
                result.add(ids.get(i));
            }
        }
        return result;
    }

    public int[] encode(String text) {
        List<String> chunks = splitBySpecialTokens(text);
        List<Integer> finalIds = new ArrayList<>();
        
        for (String chunk : chunks) {
            if (specialTokens.contains(chunk)) {
                finalIds.add(specialToId.get(chunk));
                continue;
            }
            
            byte[] bytes = chunk.getBytes(StandardCharsets.UTF_8);
            List<Integer> ids = new ArrayList<>();
            for (byte b : bytes) ids.add(Byte.toUnsignedInt(b));
            
            // Apply learned merges in order
            for (Pair p : merges) {
                ids = applyMerge(ids, p, reverseVocab.get(vocab.get(p.left) + vocab.get(p.right)));
            }
            finalIds.addAll(ids);
        }
        
        int[] result = new int[finalIds.size()];
        for (int i = 0; i < finalIds.size(); i++) result[i] = finalIds.get(i);
        return result;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String s = vocab.get(id);
            if (s != null) sb.append(s);
        }
        
        // Convert Unicode symbols back to bytes
        String unicodeStr = sb.toString();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        for (int i = 0; i < unicodeStr.length(); i++) {
            char c = unicodeStr.charAt(i);
            Integer b = unicodeToByte.get(c);
            if (b != null) baos.write(b);
            else {
                // Should not happen for text chunks, but special tokens are handled as strings
                byte[] specialBytes = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                try { baos.write(specialBytes); } catch (Exception e) {}
            }
        }
        
        // Special mapping for our special tokens which are multi-char strings in vocab
        // The above loop handles single-char Unicode mappings. 
        // We need a more robust way to handle the mix.
        
        return handleMixedDecoding(ids);
    }

    private String handleMixedDecoding(int[] ids) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        for (int id : ids) {
            String s = vocab.get(id);
            if (s == null) continue;
            
            if (specialTokens.contains(s)) {
                try { baos.write(s.getBytes(StandardCharsets.UTF_8)); } catch (Exception e) {}
            } else {
                for (int i = 0; i < s.length(); i++) {
                    char c = s.charAt(i);
                    Integer b = unicodeToByte.get(c);
                    if (b != null) baos.write(b);
                }
            }
        }
        return new String(baos.toByteArray(), StandardCharsets.UTF_8);
    }




}

