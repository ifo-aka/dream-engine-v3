import java.io.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * Byte-level tokenizer: Each byte (0-255) = token ID
 * Vocab: 260 tokens (256 bytes + 4 special tokens)
 */
public class ByteTokenizer {
    private static final int VOCAB_SIZE = 260;
    private static final int BOS_ID = 256;
    private static final int EOS_ID = 257;
    private static final int USER_ID = 258;
    private static final int ASSISTANT_ID = 259;
    
    private static final Map<String, Integer> STRING_TO_ID = new HashMap<>();
    private static final Map<Integer, String> ID_TO_STRING = new HashMap<>();
    
    static {
        STRING_TO_ID.put("<|bos|>", BOS_ID);
        STRING_TO_ID.put("<|eos|>", EOS_ID);
        STRING_TO_ID.put("<|user|>", USER_ID);
        STRING_TO_ID.put("<|assistant|>", ASSISTANT_ID);
        
        for (Map.Entry<String, Integer> entry : STRING_TO_ID.entrySet()) {
            ID_TO_STRING.put(entry.getValue(), entry.getKey());
        }
    }
    
    public int getVocabSize() {
        return VOCAB_SIZE;
    }
    
    public int[] encode(String text) {
        if (text == null || text.isEmpty()) return new int[0];
        
        // Split by special tokens using a regex that captures the delimiters
        String regex = "(<\\|bos\\|>|<\\|eos\\|>|<\\|user\\|>|<\\|assistant\\|>)";
        String[] parts = text.split(regex, -1);
        
        // We also need to find the actual tags to keep order
        java.util.regex.Matcher matcher = java.util.regex.Pattern.compile(regex).matcher(text);
        List<String> tags = new ArrayList<>();
        while (matcher.find()) {
            tags.add(matcher.group());
        }
        
        List<Integer> allTokens = new ArrayList<>();
        for (int i = 0; i < parts.length; i++) {
            // Encode the text part as raw bytes
            if (!parts[i].isEmpty()) {
                byte[] bytes = parts[i].getBytes(StandardCharsets.UTF_8);
                for (byte b : bytes) {
                    allTokens.add(b & 0xFF);
                }
            }
            // Add the tag that followed this part
            if (i < tags.size()) {
                allTokens.add(STRING_TO_ID.get(tags.get(i)));
            }
        }
        
        int[] result = new int[allTokens.size()];
        for (int i = 0; i < allTokens.size(); i++) result[i] = allTokens.get(i);
        return result;
    }
    
    public String decode(int[] tokens) {
        if (tokens == null) return "";
        StringBuilder sb = new StringBuilder();
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        
        for (int token : tokens) {
            if (token >= 0 && token <= 255) {
                baos.write(token);
            } else {
                // Flush pending bytes
                if (baos.size() > 0) {
                    sb.append(new String(baos.toByteArray(), StandardCharsets.UTF_8));
                    baos.reset();
                }
                // Append special token string
                String tag = ID_TO_STRING.get(token);
                if (tag != null) {
                    sb.append(tag);
                }
            }
        }
        
        if (baos.size() > 0) {
            sb.append(new String(baos.toByteArray(), StandardCharsets.UTF_8));
        }
        
        return sb.toString();
    }
    
    public void save(String path) throws IOException {
        // Byte tokenizer has no state to save
        Map<String, Integer> meta = new HashMap<>();
        meta.put("vocabSize", VOCAB_SIZE);
        new ObjectMapper().writeValue(new File(path), meta);
    }
    
    public void load(String path) throws IOException {
        // No state to load for byte tokenizer
    }
}
