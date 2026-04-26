import java.util.*;
import java.util.regex.*;

public class Tokenizer {
    private Map<String, Integer> tokenToId = new HashMap<>();
    private Map<Integer, String> idToToken = new HashMap<>();
    private int vocabSize = 0;
    private static final String UNK = "<UNK>";

    public void fit(String text) {
        Map<String, Integer> freq = new HashMap<>();
        // Matcher for words, spaces, and punctuation
        Matcher m = Pattern.compile("(\\w+|\\s+|[^\\w\\s])").matcher(text);
        while (m.find()) {
            String token = m.group();
            freq.put(token, freq.getOrDefault(token, 0) + 1);
        }

        tokenToId.clear();
        idToToken.clear();
        vocabSize = 0;
        
        addToken(UNK);

        // Sort by frequency and take top 5000 tokens (Rich dictionary)
        List<Map.Entry<String, Integer>> list = new ArrayList<>(freq.entrySet());
        list.sort((a,b) -> b.getValue().compareTo(a.getValue()));

        int limit = Math.min(list.size(), 5000);
        for(int i=0; i<limit; i++) {
            addToken(list.get(i).getKey());
        }
    }

    private void addToken(String t) {
        if (!tokenToId.containsKey(t)) {
            tokenToId.put(t, vocabSize);
            idToToken.put(vocabSize, t);
            vocabSize++;
        }
    }

    public int[] encode(String text) {
        List<Integer> ids = new ArrayList<>();
        Matcher m = Pattern.compile("(\\w+|\\s|[^\\w\\s])").matcher(text);
        while (m.find()) {
            String token = m.group();
            ids.add(tokenToId.getOrDefault(token, tokenToId.get(UNK)));
        }
        int[] result = new int[ids.size()];
        for(int i=0; i<ids.size(); i++) result[i] = ids.get(i);
        return result;
    }

    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            sb.append(idToToken.getOrDefault(id, ""));
        }
        return sb.toString();
    }

    public void save(String path) {
        try (java.io.DataOutputStream out = new java.io.DataOutputStream(new java.io.FileOutputStream(path))) {
            out.writeInt(vocabSize);
            out.writeInt(tokenToId.size());
            for (Map.Entry<String, Integer> entry : tokenToId.entrySet()) {
                out.writeUTF(entry.getKey());
                out.writeInt(entry.getValue());
            }
            System.out.println("Tokenizer state saved to " + path);
        } catch (Exception e) {
            System.err.println("Error saving tokenizer: " + e.getMessage());
        }
    }

    public void load(String path) {
        try (java.io.DataInputStream in = new java.io.DataInputStream(new java.io.FileInputStream(path))) {
            this.vocabSize = in.readInt();
            int size = in.readInt();
            tokenToId.clear();
            idToToken.clear();
            for (int i = 0; i < size; i++) {
                String token = in.readUTF();
                int id = in.readInt();
                tokenToId.put(token, id);
                idToToken.put(id, token);
            }
            System.out.println("Tokenizer state loaded from " + path);
        } catch (Exception e) {
            System.err.println("Error loading tokenizer: " + e.getMessage());
        }
    }

    public int getVocabSize() { return vocabSize; }
}
