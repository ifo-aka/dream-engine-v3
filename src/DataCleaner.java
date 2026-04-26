import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;

public class DataCleaner {
    public static void main(String[] args) {
        System.out.println("=== Dataset Analysis & Cleaning Tool ===");
        String inputFile = "dataset.txt";
        String outputFile = "dataset_clean.txt";
        
        try {
            if (!Files.exists(Paths.get(inputFile))) {
                System.out.println("Error: " + inputFile + " not found.");
                return;
            }

            byte[] bytes = Files.readAllBytes(Paths.get(inputFile));
            String content = new String(bytes, StandardCharsets.UTF_8);
            
            System.out.println("Original Size: " + (content.length() / 1024) + " KB");
            
            // Analysis
            Map<Character, Integer> weirdChars = new HashMap<>();
            int totalWeird = 0;
            
            for (char c : content.toCharArray()) {
                if (c > 127 && c != '\n' && c != '\r' && c != '\t') {
                    weirdChars.put(c, weirdChars.getOrDefault(c, 0) + 1);
                    totalWeird++;
                }
            }
            
            System.out.println("\n--- Analysis Report ---");
            System.out.println("Non-ASCII Characters Found: " + totalWeird);
            if (totalWeird > 0) {
                System.out.println("Top offenders:");
                weirdChars.entrySet().stream()
                    .sorted((a, b) -> b.getValue().compareTo(a.getValue()))
                    .limit(10)
                    .forEach(e -> System.out.printf("  '%c' (U+%04X): %d%n", e.getKey(), (int)e.getKey(), e.getValue()));
            } else {
                System.out.println("File is already clean ASCII/UTF-8!");
            }

            // Cleaning
            System.out.println("\n--- Cleaning ---");
            String clean = content;
            
            // 1. Smart Quotes & Dashes
            clean = clean.replace("\u201C", "\"").replace("\u201D", "\""); // Smart double quotes
            clean = clean.replace("\u2018", "'").replace("\u2019", "'");   // Smart single quotes
            clean = clean.replace("\u2014", "-").replace("\u2013", "-");   // Em-dash, En-dash
            clean = clean.replace("\u2026", "...");                        // Ellipsis
            clean = clean.replace("\u00AB", "\"").replace("\u00BB", "\""); // Guillemets
            
            // 2. Remove remaining non-printable / weird characters
            // Keep: space (32) to tilde (126), plus newline, carriage return, tab
            clean = clean.replaceAll("[^\\x20-\\x7E\\n\\r\\t]", ""); 
            
            // 3. Normalize whitespace to max 2 newlines
            clean = clean.replaceAll("\\n{3,}", "\n\n");
            
            System.out.println("Cleaned Size: " + (clean.length() / 1024) + " KB");
            int removed = content.length() - clean.length();
            System.out.println("Removed " + removed + " junk characters.");

            if (content.length() == clean.length() && totalWeird == 0) {
                 System.out.println("File was clean. No changes made.");
            } else {
                Files.write(Paths.get(outputFile), clean.getBytes(StandardCharsets.UTF_8));
                System.out.println("Saved clean version to: " + outputFile);
                
                // Replace original
                Files.move(Paths.get(outputFile), Paths.get(inputFile), StandardCopyOption.REPLACE_EXISTING);
                System.out.println("Replaced original dataset.txt with clean version.");
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
