# Manual Dataset Creation Guide

If automatic downloads fail, manually create a larger dataset:

## Option 1: Download Books Manually

Visit these URLs in your browser and save as text files:

1. **Pride and Prejudice**: https://www.gutenberg.org/files/1342/1342-0.txt
2. **Alice in Wonderland**: https://www.gutenberg.org/files/11/11-0.txt  
3. **Frankenstein**: https://www.gutenberg.org/cache/epub/84/pg84.txt
4. **Sherlock Holmes**: https://www.gutenberg.org/files/1661/1661-0.txt
5. **Moby Dick**: https://www.gutenberg.org/files/2701/2701-0.txt

Then combine them:
```powershell
Get-Content book1.txt,book2.txt,book3.txt,book4.txt,book5.txt | Set-Content dataset.txt
```

## Option 2: Use Local Text Files

Collect any large text files you have (books, articles, etc.) and combine them:

```powershell
# Find all .txt files and combine
Get-ChildItem -Path . -Filter *.txt | Get-Content | Set-Content dataset.txt
```

**Minimum size**: 3MB (3000 KB)  
**Recommended**: 5-10MB for best results

## Option 3: Wikipedia Text (Advanced)

Download Wikipedia dumps from https://dumps.wikimedia.org/

Extract plain text using:
```bash
# Requires wikiextractor tool 
```

## Verify Dataset

Check size:
```powershell
(Get-Item dataset.txt).Length / 1KB
```

Should show > 3000 KB
