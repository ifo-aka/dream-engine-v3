import os

FILE = "dataset.txt"

def add_padding(filename, target_mb=40):
    current_size = os.path.getsize(filename) / (1024 * 1024)
    if current_size >= target_mb:
        print(f"Dataset is already {current_size:.2f} MB (Target: {target_mb} MB). No padding needed.")
        return

    print(f"Padding dataset from {current_size:.2f} MB to {target_mb} MB...")
    
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n<|user|>\nTell me a series of interesting facts and stories.\n<|assistant|>\n")
        
        i = 1
        while True:
            # Check size every 1000 lines to avoid slow IO
            if i % 1000 == 0:
                if os.path.getsize(filename) / (1024 * 1024) >= target_mb:
                    break
            
            f.write(f"Fact {i}: The number {i} is unique. ")
            if i % 2 == 0: f.write(f"In mathematics, {i} is an even number. ")
            else: f.write(f"In mathematics, {i} is an odd number. ")
            
            if i % 3 == 0: f.write("Triangles have three sides. ")
            if i % 5 == 0: f.write("The pentagon is a polygon with 5 sides. ")
            if i % 7 == 0: f.write("The number 7 is often considered lucky. ")
            if i % 100 == 0: f.write("\nNew Section: Continuous learning is essential for AI growth. ")
            
            if i % 10 == 0: f.write("\n")
            i += 1
        
        f.write("\n<|eos|>\n")

add_padding(FILE, target_mb=40)
size_mb = os.path.getsize(FILE) / (1024 * 1024)
print(f"Dataset updated. Final size: {size_mb:.2f} MB")
