import os

root_dir = "." 
markers_to_remove = [
    
    "visibility: hidden;",
    "Hide body initially",
    "Load global MathJax configuration"
]

for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".html"):
      file_path = os.path.join(subdir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if 'mathjax-config.js' in content:
                lines = content.splitlines()
                new_lines = []
                for line in lines:
                    if any(marker in line for marker in markers_to_remove):
                        continue
                    new_lines.append(line)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(new_lines))
                print(f"🔄 Reverted: {file_path}")
