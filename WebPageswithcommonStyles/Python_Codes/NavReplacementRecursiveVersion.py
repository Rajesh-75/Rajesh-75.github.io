import os

# Your Master Nav Code
# TIP: If your files are in different folders, ensure your hrefs 
# point to the correct locations!
nav_code = """
<nav style="background: #2c3e50; padding: 10px; text-align: center; margin-bottom: 20px; border-radius: 4px;">
    <a href="/index.html" style="color: white; margin: 0 15px; text-decoration: none; font-weight: bold;">Home</a>
    <a href="/Probability/ND.html" style="color: white; margin: 0 15px; text-decoration: none; font-weight: bold;">Normal Distribution</a>
    <a href="/Probability/MLE.html" style="color: white; margin: 0 15px; text-decoration: none; font-weight: bold;">MLE</a>
</nav>
"""

# The root directory where your tutorial starts
root_dir = "." 

for subdir, dirs, files in os.walk(root_dir):
    for filename in files:
        if filename.endswith(".html"):
            file_path = os.path.join(subdir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if nav is already there to avoid double-pasting
            if '<nav' not in content:
                # Find the <body> tag and insert the nav immediately after
                if '<body>' in content:
                    new_content = content.replace('<body>', f'<body>\n{nav_code}')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    print(f"✅ Successfully updated: {file_path}")
                else:
                    print(f"⚠️ Skipped: {file_path} (No <body> tag found)")
            else:
                print(f"ℹ️ Already exists: {file_path}")