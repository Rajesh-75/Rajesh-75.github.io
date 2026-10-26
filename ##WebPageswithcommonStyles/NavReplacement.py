import os

# Your Master Nav Code
nav_code = """
<nav style="...">
    </nav>
"""

for filename in os.listdir("."):
    if filename.endswith(".html"):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert nav if not already there
        if '<nav' not in content:
            new_content = content.replace('<body>', f'<body>\n{nav_code}')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(new_content)
        print(f"Updated: {filename}")