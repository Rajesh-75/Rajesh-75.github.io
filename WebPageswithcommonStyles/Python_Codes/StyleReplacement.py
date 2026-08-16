import os
import re

# The standardized head block you want
NEW_HEAD = """<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="style.css">
    <title>Machine Learning Tutorial</title>
</head>"""

def clean_html_files():
    # Loop through every file in the current directory
    for filename in os.listdir('.'):
        if filename.endswith(".html"):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Remove the entire <style>...</style> block (multi-line)
            content = re.sub(r'<style.*?>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

            # 2. Replace the old <head>...</head> with the new one
            content = re.sub(r'<head.*?>.*?</head>', NEW_HEAD, content, flags=re.DOTALL | re.IGNORECASE)

            # Write the cleaned content back to the file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Successfully standardized: {filename}")

if __name__ == "__main__":
    print("Starting standardization process...")
    # It's always a good idea to have a backup!
    clean_html_files()
    print("Done! All HTML files are now linked to style.css.")