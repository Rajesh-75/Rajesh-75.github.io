import os
import re

# Template now includes your specific MathJax config + the MathJax library loader
NEW_HEAD_TEMPLATE = """<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="style.css">
    <title>{TITLE}</title>
    <script type="text/javascript" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']]
            }}
        }};
    </script>
</head>"""

def clean_html_files():
    for filename in os.listdir('.'):
        if filename.endswith(".html"):
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # 1. Capture original title
            title_search = re.search(r'<title>(.*?)</title>', content, flags=re.IGNORECASE | re.DOTALL)
            original_title = title_search.group(1).strip() if title_search else "Machine Learning Tutorial"

            # 2. Remove old internal <style> blocks
            content = re.sub(r'<style.*?>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

            # 3. Create the customized head (Injecting the Title into the Template)
            custom_head = NEW_HEAD_TEMPLATE.format(TITLE=original_title)

            # 4. Replace the old <head> with the new one (includes MathJax)
            content = re.sub(r'<head.*?>.*?</head>', custom_head, content, flags=re.DOTALL | re.IGNORECASE)

            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"Standardized with MathJax: {filename}")

if __name__ == "__main__":
    print("Adding MathJax and unifying styles across all pages...")
    clean_html_files()
    print("\nDone! All math formulas should now render correctly.")
