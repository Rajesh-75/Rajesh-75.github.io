import os
import re

# The exact working MathJax block provided by you
TARGET_MATHJAX_BLOCK = """    <script>
        window.MathJax = {
            tex: {
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
            },
            options: {
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            },
            startup: {
                pageReady: () => {
                    return MathJax.startup.defaultPageReady().then(() => {
                        document.body.style.visibility = 'visible';
                    });
                }
            }
        };
    </script>
    <script type="text/javascript" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
    </script>"""

def update_html_files(directory="."):
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            filepath = os.path.join(directory, filename)
            
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # Rule: If the exact target script block is already present, do not change it
            if "window.MathJax = {" in content and "document.body.style.visibility = 'visible';" in content:
                print(f"Skipped (already up to date): {filename}")
                continue

            # Remove old/conflicting MathJax configuration or external references
            # 1. Remove old mathjax-config.js reference if present
            content = re.sub(r'\s*<script\b[^>]*src=["\']mathjax-config\.js["\'][^>]*>\s*</script>', '', content)
            
            # 2. Remove older inline window.MathJax blocks
            content = re.sub(r'\s*<script\b[^>]*>\s*window\.MathJax\s*=\s*\{.*?\};\s*</script>', '', content, flags=re.DOTALL)
            
            # 3. Remove old MathJax CDN script tags
            content = re.sub(r'\s*<script\b[^>]*src=["\'][^"\']*mathjax@3[^"\']*["\'][^>]*>\s*</script>', '', content)

            # Insert the new working block right before </head>
            if '</head>' in content:
                content = content.replace('</head>', f'\n{TARGET_MATHJAX_BLOCK}\n</head>')
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Updated successfully: {filename}")
            else:
                print(f"Warning: </head> tag not found in {filename}, skipped.")

if __name__ == "__main__":
    print("Starting MathJax script updater...")
    update_html_files()
    print("Done!")
