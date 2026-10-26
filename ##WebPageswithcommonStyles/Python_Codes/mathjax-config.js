// mathjax-config.js
window.MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
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

// Safety net: Force page to show after 3 seconds no matter what
setTimeout(() => {
    if (document.body.style.visibility === 'hidden') {
        document.body.style.visibility = 'visible';
    }
}, 3000);