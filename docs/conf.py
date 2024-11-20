# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Robust Gymnasium'
copyright = '2024, Anonymous authors' # Robust RL Team
author = 'Anonymous authors'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = []

extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    # 'myst_parser',
    # 'sphinx_copybutton',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    # 'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster' #'sphinx_book_theme' # 'sphinxawesome_theme' # 'sphinx_book_theme'
html_title = 'Robust-Gymnasium Tutorial'
html_baseurl = 'https://www.robust-gymnasium.com'
html_copy_source = False
# html_favicon = '_static/images/favicon.png'
html_context = {
    'conf_py_path': '/docs/',
    'display_github': False,
    'github_user': 'Anonymous authors',
    'github_repo': 'robustfety-gymnasium',
    'github_version': 'main',
    'slug': 'robust-gymnasium',
}

# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_css_files = [
    'custom.css',  # 引用自定义 CSS 文件
]

