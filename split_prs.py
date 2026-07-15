import os
import subprocess

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

# Update __init__.py and index.rst safely
def patch_init(add_imports, add_all):
    with open("src/langchain_google_alloydb_pg/__init__.py", "r") as f:
        content = f.read()
    
    # insert imports before __version__
    content = content.replace("from .version import __version__", 
                              add_imports + "from .version import __version__")
    
    # insert all
    content = content.replace('    "__version__",', '    "__version__",\n' + "\n".join(f'    "{x}",' for x in add_all))
    
    with open("src/langchain_google_alloydb_pg/__init__.py", "w") as f:
        f.write(content)

def patch_docs(add_docs):
    with open("docs/index.rst", "r") as f:
        content = f.read()
    
    docs_str = "\n".join(f"    langchain_google_alloydb_pg/{x}" for x in add_docs)
    content = content.replace("    langchain_google_alloydb_pg/model_manager", 
                              "    langchain_google_alloydb_pg/model_manager\n" + docs_str)
    
    with open("docs/index.rst", "w") as f:
        f.write(content)

def main():
    run("git fetch upstream")

    # PR 1
    run("git checkout -B feat/vector-optimizations upstream/main")
    run("""git checkout feat/alloydb-ai-features -- \
      src/langchain_google_alloydb_pg/async_vectorstore.py \
      src/langchain_google_alloydb_pg/vectorstore.py \
      src/langchain_google_alloydb_pg/engine.py \
      src/langchain_google_alloydb_pg/indexes.py \
      tests/test_async_vectorstore.py \
      tests/test_vectorstore.py \
      tests/test_engine.py \
      tests/test_indexes.py""")
    run("git add .")
    run("git commit -m 'feat: vector index & columnar engine optimizations'")
    run("git push -f origin feat/vector-optimizations")
    try:
        run('gh pr create --title "feat: Vector Index & Columnar Engine Optimizations" --body "Separated PR 1 out of 3."')
    except subprocess.CalledProcessError:
        print("PR 1 might already exist or failed.")

    # PR 2
    run("git checkout -B feat/ai-tools upstream/main")
    run("""git checkout feat/alloydb-ai-features -- \
      src/langchain_google_alloydb_pg/tools.py \
      src/langchain_google_alloydb_pg/document_compressor.py \
      tests/test_tools.py \
      tests/test_document_compressor.py \
      docs/langchain_google_alloydb_pg/tools.rst \
      docs/langchain_google_alloydb_pg/document_compressor.rst""")
    run("git checkout upstream/main -- src/langchain_google_alloydb_pg/__init__.py docs/index.rst")
    patch_init("from .document_compressor import AlloyDBDocumentCompressor\nfrom .tools import AlloyDBIfTool, AlloyDBSentimentTool, AlloyDBSummaryTool\n", 
               ["AlloyDBDocumentCompressor", "AlloyDBIfTool", "AlloyDBSentimentTool", "AlloyDBSummaryTool"])
    patch_docs(["document_compressor", "tools"])
    run("git add .")
    run("git commit -m 'feat: AI Tools & GenAI Functions'")
    run("git push -f origin feat/ai-tools")
    try:
        run('gh pr create --title "feat: AI Tools & GenAI Functions" --body "Separated PR 2 out of 3."')
    except subprocess.CalledProcessError:
        print("PR 2 might already exist or failed.")

    # PR 3
    run("git checkout -B feat/nl2sql-embeddings upstream/main")
    run("""git checkout feat/alloydb-ai-features -- \
      src/langchain_google_alloydb_pg/toolkit.py \
      src/langchain_google_alloydb_pg/embeddings.py \
      tests/test_toolkit.py \
      tests/test_embeddings.py \
      docs/langchain_google_alloydb_pg/toolkit.rst""")
    run("git checkout upstream/main -- src/langchain_google_alloydb_pg/__init__.py docs/index.rst")
    patch_init("from .toolkit import AlloyDBNL2SQLTool, AlloyDBToolkit\n", ["AlloyDBNL2SQLTool", "AlloyDBToolkit"])
    patch_docs(["toolkit"])
    run("git add .")
    run("git commit -m 'feat: Natural Language SQL Toolkit & Embeddings'")
    run("git push -f origin feat/nl2sql-embeddings")
    try:
        run('gh pr create --title "feat: Natural Language SQL Toolkit & Embeddings" --body "Separated PR 3 out of 3."')
    except subprocess.CalledProcessError:
        print("PR 3 might already exist or failed.")

    # Finally checkout back to the original branch
    run("git checkout feat/alloydb-ai-features")

if __name__ == "__main__":
    main()
