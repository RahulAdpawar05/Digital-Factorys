from flask import Flask, render_template, request, jsonify
import os
from utils.document_comparison import process_query_with_documents
import gunicorn

application = Flask(__name__)
app = application
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Predefined path to Document 1
DOC1_PATH = "uploads/MAHA RERA MODEL AGREEMENT FOR SALE.pdf"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare-documents/", methods=["POST"])
def compare_documents():
    # Get file and query from the form
    doc2 = request.files.get("doc2")
    query = request.form.get("query", "compare the doc2 with doc1, give the suggestion, good points and area of improvements in doc2 as compare to doc1")

    if not doc2:
        error_message = "Document 2 is required."
        return render_template("result.html", error=error_message)

    # Save uploaded Document 2 temporarily
    doc2_path = os.path.join(app.config["UPLOAD_FOLDER"], doc2.filename)
    doc2.save(doc2_path)

    # Process documents
    try:
        result = process_query_with_documents(DOC1_PATH, doc2_path, query)
        print("Final Result in Python:", result)  # Debugging
        return render_template("result.html", result=result)
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template("result.html", error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
