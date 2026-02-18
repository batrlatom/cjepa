
import sys
import os

try:
    import pypdf
except ImportError:
    print("pypdf not found")
    sys.exit(1)

pdf_path = "/home/tomas/.gemini/antigravity/brain/3f2280ed-944e-4479-b339-8c80a4a99041/.tempmediaStorage/e097ae535caf1b42.pdf"
output_path = "/home/tomas/.gemini/antigravity/scratch/paper_text.txt"

try:
    reader = pypdf.PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    with open(output_path, "w") as f:
        f.write(text)
    print(f"Text written to {output_path}")
except Exception as e:
    print(f"Error reading PDF: {e}")
