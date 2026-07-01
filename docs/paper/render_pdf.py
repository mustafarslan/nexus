import fitz
import sys

def analyze(pdf_path):
    print(f"Analyzing {pdf_path}")
    doc = fitz.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=150)
    # We can't see images, but we can list paths and text to see intersections manually
    paths = page.get_drawings()
    text_blocks = page.get_text("blocks")
    
    # We'll just look for texts that are suspiciously close.
    for b in text_blocks:
        print(f"TEXT {b[:4]}: {b[4].strip().replace('\n', ' ')}")

analyze("test_fig1.pdf")
print("----------------")
analyze("test_fig.pdf")
