import fitz

def check_pdf(pdf_path):
    print(f"Checking {pdf_path} for overlaps...")
    doc = fitz.open(pdf_path)
    page = doc[0]
    words = page.get_text("words")
    # A word is (x0, y0, x1, y1, "word", block_no, line_no, word_no)
    
    overlaps = []
    for i in range(len(words)):
        w1 = words[i]
        r1 = fitz.Rect(w1[:4])
        for j in range(i+1, len(words)):
            w2 = words[j]
            # Ignore same block or if they are just adjacent in text
            # We want actual visual overlaps.
            r2 = fitz.Rect(w2[:4])
            
            # Inflate slightly to ignore touching boundaries
            # Actually, intersect gives the area.
            ix = r1.intersect(r2)
            if not ix.is_empty and ix.get_area() > 1.0: # arbitrary threshold for overlap
                overlaps.append((w1[4], w2[4], ix.get_area()))
                
    if overlaps:
        print("OVERLAPS DETECTED:")
        for o in overlaps[:10]:
            print(f"'{o[0]}' overlaps with '{o[1]}' (area {o[2]:.2f})")
    else:
        print("No significant overlaps detected.")
    print()

check_pdf("test_fig1.pdf")
check_pdf("test_fig.pdf")
