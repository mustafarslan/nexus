import fitz

def check(pdf_path):
    print(f"--- {pdf_path} ---")
    doc = fitz.open(pdf_path)
    page = doc[0]
    
    blocks = page.get_text("dict")["blocks"]
    for b1 in blocks:
        if b1["type"] != 0: continue
        r1 = fitz.Rect(b1["bbox"])
        for b2 in blocks:
            if b2["type"] != 0: continue
            if b1 == b2: continue
            r2 = fitz.Rect(b2["bbox"])
            ix = r1.intersect(r2)
            if not ix.is_empty and ix.get_area() > 0:
                t1 = "".join([l["spans"][0]["text"] for l in b1["lines"]])
                t2 = "".join([l["spans"][0]["text"] for l in b2["lines"]])
                print(f"TEXT OVERLAP: '{t1}' vs '{t2}' area={ix.get_area()}")
                
    drawings = page.get_drawings()
    for b in blocks:
        if b["type"] != 0: continue
        r1 = fitz.Rect(b["bbox"])
        t = "".join([l["spans"][0]["text"] for l in b["lines"]])
        
        for d in drawings:
            if d["fill"] is not None and d["color"] is None:
                continue
            r2 = d["rect"]
            ix = r1.intersect(r2)
            if not ix.is_empty and ix.get_area() > 0:
                print(f"TEXT/DRAW OVERLAP: '{t}' overlaps drawing at {r2}")

check("test_fig2.pdf")
