import fitz

pdf_path = "/Users/gregory/Desktop/Lew Rothman Bio Scan 11-06-25.pdf"

doc = fitz.open(pdf_path)

page = doc[1]  # Disease page (page index 1)
zoom = 400 / 72
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat, alpha=False)

print("HEIGHT:", pix.height)
print("WIDTH:", pix.width)

doc.close()
