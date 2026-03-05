import fitz
import numpy as np
from PIL import Image, ImageDraw

pdf_path = "/Users/gregory/Desktop/Lew Rothman Bio Scan 11-06-25.pdf"

doc = fitz.open(pdf_path)
page = doc[1]

zoom = 400 / 72
mat = fitz.Matrix(zoom, zoom)
pix = page.get_pixmap(matrix=mat, alpha=False)

img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
    pix.height, pix.width, 3
)

image = Image.fromarray(img_array)
draw = ImageDraw.Draw(image)

print("HEIGHT:", pix.height)
print("WIDTH:", pix.width)

for y in range(0, pix.height, 200):
    draw.line([(0, y), (pix.width, y)], fill=(255, 0, 0), width=1)

for x in range(0, pix.width, 200):
    draw.line([(x, 0), (x, pix.height)], fill=(255, 0, 0), width=1)

image.save("debug_grid.png")

print("Saved debug_grid.png")

doc.close()
