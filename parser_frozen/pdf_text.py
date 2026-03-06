import pdfplumber

def extract_disease_names(pdf_path):

    diseases = []

    with pdfplumber.open(pdf_path) as pdf:

        page = pdf.pages[1]

        text = page.extract_text()

        lines = text.split("\n")

        for line in lines:

            clean = line.strip()

            if len(clean) < 4:
                continue

            if len(clean) > 60:
                continue

            diseases.append(clean)

    return diseases