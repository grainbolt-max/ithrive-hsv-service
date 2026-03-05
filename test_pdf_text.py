from parser.pdf_text import extract_disease_names

PDF_FILE = "sample.pdf"

def main():

    names = extract_disease_names(PDF_FILE)

    print("Detected disease names:")
    print(names)

if __name__ == "__main__":
    main()