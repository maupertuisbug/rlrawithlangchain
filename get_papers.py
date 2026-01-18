import os
import requests
import pdfplumber
import requests
import fitz
import requests
import xml.etree.ElementTree as ET

links = [
    "https://arxiv.org/pdf/2402.15567",
    "https://arxiv.org/pdf/2304.04782",
    "https://arxiv.org/pdf/2110.06169",
    "https://arxiv.org/pdf/2209.14935",
    "https://arxiv.org/pdf/2102.11271",
]

def download_pdf(url, output_path, metadata):
    response = requests.get(url)
    response.raise_for_status() 
    doc = fitz.open(stream=response.content, filetype="pdf")
    doc.set_metadata(metadata)
    doc.save(output_path)
    doc.close()

def get_papers():

    paper = 0 
    for url in links:
        paper_id = "paper_" + str(paper+1) + ".pdf"
        paper = paper + 1
        output_file = os.path.join("papers", paper_id)
        arxiv_id = url[-10:]
        url_to_get_titles = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(url_to_get_titles)

        if response.status_code != 200:
            return None
        root = ET.fromstring(response.text)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find("atom:entry", ns)
        title = entry.find("atom:title", ns)
        title =  title.text.strip() if title is not None else None
        print("Downloading paper with title :", title)
        metadata = {
            "title" : title
        }
        if os.path.exists(output_file):
            print(f"Skipping (already exists): {output_file}")
            continue
        try:
            print(f"Downloading: {url}")
            download_pdf(url, output_file, metadata)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Failed: {url} -> {e}")

