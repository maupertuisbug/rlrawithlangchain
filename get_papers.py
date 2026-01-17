import os
import requests



links = [
    "https://arxiv.org/pdf/2402.15567",
    "https://arxiv.org/pdf/2304.04782",
    "https://arxiv.org/pdf/2110.06169",
]


def download_pdf(url, output_path):
    response = requests.get(url)
    response.raise_for_status() 

    with open(output_path, "wb") as f:
        f.write(response.content)

def get_papers():

    paper = 0 
    for url in links:
        paper_id = "paper_" + str(paper+1) + ".pdf"
        paper = paper + 1
        output_file = os.path.join("papers", paper_id)

        if os.path.exists(output_file):
            print(f"Skipping (already exists): {output_file}")
            continue

        try:
            print(f"Downloading: {url}")
            download_pdf(url, output_file)
            print(f"Saved: {output_file}")
        except Exception as e:
            print(f"Failed: {url} -> {e}")


