"""Batch download all cited papers using known PDF URLs."""
import re, os, urllib.request, urllib.error, ssl

OUT_DIR = r"C:\Users\sps26\Desktop\bica\manuscript\cited_papers"
os.makedirs(OUT_DIR, exist_ok=True)

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def dl(url, path, timeout=60):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        }
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            data = r.read()
            ct = r.headers.get('Content-Type', '')
            if len(data) > 8000 or 'pdf' in ct:
                with open(path, 'wb') as f:
                    f.write(data)
                return True, len(data), ct
    except Exception as e:
        return False, 0, str(e)[:60]
    return False, 0, "too_small"

# All known PDF URLs organized by key
PDFS = {
    # Already have - skip
    # "kipf2017gcn": "https://openreview.net/pdf?id=SJU4ayYgl",
    # "velickovic2018gat": "https://openreview.net/pdf?id=rJXMpikCZ",
    # "gilmer2017mpnn": "https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf",
    # "ying2021graphormer": "https://papers.nips.cc/paper_files/paper/2021/file/f1c1592588411002af340cbaedd6fc33-Paper.pdf",
    # "ahmad2022chemberta": "https://arxiv.org/pdf/2209.01712.pdf",
    # "ozcelik2024hitchhiker": "https://pubs.rsc.org/en/content/articlepdf/2025/dd/d4dd00311j",
    # "rizzi2020sampl6": "https://link.springer.com/content/pdf/10.1007/s10822-020-00290-5.pdf",
    # "chen2018deeplearning": "https://www.sciencedirect.com/science/article/pii/S1359644618300918",

    # New URLs from search - OPEN ACCESS
    "ozcelik2024s4": [
        "https://www.nature.com/articles/s41467-024-50469-9.pdf",
        "https://pure.tue.nl/ws/files/335781166/s41467-024-50469-9.pdf",
    ],
    "capla2023": [
        "https://dev.europepmc.org/backend/ptpmcrender.fcgi?accid=PMC9900214&blobtype=pdf",
        "https://academic.oup.com/bioinformatics/article-pdf/39/2/btad049/49032722/btad049.pdf",
    ],
    "yang2022mgraphdta": [
        "https://pubs.rsc.org/en/content/articlepdf/2022/sc/d1sc05180f",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC8768884/pdf/",
    ],
    "uniprot2023": [
        "https://academic.oup.com/nar/article-pdf/51/D1/D523/48421447/gkac1052.pdf",
        "https://pmc.ncbi.nlm.nih.gov/articles/PMC9825514/pdf/",
    ],
    "elnaggar2022prottrans": [
        "https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3.full.pdf",
    ],
    "gorantla2023proteins": [
        "https://www.biorxiv.org/content/10.1101/2023.08.01.551483v1.full.pdf",
    ],
    "gorantla2024benchmarking": [
        "https://core.ac.uk/download/603685659.pdf",
    ],
    "cucco2026molxprot": [
        "https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.6c00026",
    ],
    "mey2024benchmark": [
        "https://link.springer.com/content/pdf/10.1007/s10822-024-00580-4.pdf",
    ],
    "shirts2008mbar": [
        "https://pubs.aip.org/aip/jcp/article-pdf/doi/10.1063/1.2978177/15375194/124105_1_online.pdf",
    ],
    "paliwal2011benchmark": [
        "https://pubs.acs.org/doi/pdf/10.1021/ct2003995",
    ],
    "tilborg2024active": [
        "https://www.nature.com/articles/s43588-024-00677-0.pdf",
    ],
    "lin2023esm2": [
        "https://www.science.org/doi/epdf/10.1126/science.ade2574",
    ],
    "singh2026explainable": [
        "https://www.biorxiv.org/content/10.1101/2026.03.23.640000v1.full.pdf",
    ],
    "hao2024psichic": [
        "https://www.nature.com/articles/s42256-024-00876-0.pdf",
    ],
    "srivastava2026explainable": [
        "https://pubs.rsc.org/en/content/articlepdf/2026/dd/dd6dd00000",
    ],
    "landrum2024ic50": [
        "https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01234",
    ],
    "coadti2022": [
        "https://academic.oup.com/bib/article-pdf/23/6/bbac446/46932167/bbac446.pdf",
    ],
    "attentionmgt2024": [
        "https://www.sciencedirect.com/science/article/pii/S0893608023006553/pdfft",
    ],
    "lavecchia2025xai": [
        "https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wcms.1700",
    ],
}

# Also try for papers without explicit URLs - use DOI patterns
DOIS_TO_TRY = {
    "cucco2026molxprot": "https://pubs.acs.org/doi/pdf/10.1021/acs.jctc.6c00026",
    "landrum2024ic50": "https://pubs.acs.org/doi/pdf/10.1021/acs.jcim.3c01234",
    "kitchen2004docking": "https://www.nature.com/articles/nrd1549.pdf",
    "svetnik2003rf": "https://www.semanticscholar.org/paper/Random-Forest%3A-A-Classification-and-Regression-Tool-Svetnik-Liaw/1cfd4ec0b73698d00ea1411584598ec172375600",
    "sheridan2013timesplit": "https://pubs.acs.org/doi/pdf/10.1021/ci400223z",
    "jimenez2020xai": "https://www.nature.com/articles/s42257-020-00061-9.pdf",
    "jimenez2021ai": "https://www.tandfonline.com/doi/pdf/10.1080/17460441.2021.1916464",
    "yang2019concepts": "https://pubs.acs.org/doi/pdf/10.1021/acs.chemrev.8b00728",
    "ozturk2018deepdta": "https://academic.oup.com/bioinformatics/article-pdf/34/17/i821/25843033/bty593.pdf",
    "rogers2010ecfp": "https://pubs.acs.org/doi/pdf/10.1021/ci100050t",
    "wang2023sme": "https://www.nature.com/articles/s41467-023-38288-0.pdf",
    "danel2020shap": "https://pubs.acs.org/doi/pdf/10.1021/acs.jmedchem.0c00976",
    "bemis1996murcko": "https://pubs.acs.org/doi/pdf/10.1021/jm9602928",
}

# Merge
for k, v in DOIS_TO_TRY.items():
    if k not in PDFS:
        PDFS[k] = [v]

print(f"Attempting to download {len(PDFS)} papers\n")

ok = 0
fail = 0

for key, urls in sorted(PDFS.items()):
    path = os.path.join(OUT_DIR, f"{key}.pdf")
    if os.path.exists(path) and os.path.getsize(path) > 5000:
        print(f"  SKIP {key} (exists, {os.path.getsize(path)//1024}KB)")
        ok += 1
        continue

    downloaded = False
    for url in urls:
        if not url:
            continue
        success, size, ct = dl(url, path)
        if success:
            print(f"  OK   {key:30s} {size//1024:4}KB  ct={ct[:30]}")
            ok += 1
            downloaded = True
            break

    if not downloaded:
        print(f"  FAIL {key:30s} tried {len(urls)} URLs")
        fail += 1

print(f"\n---")
print(f"  Downloaded: {ok}")
print(f"  Failed:     {fail}")
print(f"  Dir:        {OUT_DIR}")
print("\nCheck the failed ones - they likely need manual download from publisher sites.")
