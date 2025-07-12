import os
import time
import re
import warnings
import requests
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from statsmodels.tsa.arima.model import ARIMA
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError, ChunkedEncodingError, RequestException
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

CONFERENCES = ['CVPR', 'ICML', 'ICLR', 'KDD']
START_YEAR = 2020
END_YEAR = pd.Timestamp.now().year - 1
DATA_DIR = 'data'
RAW_CSV = os.path.join(DATA_DIR, 'raw_papers.csv')
CLEAN_CSV = os.path.join(DATA_DIR, 'clean_papers.csv')
RESULT_DIR = 'results'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

KEY_MAP = {'neurips': 'nips'}

print("Configuration loaded:")
print(f"Conferences: {CONFERENCES[:5]}")
print(f"Years: {START_YEAR}-{END_YEAR}")

def build_urls(conf, year):
    key = KEY_MAP.get(conf.lower(), conf.lower())
    urls = [
        f'https://dblp.org/db/conf/{key}/{key}{year}.html',
        f'https://dblp.org/db/conf/{key}/{year}.html'
    ]
    print(f"[build_urls] {conf} {year} -> {urls}")
    return urls

def fetch_conference(conf, year):
    print(f"[fetch_conference] start for {conf} {year}")
    if conf == 'ICCV' and year % 2 == 0:
        print(f"[fetch_conference] skipping {conf} {year}")
        return []
    for url in build_urls(conf, year):
        print(f"[fetch_conference] trying URL: {url}")
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            papers = []
            for item in soup.select('ul.publ-list > li.entry'):
                title = item.select_one('span.title')
                if not title:
                    continue
                author_tags = item.select('span[itemprop=author] a')
                if author_tags:
                    authors = [a.get_text(strip=True) for a in author_tags]
                else:
                    authors = [span.get_text(strip=True) for span in item.select('span.author')]
                link = item.select_one('a[href^="https://doi.org"]') or item.select_one('nav.publ > ul > li > a[href]')
                url_link = link['href'] if link else ''
                papers.append({'title': title.get_text(strip=True), 'authors': ', '.join(dict.fromkeys(authors)), 'year': year, 'conf': conf, 'url': url_link})
            print(f"[fetch_conference] fetched {len(papers)} papers from {url}")
            return papers
        except (HTTPError, ChunkedEncodingError, RequestException) as e:
            print(f"[fetch_conference] warning: failed at {url}: {e}")
    print(f"[fetch_conference] all attempts failed for {conf} {year}")
    return []

def scrape_all():
    print("[scrape_all] started")
    all_papers = []
    for conf in CONFERENCES[:5]:
        for year in range(START_YEAR, END_YEAR + 1):
            print(f"[scrape_all] fetching {conf} {year}")
            papers = fetch_conference(conf, year)
            print(f"[scrape_all] got {len(papers)} records for {conf} {year}")
            all_papers.extend(papers)
            time.sleep(1)
    df = pd.DataFrame(all_papers)
    df.to_csv(RAW_CSV, index=False)
    print(f"[scrape_all] raw data saved to {RAW_CSV}, total records: {len(df)}")

def clean_data():
    print("[clean_data] started")
    df = pd.read_csv(RAW_CSV)
    before = len(df)
    df = df.dropna(subset=['title', 'year'])
    df['authors'] = df['authors'].fillna('')
    df['authors'] = df['authors'].apply(lambda x: [a.strip() for a in x.split(',') if a.strip()])
    df = df.drop_duplicates(subset=['title', 'conf', 'year'])
    df.to_csv(CLEAN_CSV, index=False)
    after = len(df)
    print(f"[clean_data] cleaned data saved to {CLEAN_CSV}, removed {before-after} duplicates/nulls")

def plot_trends():
    print("[plot_trends] started")
    df = pd.read_csv(CLEAN_CSV)
    counts = df.groupby(['conf', 'year']).size().unstack(fill_value=0)
    for conf in counts.index:
        x = counts.columns.tolist()
        y = counts.loc[conf].tolist()
        plt.plot(x, y, marker='o', label=conf)
        for xi, yi in zip(x, y):
            plt.text(xi, yi, str(yi), ha='center', va='bottom')
    plt.xlabel('Year')
    plt.ylabel('Paper Count')
    plt.title('Annual Paper Counts')
    plt.legend()
    output = os.path.join(RESULT_DIR, 'paper_trend.png')
    plt.savefig(output)
    plt.close()
    print(f"[plot_trends] plot saved to {output}")

def generate_wordcloud():
    print("[generate_wordcloud] started")
    df = pd.read_csv(CLEAN_CSV)
    text = ' '.join(df['title'].astype(str)).lower()
    words = re.findall(r"\b[a-z]{2,}\b", text)
    stops = {'the','and','for','with','from','using','via','per','to','of','in','on','a','an'}
    words = [w for w in words if w not in stops]
    freq = pd.Series(words).value_counts()
    wc = WordCloud(width=800, height=400).generate_from_frequencies(freq.to_dict())
    plt.imshow(wc)
    plt.axis('off')
    output = os.path.join(RESULT_DIR, 'keyword_wordcloud.png')
    plt.savefig(output)
    plt.close()
    print(f"[generate_wordcloud] wordcloud saved to {output}")

def predict_counts():
    print("[predict_counts] started")
    df = pd.read_csv(CLEAN_CSV)
    counts = df.groupby(['conf', 'year']).size().unstack(fill_value=0)
    results = []
    for conf, series in counts.iterrows():
        print(f"[predict_counts] forecasting for {conf}")
        if series.sum() == 0 or series.count() < 3:
            print(f"[predict_counts] skipping {conf} due to insufficient data")
            continue
        model = ARIMA(series.astype(float), order=(1,1,1)).fit()
        pred = model.forecast(1)
        value = int(pred.iloc[0])
        print(f"[predict_counts] {conf} -> {value}")
        results.append({
            'conf': conf,
            'next_year': END_YEAR + 1,
            'predicted_count': value
        })
    # 将预测结果存入 CSV
    pred_df = pd.DataFrame(results)
    pred_file = os.path.join(DATA_DIR, 'predictions.csv')
    pred_df.to_csv(pred_file, index=False)
    print(f"[predict_counts] predictions saved to {pred_file}")


def main():
    print("[main] process starting")
    scrape_all()
    clean_data()
    plot_trends()
    generate_wordcloud()
    predict_counts()
    print("[main] pipeline complete")

if __name__ == '__main__':
    main()
