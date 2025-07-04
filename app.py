import requests
from bs4 import BeautifulSoup, NavigableString, Tag
import streamlit as st
import json
import openai
import re
from urllib.parse import urlparse, urljoin
import urllib.robotparser
import datetime
import textwrap
from collections import deque

# --- API Key Checks ---
if ("OPENAI_API_KEY" not in st.secrets and 
    "TOGETHER_API_KEY" not in st.secrets and
    "GEMINI_API_KEY" not in st.secrets):
    st.error(
        "No API key set. Configure at least one of: OPENAI_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY in Streamlit secrets."
    )
    st.stop()

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Constants ---
VALID_SCHEMES = ["http", "https"]
EXCLUSION_PATTERNS = [
    r"/training",
    r"/walkthrough",
    r"/downloads",
    r"/faq",
    r"reddit\\.com",
    r"/videos",
    r"/watch",
    r"/lifecycle/faq",
    r"download[s]?/",
    r"/doc[s]?","
    r"/learning",
    r"/tutorial",
    r"copilot.*create"
]

# --- Helper Functions ---

def is_valid_url(url):
    try:
        parsed = urlparse(url)
        return parsed.scheme in VALID_SCHEMES and bool(parsed.netloc)
    except:
        return False


def is_excluded_url(url):
    return any(re.search(pat, url, re.IGNORECASE) for pat in EXCLUSION_PATTERNS)


def clean_text_with_sup_sub(tag):
    for sup in tag.find_all(['sup','sub']):
        sup.unwrap()
    return tag.get_text(strip=True, separator=' ')


def fetch_and_summarize(url, link_filter_prompt=None, model_choice="gpt-3.5-turbo", show_link_scores=False):
    headers = {"User-Agent":"Mozilla/5.0 (compatible; WebScraperBot/1.0)"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title = soup.title.string.strip() if soup.title else "No title"
    headings = [h.get_text(strip=True) for h in soup.find_all(['h1','h2'])]
    paragraphs = [clean_text_with_sup_sub(p) for p in soup.find_all('p')][:10]

    raw_links = [(a.get_text(strip=True), urljoin(url, a['href']))
                 for a in soup.find_all('a', href=True)]
    domain = urlparse(url).netloc
    internal_links = [(t,l) for t,l in raw_links if urlparse(l).netloc==domain or l.startswith('/')]

    ranked = []
    for text, link in internal_links:
        if is_excluded_url(link):
            st.info(f"🚫 Excluded: {link}")
            continue
        score = len(text)*0.5 - urlparse(link).path.count('/')*2
        ranked.append((text, link, score))
    ranked = sorted(ranked, key=lambda x: x[2], reverse=True)[:10]
    links = [(t, l) for t,l,_ in ranked]

    return {"url":url, "title":title, "headings":headings,
            "paragraphs":paragraphs, "links":links,
            "html":soup.prettify()}


def crawl_internal_links(start_url, max_pages, user_agent, link_filter_prompt, model_choice, show_link_scores, show_blocked_links):
    seen, results, queue = set(), [], deque([start_url])
    while queue and len(seen)<max_pages:
        u = queue.popleft()
        if u in seen: continue
        seen.add(u)
        data = fetch_and_summarize(u, link_filter_prompt, model_choice, show_link_scores)
        results.append(data)
        for _,link in data.get('links',[]):
            if is_valid_url(link) and link not in seen:
                # robots.txt check
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url(f"{urlparse(link).scheme}://{urlparse(link).netloc}/robots.txt")
                rp.read()
                if rp.can_fetch(user_agent, link):
                    queue.append(link)
                elif show_blocked_links:
                    st.warning(f"Blocked by robots.txt: {link}")
    return results


def summarize_with_gpt(data, model, depth, tone):
    text = "\n\n".join(data.get('paragraphs',[]))
    prompt = f"""
You are an AI assistant. Summarize into:
- Executive Overview
- Key Ideas
- Insights
- Recommendations
Tone: {tone}, Depth: {depth}

Content:
{text}
"""
    resp = openai.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You structure summaries."},
            {"role":"user","content":prompt}
        ]
    )
    return resp.choices[0].message.content.strip(), model

# --- Streamlit UI ---
st.set_page_config(page_title="Web Scraper Summary Tool", page_icon="🕷️")
st.title("🕷️ Web Scraper Summary Tool")

# Sidebar: API Quota
st.sidebar.subheader("📊 API Quota Remaining")
api_limits = {"OPENAI_API_KEY":90000,"TOGETHER_API_KEY":2500,"GEMINI_API_KEY":10000}
api_used = {k:0 for k in api_limits}
for k,q in api_limits.items():
    if k in st.secrets:
        st.sidebar.text(f"{k.replace('_API_KEY','')}: {q - api_used[k]} remaining")

# Sidebar: Crawler Settings
st.sidebar.subheader("🛡️ Crawler Settings")
use_keyword = st.sidebar.checkbox("Keyword-based crawl", value=False)
keyword_q = st.sidebar.text_input("Search query", "AI strategy site:mckinsey.com")
user_agent = st.sidebar.text_input("User-Agent", "WebScraperBot")
show_blocked = st.sidebar.checkbox("Show blocked by robots.txt", False)
show_scores = st.sidebar.checkbox("Show link scores", False)
link_prompt = st.sidebar.text_area("Link filter prompt", "Focus on AI and strategy")
max_pages = st.sidebar.slider("Max internal pages",1,10,3)

# Main Input
urls_text = st.text_area("Enter URLs (one per line):",
                          "https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage")
model_choice = st.selectbox("Choose Model", ["gpt-3.5-turbo","gpt-4","mixtral-8x7b","gemini-pro"])
depth = st.selectbox("Summary Depth", ["short","medium","long"], index=1)
tone = st.selectbox("Summary Tone", ["neutral","professional","friendly"], index=0)

if st.button("Summarize"):
    if use_keyword:
        from time import sleep
        urls = []
        # call search_web_for_keyword() if implemented
        st.error("Keyword-based crawl not implemented yet.")
        st.stop()
    else:
        urls = [u.strip() for u in urls_text.splitlines() if is_valid_url(u.strip())]
        if not urls:
            st.error("Enter at least one valid URL.")
            st.stop()

    for i,u in enumerate(urls):
        st.subheader(f"🔎 Crawling {u} (up to {max_pages} pages)")
        results = crawl_internal_links(u, max_pages, user_agent, link_prompt,
                                       model_choice, show_scores, show_blocked)
        st.markdown(f"✅ Crawled {len(results)} page(s), filter: `{model_choice}`")

        for j,res in enumerate(results):
            summary, used = summarize_with_gpt(res, model_choice, depth, tone)
            res["summary"] = summary
            res["model_used"] = used
            res["timestamp"] = datetime.datetime.now().isoformat()

            # Render UI
            st.markdown(f"### {res['title']} [🔗]({res['url']})")
            st.markdown(f"**Model used:** {used}")
            st.text_area("🧠 Summary", summary, height=300, key=f"sum_{i}_{j}")
            with st.expander("🕷️ Crawling Info"):
                st.write(f"**URL:** {res['url']}")
                st.write("**Headings:**", res.get('headings',[]))
                st.write("**Links:**", [l for _,l in res.get('links',[])])
            st.download_button(
                "📥 Download JSON",
                data=json.dumps(res, indent=2),
                file_name=f"summary_{i}_{j}.json"
            )
