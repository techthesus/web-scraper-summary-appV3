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

# Check for API keys
if ("OPENAI_API_KEY" not in st.secrets and 
    "TOGETHER_API_KEY" not in st.secrets and
    "GEMINI_API_KEY" not in st.secrets):
    st.error("No API key set. Please configure at least one of: OPENAI_API_KEY, TOGETHER_API_KEY, GEMINI_API_KEY in Streamlit secrets.")
    st.stop()

if "OPENAI_API_KEY" in st.secrets:
    openai.api_key = st.secrets["OPENAI_API_KEY"]

VALID_SCHEMES = ["http", "https"]

EXCLUSION_PATTERNS = [
    r"/training",
    r"/walkthrough",
    r"/create",
    r"/downloads",
    r"/faq",
    r"reddit\\.com"
]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in VALID_SCHEMES, result.netloc])
    except Exception:
        return False

def is_excluded_url(url):
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False

def clean_text_with_sup_sub(soup):
    for tag in soup.find_all(['sup', 'sub']):
        tag.unwrap()
    return soup.get_text(strip=True, separator=' ')

def fetch_and_summarize(url, link_filter_prompt=None, model_choice="gpt-3.5-turbo", show_link_scores=False):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; WebScraperBot/1.0)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        headings = [h.get_text(strip=True) for h in soup.find_all(['h1', 'h2'])]
        paragraphs = [clean_text_with_sup_sub(p) for p in soup.find_all('p')][:10]
        raw_links = [(a.get_text(strip=True), urljoin(url, a.get('href'))) for a in soup.find_all('a', href=True)]
        domain = urlparse(url).netloc
        internal_links = [(text, link) for text, link in raw_links if urlparse(link).netloc == domain or link.startswith('/')]

        ranked_links = []
        fallback_keywords = ['insight','ai','tech','future','digital','strategy']

        for text, link in internal_links:
            if is_excluded_url(link):
                continue

            filtered_by = "not_filtered"
            text_lower = text.lower()
            score = len(text) * 0.5 - urlparse(link).path.strip('/').count('/') * 2
            score += 10 if any(k in text_lower for k in fallback_keywords) else 0

            use_semantic_filtering = link_filter_prompt is not None and link_filter_prompt.strip()
            if use_semantic_filtering:
                try:
                    llm_prompt = f"Filter this link: '{text}' with URL '{link}' based on this intent: '{link_filter_prompt}'. Reply 'yes' if relevant, 'no' if not."
                    if model_choice.startswith("gpt") and "OPENAI_API_KEY" in st.secrets:
                        response_llm = openai.chat.completions.create(
                            model=model_choice,
                            messages=[
                                {"role": "system", "content": "You evaluate relevance of webpage links."},
                                {"role": "user", "content": llm_prompt}
                            ]
                        )
                        decision = response_llm.choices[0].message.content.strip().lower()
                        if "no" in decision:
                            filtered_by = f"{model_choice}_filtered"
                            st.info(f"🚫 Filtered out by {model_choice}: {text}")
                            continue
                        else:
                            filtered_by = f"{model_choice}_allowed"
                            st.info(f"✅ Allowed by {model_choice}: {text}")
                except Exception as e:
                    filtered_by = "fallback"
                    st.warning(f"⚠️ Fallback used for: {text} (Error: {str(e)})")
                    score += 5 if any(k in text_lower for k in fallback_keywords) else 0

            ranked_links.append((text, link, score, filtered_by))

        ranked_links = sorted(ranked_links, key=lambda x: x[2], reverse=True)
        if len(ranked_links) == 0:
            ranked_links = [(text, link, len(text) * 0.5, "basic_scoring") for text, link in internal_links[:10] if not is_excluded_url(link)]

        links = [(f"{text} (Score: {score}, Filtered by: {filtered_by})" if show_link_scores else text, link) 
                for text, link, score, filtered_by in ranked_links[:10]]

        html_content = soup.prettify()

        data = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "html": html_content
        }

        return data
    except Exception as e:
        return {"error": str(e)}

# Additional feature - API usage remaining counter (for demonstration)
def show_api_usage():
    if "SERPER_API_KEY" in st.secrets:
        st.sidebar.info("🔄 API Usage Remaining for Serper: Approx. 2500/month (check your account dashboard for exact)")
    if "GOOGLE_CSE_API_KEY" in st.secrets:
        st.sidebar.info("🔄 Google CSE: Daily limit based on project quota (varies)")
    if "OPENAI_API_KEY" in st.secrets:
        st.sidebar.info("🔄 OpenAI: View usage at https://platform.openai.com/account/usage")

show_api_usage()

# UI Update Snippet for summary presentation (include in your Streamlit display logic)
def render_summary_block(result):
    title = result.get("title", "Untitled")
    url = result.get("url", "#")
    summary = result.get("summary", "")
    model_used = result.get("model_used", "")

    st.markdown(f"### {title} [🔗]({url})")
    st.markdown(f"**Model used:** {model_used}")
    st.text_area("🧠 Summary", summary, height=400)

    with st.expander("📄 Crawling Info"):
        st.write("**Headings:**", result.get('headings', []))
        st.write("**Links:**", [link[0] for link in result.get('links', [])])
