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

# Define exclusion patterns for unwanted link types
EXCLUSION_PATTERNS = [
    # Training and learning content
    r'/training/',
    r'/learn/',
    r'/learning/',
    r'/tutorial/',
    r'/tutorials/',
    r'/walkthrough/',
    r'/walkthroughs/',
    r'/course/',
    r'/courses/',
    r'/academy/',
    r'/certification/',
    r'/certifications/',
    r'/education/',
    r'/workshop/',
    r'/workshops/',
    
    # Video and multimedia content
    r'/video/',
    r'/videos/',
    r'/webinar/',
    r'/webinars/',
    r'/demo/',
    r'/demos/',
    r'/watch/',
    r'/media/',
    r'/multimedia/',
    r'\.mp4$',
    r'\.avi$',
    r'\.mov$',
    r'\.wmv$',
    r'youtube\.com',
    r'vimeo\.com',
    
    # Downloads and documents
    r'/download/',
    r'/downloads/',
    r'/document/',
    r'/documents/',
    r'/file/',
    r'/files/',
    r'/assets/',
    r'/resources/',
    r'\.pdf$',
    r'\.doc$',
    r'\.docx$',
    r'\.xls$',
    r'\.xlsx$',
    r'\.ppt$',
    r'\.pptx$',
    r'\.zip$',
    r'\.rar$',
    r'\.tar$',
    r'\.gz$',
    
    # Additional common exclusions
    r'/support/',
    r'/help/',
    r'/faq/',
    r'/contact/',
    r'/about/',
    r'/privacy/',
    r'/terms/',
    r'/legal/',
    r'/careers/',
    r'/jobs/',
    r'/events/',
    r'/event/',
    r'/press/',
    r'/news/',
    r'/blog/',
    r'/forums/',
    r'/forum/',
    r'/community/',
]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme in VALID_SCHEMES, result.netloc])
    except Exception:
        return False

def should_exclude_link(url, text=""):
    """Check if a link should be excluded based on URL patterns and text content"""
    url_lower = url.lower()
    text_lower = text.lower()
    
    # Check URL patterns
    for pattern in EXCLUSION_PATTERNS:
        if re.search(pattern, url_lower, re.IGNORECASE):
            return True, f"URL pattern: {pattern}"
    
    # Check text content for additional exclusion keywords
    exclusion_keywords = [
        'training', 'learn', 'tutorial', 'walkthrough', 'course', 'academy',
        'certification', 'education', 'workshop', 'video', 'webinar', 'demo',
        'download', 'document', 'file', 'guide', 'manual', 'handbook',
        'getting started', 'how to', 'step by step', 'quick start'
    ]
    
    for keyword in exclusion_keywords:
        if keyword in text_lower:
            return True, f"Text keyword: {keyword}"
    
    return False, ""

def clean_text_with_sup_sub(soup):
    for tag in soup.find_all(['sup', 'sub']):
        tag.unwrap()
    return soup.get_text(strip=True, separator=' ')

def fetch_and_summarize(url, link_filter_prompt=None, model_choice="gpt-3.5-turbo", show_link_scores=False, show_excluded_links=False):
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
        excluded_links = []
        fallback_keywords = ['insight','ai','tech','future','digital','strategy']
        
        for text, link in internal_links:
            # Check if link should be excluded
            should_exclude, exclusion_reason = should_exclude_link(link, text)
            if should_exclude:
                excluded_links.append((text, link, exclusion_reason))
                if show_excluded_links:
                    st.warning(f"🚫 Excluded: {text} - {link} (Reason: {exclusion_reason})")
                continue
            
            filtered_by = "not_filtered"
            text_lower = text.lower()
            score = len(text) * 0.5 - urlparse(link).path.strip('/').count('/') * 2
            score += 10 if any(k in text_lower for k in ['insight','ai','tech','future','digital','strategy']) else 0

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
        
        # Show exclusion summary
        if excluded_links:
            st.info(f"📊 Excluded {len(excluded_links)} links (training, downloads, videos, etc.)")
        
        ranked_links = sorted(ranked_links, key=lambda x: x[2], reverse=True)
        if len(ranked_links) == 0:
            # If all links were excluded, show a subset of the best excluded links
            if excluded_links:
                st.warning("⚠️ All internal links were excluded. Consider adjusting exclusion rules.")
                # Take first 5 excluded links as fallback
                ranked_links = [(text, link, len(text) * 0.5, "fallback_excluded") for text, link, _ in excluded_links[:5]]
            else:
                ranked_links = [(text, link, len(text) * 0.5, "basic_scoring") for text, link in internal_links[:10]]
        
        links = [(f"{text} (Score: {score}, Filtered by: {filtered_by})" if show_link_scores else text, link) 
                for text, link, score, filtered_by in ranked_links[:10]]

        html_content = soup.prettify()

        data = {
            "url": url,
            "title": title,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "excluded_links": excluded_links,
            "html": html_content
        }

        return data
    except Exception as e:
        return {"error": str(e)}

def is_allowed_by_robots(url, user_agent):
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True  # Default to allowing if robots.txt check fails

def crawl_internal_links(start_url, max_pages=3, user_agent="WebScraperBot", link_filter_prompt=None, model_choice="gpt-3.5-turbo", show_link_scores=False, show_blocked_links=False, show_excluded_links=False):
    seen = set()
    results = []
    queue = deque([start_url])

    while queue and len(seen) < max_pages:
        current_url = queue.popleft()
        if current_url in seen:
            continue
        seen.add(current_url)
        result = fetch_and_summarize(current_url, link_filter_prompt, model_choice, show_link_scores, show_excluded_links)
        if "error" not in result:
            results.append(result)
            for _, link in result.get("links", []):
                if is_valid_url(link) and link not in seen:
                    # Additional check to ensure we don't crawl excluded link types
                    should_exclude, _ = should_exclude_link(link)
                    if not should_exclude:
                        if is_allowed_by_robots(link, user_agent):
                            queue.append(link)
                        elif show_blocked_links:
                            st.warning(f"Blocked by robots.txt: {link}")
                    elif show_excluded_links:
                        st.info(f"Skipped crawling excluded link: {link}")
    return results

def summarize_with_gpt(data, selected_model, depth, tone):
    combined = "\n\n".join(data.get("paragraphs", []))
    prompt = f"""
You are an AI assistant. Summarize the following content into an executive overview, categorized bullet points, and detailed insights.
Tone: {tone}
Depth: {depth}
Include:
- Executive Summary
- Key Ideas
- Insights
- Recommendations
- Bullet format if appropriate
- Separate sections by headings

Content:
{combined}
"""

    try:
        if selected_model.startswith("gpt") and "OPENAI_API_KEY" in st.secrets:
            response = openai.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": "You summarize and structure webpage content."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content.strip(), "OpenAI: " + selected_model

        elif selected_model.startswith("mixtral") and "TOGETHER_API_KEY" in st.secrets:
            headers = {
                "Authorization": f"Bearer {st.secrets['TOGETHER_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {"role": "system", "content": "You summarize and structure webpage content."},
                    {"role": "user", "content": prompt}
                ]
            }
            res = requests.post("https://api.together.xyz/v1/chat/completions", headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"].strip(), "Together.ai: Mixtral"

        elif selected_model.startswith("gemini") and "GEMINI_API_KEY" in st.secrets:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={st.secrets['GEMINI_API_KEY']}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{"parts": [{"text": prompt}]}]
            }
            res = requests.post(url, headers=headers, json=payload)
            res.raise_for_status()
            return res.json()['candidates'][0]['content']['parts'][0]['text'].strip(), "Google: Gemini"

    except Exception as e:
        return f"Model error: {e}", selected_model

    return "No available summarization model succeeded.", "None"

def search_web_for_keyword(query):
    """Search web using available search APIs"""
    urls = []
    
    SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
    GOOGLE_CSE_API_KEY = st.secrets.get("GOOGLE_CSE_API_KEY", None)
    GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", None)

    # Try Serper API first
    if SERPER_API_KEY:
        try:
            headers = {
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            }
            payload = {
                "q": query,
                "gl": "us",
                "hl": "en",
                "num": 10
            }
            res = requests.post("https://google.serper.dev/search", json=payload, headers=headers, timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                urls = [r["link"] for r in data.get("organic", [])]
                if urls:
                    st.success(f"✅ Serper API found {len(urls)} URLs")
                    return urls
            elif res.status_code == 404:
                st.error("❌ Serper API endpoint not found. Please check your API configuration.")
            elif res.status_code == 401:
                st.error("❌ Invalid Serper API key. Please check your API key in secrets.")
            elif res.status_code == 429:
                st.error("❌ Serper API rate limit exceeded. Please try again later.")
            else:
                st.error(f"❌ Serper API error: {res.status_code} - {res.text}")
                
        except requests.exceptions.Timeout:
            st.warning("⏱️ Serper API request timed out")
        except requests.exceptions.ConnectionError:
            st.warning("🔌 Unable to connect to Serper API")
        except Exception as e:
            st.warning(f"⚠️ Serper API error: {str(e)}")

    # Try Google Custom Search API as fallback
    if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
        try:
            params = {
                "key": GOOGLE_CSE_API_KEY,
                "cx": GOOGLE_CSE_ID,
                "q": query,
                "num": 10
            }
            res = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                urls = [item['link'] for item in data.get('items', [])]
                if urls:
                    st.success(f"✅ Google CSE found {len(urls)} URLs")
                    return urls
            elif res.status_code == 403:
                error_details = res.json().get('error', {})
                error_message = error_details.get('message', 'Unknown error')
                if 'Daily Limit Exceeded' in error_message:
                    st.error("❌ Google CSE daily quota exceeded. Please try again tomorrow.")
                elif 'API key not valid' in error_message:
                    st.error("❌ Invalid Google CSE API key. Please check your API key in secrets.")
                else:
                    st.error(f"❌ Google CSE access denied: {error_message}")
            elif res.status_code == 400:
                st.error("❌ Invalid Google CSE request. Please check your search query and CSE ID.")
            else:
                st.error(f"❌ Google CSE error: {res.status_code} - {res.text}")
                
        except requests.exceptions.Timeout:
            st.warning("⏱️ Google CSE request timed out")
        except requests.exceptions.ConnectionError:
            st.warning("🔌 Unable to connect to Google CSE API")
        except Exception as e:
            st.warning(f"⚠️ Google CSE error: {str(e)}")

    # If no APIs worked, provide helpful guidance
    if not urls:
        st.error("❌ No search results found. Please check:")
        st.write("1. **API Keys**: Ensure your SERPER_API_KEY or GOOGLE_CSE_API_KEY are correctly set in Streamlit secrets")
        st.write("2. **Serper API**: Sign up at https://serper.dev for free API access")
        st.write("3. **Google CSE**: Set up Custom Search Engine at https://programmablesearchengine.google.com/")
        st.write("4. **Search Query**: Try a simpler search query")
        
    return urls

# Streamlit UI
st.set_page_config(page_title="Web Scraper Summary Tool", page_icon="🕷️")
st.title("🕷️ Web Scraper Summary Tool")

# Sidebar configuration
st.sidebar.subheader("🌐 Keyword Web Crawler")
keyword_query = st.sidebar.text_input("Search query (Google-style)", "AI strategy site:mckinsey.com")
use_keyword_crawl = st.sidebar.checkbox("Use keyword-based crawling", value=False)

# API Configuration Help
if use_keyword_crawl:
    st.sidebar.info("📝 **API Setup Required:**")
    SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
    GOOGLE_CSE_API_KEY = st.secrets.get("GOOGLE_CSE_API_KEY", None)
    GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", None)
    
    if not SERPER_API_KEY and not (GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID):
        st.sidebar.error("❌ No search API configured!")
        st.sidebar.write("**Option 1 - Serper API (Recommended):**")
        st.sidebar.write("1. Sign up at https://serper.dev")
        st.sidebar.write("2. Get free API key (2,500 queries/month)")
        st.sidebar.write("3. Add `SERPER_API_KEY` to Streamlit secrets")
        st.sidebar.write("**Option 2 - Google Custom Search:**")
        st.sidebar.write("1. Create CSE at https://programmablesearchengine.google.com/")
        st.sidebar.write("2. Get API key from Google Cloud Console")
        st.sidebar.write("3. Add `GOOGLE_CSE_API_KEY` and `GOOGLE_CSE_ID` to secrets")
    else:
        if SERPER_API_KEY:
            st.sidebar.success("✅ Serper API configured")
        if GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID:
            st.sidebar.success("✅ Google CSE configured")

st.sidebar.subheader("🛡️ Crawler Settings")
user_agent = st.sidebar.text_input("User-Agent string (for robots.txt)", "WebScraperBot")
show_blocked_links = st.sidebar.checkbox("Log links blocked by robots.txt", value=False)
show_excluded_links = st.sidebar.checkbox("Show excluded links (training, downloads, videos)", value=False)
show_link_scores = st.sidebar.checkbox("Show link scores", value=False)

st.sidebar.subheader("🔗 Link Filtering")
link_filter_prompt = st.sidebar.text_area("Link filter prompt (optional)", 
                                        "Focus on AI, strategy, and technology insights")

# Main interface
url = st.text_area("Enter one or more webpage URLs (one per line):", 
                   "https://www.mckinsey.com/capabilities/quantumblack/our-insights/seizing-the-agentic-ai-advantage")

model_choice = st.selectbox("Choose AI Model", [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-1106",
    "gpt-4",
    "mixtral-8x7b",
    "gemini-pro"
])

depth = st.selectbox("Summary Depth", ["short", "medium", "long"], index=1)
tone = st.selectbox("Summary Tone", ["neutral", "professional", "friendly", "assertive"], index=0)
max_pages = st.slider("Max internal pages to crawl (per site):", 1, 10, 3)

# Main execution
if st.button("Summarize"):
    if use_keyword_crawl:
        # Check if search APIs are available
        SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", None)
        GOOGLE_CSE_API_KEY = st.secrets.get("GOOGLE_CSE_API_KEY", None)
        GOOGLE_CSE_ID = st.secrets.get("GOOGLE_CSE_ID", None)
        
        if not SERPER_API_KEY and not (GOOGLE_CSE_API_KEY and GOOGLE_CSE_ID):
            st.error("❌ No search API configured. Please set up either Serper API or Google Custom Search Engine.")
            st.stop()
            
        st.info(f"🔍 Searching for: '{keyword_query}'")
        urls = search_web_for_keyword(keyword_query)
        
        if not urls:
            st.error("❌ No URLs found for the given search query. Please try:")
            st.write("- Using different search terms")
            st.write("- Checking your API key configuration")
            st.write("- Trying manual URL input instead")
            st.stop()
    else:
        urls = [line.strip() for line in url.splitlines() if line.strip() and is_valid_url(line.strip())]
        if not urls:
            st.error("Please enter at least one valid URL starting with http:// or https://")
            st.stop()

    for i, u in enumerate(urls):
        st.subheader(f"🔎 Crawling {u} (up to {max_pages} internal pages)")
        crawled_results = crawl_internal_links(
            u, 
            max_pages=max_pages, 
            user_agent=user_agent,
            link_filter_prompt=link_filter_prompt,
            model_choice=model_choice,
            show_link_scores=show_link_scores,
            show_blocked_links=show_blocked_links,
            show_excluded_links=show_excluded_links
        )
        
        st.markdown(f"✅ Crawled: {len(crawled_results)} page(s)")
        st.markdown(f"📌 Filter model: `{model_choice}`")

        for j, result in enumerate(crawled_results):
            st.markdown(f"### 📄 Summary {j+1} - {result['url']}")
            st.write(f"**Title:** {result['title']}")

            with st.spinner("Generating summary using AI model..."):
                ai_summary, used_model = summarize_with_gpt(result, model_choice, depth, tone)

            st.markdown(f"**Model used:** {used_model}")
            st.text_area("🧠 Summary", ai_summary, height=400, key=f"summary_{i}_{j}")

            result.update({
                "summary": ai_summary,
                "model_used": used_model,
                "timestamp": datetime.datetime.now().isoformat()
            })

            st.download_button(
                label="📥 Download Summary JSON",
                data=json.dumps(result, indent=4, ensure_ascii=False),
                file_name=f"summary_{i+1}_{j+1}.json",
                key=f"download_{i}_{j}"
            )

            with st.expander("🔗 Show Links and Headings"):
                st.write("**Headings:**", result.get('headings', []))
                st.write("**Links:**", [link[0] for link in result.get('links', [])])
                
                # Show excluded links if any
                if result.get('excluded_links'):
                    st.write("**Excluded Links:**")
                    for text, link, reason in result.get('excluded_links', []):
                        st.write(f"- {text} ({reason})")
