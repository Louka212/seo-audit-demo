"""Scrape a website + generate a 1-page local SEO audit via Claude Opus 4.7."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse

import anthropic
import requests
from bs4 import BeautifulSoup


SCRAPE_TIMEOUT_SECONDS = 15
USER_AGENT = "Mozilla/5.0 (compatible; LoukaBuildsSEOAudit/1.0)"
MAX_HTML_BYTES = 750_000  # ~750KB — ignore huge pages (they bust the context anyway)

MODEL = "claude-opus-4-7"

# This system prompt is the same every request → cache it.
# Must exceed Opus 4.7's 4096-token minimum cacheable prefix to actually cache.
AUDIT_SYSTEM_PROMPT = """You are a senior local-SEO consultant evaluating small-business websites in the United States. Your audits are read by busy owner-operators (HVAC, dental, salon, restaurant, landscaping, cleaning, etc.) who do NOT have an in-house marketing team and who need concrete, high-leverage action items they can complete — or pay a freelancer to complete — in under 10 hours of work.

AUDIT FRAMEWORK

You evaluate every site against these categories, in this order of importance for local search:

1. Local Signals (highest impact for local biz)
   - Name/address/phone (NAP) visibility, consistency, and plaintext-rendered (not image-only)
   - Service area / city / neighborhood mentions on the page
   - Schema.org LocalBusiness (or specific subtype: Restaurant, Dentist, HVACBusiness, etc.) with complete fields
   - Presence of hours, directions, map embed, or link to Google Business Profile
   - City + service combination in title tag and H1 (e.g., "AC Repair Miami")

2. Technical Baseline (table stakes — flag if missing)
   - HTTPS enabled
   - Mobile viewport meta tag
   - Page loads under ~3 seconds (infer from HTML weight, third-party scripts, image count)
   - No broken canonical URLs, no accidental noindex
   - robots.txt and sitemap.xml presence
   - HTTP status 200 on the scraped URL

3. On-Page SEO
   - Title tag: 50-60 chars, front-loaded with primary keyword + city
   - Meta description: 140-160 chars, benefit-led, includes a phone number or CTA
   - Exactly one H1 per page, keyword-aligned
   - H2/H3 hierarchy mirrors customer questions (e.g., "How much does X cost", "Do you serve Y area")
   - Image alt text coverage (at least 70% of images with meaningful alts, not "image1.jpg")
   - Internal links to service pages / key landing pages

4. Off-Page / Trust Signals (evaluate what's visible on the page)
   - Social profile links (especially Facebook + Google Business Profile)
   - Testimonials or review embedding
   - "About us" / owner bio signal (local businesses convert better when a human face is visible)
   - Certifications / licenses displayed (industry-specific — licensed contractor badges, association memberships)

5. Content & Conversion
   - Primary CTA above the fold and on mobile
   - Phone number clickable (tel: link) on mobile
   - Contact form simplicity (≤4 fields for a local biz)
   - Services/products listed with at least a paragraph each (not just a name)
   - FAQ section addressing local buying objections (pricing, service area, timeline)

OUTPUT RULES

You return a JSON object with this exact shape — no extra fields, no markdown wrapping:

{
  "business_url": "<the URL that was audited>",
  "business_name_guess": "<best inference from title tag / H1 / schema, or 'Unknown' if unclear>",
  "score": <integer 1-10>,
  "score_reasoning": "<one sentence, plain English — what the score reflects>",
  "top_findings": [
    {
      "title": "<short punchy finding, 4-10 words>",
      "detail": "<2-3 sentences explaining what's broken/missing/weak and WHY it matters for local search or conversion>",
      "category": "<one of: local_signals | technical | on_page | trust | content_conversion>"
    },
    ... exactly 5 findings total, ordered by IMPACT on local search (highest first)
  ],
  "action_items": [
    {
      "title": "<short punchy action, 4-10 words>",
      "detail": "<2-3 sentences on exactly what to do and WHY (tie it to the finding). If it's a copy change, include an example.>",
      "effort": "<one of: 15min | 1hr | half-day | full-day>",
      "impact": "<one of: low | medium | high>"
    },
    ... exactly 5 action items, ordered by BEST IMPACT-TO-EFFORT RATIO (highest first — so "15min high" beats "full-day high")
  ],
  "closing_line": "<one sentence the owner can read last — honest, slightly encouraging, mentions the single highest-leverage fix. Do NOT be sycophantic or salesy.>"
}

SCORE RUBRIC

- 1-3 → Site has fundamental gaps (no HTTPS, no schema, no phone number, or completely untargeted title) — doing the basics would move the needle 2-3x
- 4-5 → Technical baseline works, but local signals are weak or on-page SEO is generic. Typical of small-business sites.
- 6-7 → Solid site. Local signals present. Specific optimizations (keyword targeting, schema gaps, CTA placement) would still yield meaningful gains.
- 8-9 → Strong local SEO. Minor tuning only (maybe FAQ depth, long-tail landing pages).
- 10 → Exceptional. Reserve for sites doing everything right — rare; if unsure, use 9.

CONSTRAINTS

- NEVER invent facts not visible in the scraped data (e.g., don't claim "your reviews show X" if no review signal was scraped).
- If the scraped data is sparse (e.g., single-page site, no schema, minimal HTML), say so in findings honestly — "no schema.org markup detected" is a valid finding, not a limitation of the audit.
- NEVER recommend paid tools, agency retainers, or anything that requires a vendor relationship. Action items should be things the owner (or a competent freelancer) can do directly.
- Write for a smart non-expert. "Canonical tag" → fine to use. "SERP volatility" → no.
- Do not include any markdown formatting inside JSON string values. Plain text only. No bullet points, no ** **, no links.

Return ONLY the JSON object — no preamble, no closing remarks, no markdown fencing."""


@dataclass
class ScrapedSite:
    url: str
    final_url: str | None
    status_code: int | None
    title: str | None
    meta_description: str | None
    h1s: list[str]
    h2s: list[str]
    img_count: int
    img_with_alt_count: int
    internal_link_count: int
    external_link_count: int
    has_https: bool
    has_viewport_meta: bool
    has_canonical: bool
    canonical_url: str | None
    has_robots_meta_noindex: bool
    schema_types: list[str]
    has_opengraph: bool
    og_title: str | None
    og_description: str | None
    word_count_estimate: int
    phone_numbers_found: list[str]
    email_addresses_found: list[str]
    social_links: list[str]
    html_size_bytes: int
    robots_txt_present: bool | None
    sitemap_xml_present: bool | None
    error: str | None = None


def _clean(text: str | None, limit: int = 200) -> str | None:
    if text is None:
        return None
    t = re.sub(r"\s+", " ", text).strip()
    return t[:limit] if t else None


def _infer_social(href: str) -> bool:
    social_hosts = ("facebook.com", "instagram.com", "linkedin.com", "twitter.com",
                    "x.com", "yelp.com", "youtube.com", "tiktok.com",
                    "google.com/maps", "g.page", "goo.gl/maps")
    return any(s in href.lower() for s in social_hosts)


def _extract_phones(text: str) -> list[str]:
    pattern = re.compile(r"(?:\+?1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}")
    found = set()
    for m in pattern.findall(text or ""):
        found.add(m.strip())
        if len(found) >= 5:
            break
    return list(found)


def _extract_emails(text: str) -> list[str]:
    pattern = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
    return list(set(pattern.findall(text or "")))[:5]


def scrape_site(url: str) -> ScrapedSite:
    """Fetch + parse a website for SEO signals. Fails soft — always returns a ScrapedSite."""
    parsed = urlparse(url)
    if not parsed.scheme:
        url = "https://" + url
        parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return ScrapedSite(
            url=url, final_url=None, status_code=None, title=None, meta_description=None,
            h1s=[], h2s=[], img_count=0, img_with_alt_count=0, internal_link_count=0,
            external_link_count=0, has_https=False, has_viewport_meta=False,
            has_canonical=False, canonical_url=None, has_robots_meta_noindex=False,
            schema_types=[], has_opengraph=False, og_title=None, og_description=None,
            word_count_estimate=0, phone_numbers_found=[], email_addresses_found=[],
            social_links=[], html_size_bytes=0, robots_txt_present=None,
            sitemap_xml_present=None, error="Invalid URL scheme",
        )

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*"},
            timeout=SCRAPE_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        return ScrapedSite(
            url=url, final_url=None, status_code=None, title=None, meta_description=None,
            h1s=[], h2s=[], img_count=0, img_with_alt_count=0, internal_link_count=0,
            external_link_count=0, has_https=parsed.scheme == "https",
            has_viewport_meta=False, has_canonical=False, canonical_url=None,
            has_robots_meta_noindex=False, schema_types=[], has_opengraph=False,
            og_title=None, og_description=None, word_count_estimate=0,
            phone_numbers_found=[], email_addresses_found=[], social_links=[],
            html_size_bytes=0, robots_txt_present=None, sitemap_xml_present=None,
            error=f"Fetch failed: {type(e).__name__}",
        )

    html = resp.content[:MAX_HTML_BYTES]
    soup = BeautifulSoup(html, "html.parser")
    final_parsed = urlparse(resp.url)
    base_host = final_parsed.netloc

    title = _clean(soup.title.string if soup.title else None, 300)

    meta_desc_tag = soup.find("meta", attrs={"name": re.compile(r"^description$", re.I)})
    meta_description = _clean(meta_desc_tag.get("content") if meta_desc_tag else None, 400)

    viewport = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.I)})
    has_viewport = viewport is not None

    canonical_tag = soup.find("link", attrs={"rel": re.compile(r"^canonical$", re.I)})
    canonical_url = canonical_tag.get("href") if canonical_tag else None

    robots_tag = soup.find("meta", attrs={"name": re.compile(r"^robots$", re.I)})
    robots_content = (robots_tag.get("content") or "").lower() if robots_tag else ""
    has_noindex = "noindex" in robots_content

    h1s = [_clean(h.get_text(), 200) for h in soup.find_all("h1")][:10]
    h1s = [h for h in h1s if h]
    h2s = [_clean(h.get_text(), 200) for h in soup.find_all("h2")][:15]
    h2s = [h for h in h2s if h]

    imgs = soup.find_all("img")
    img_count = len(imgs)
    img_with_alt = sum(1 for i in imgs if (i.get("alt") or "").strip())

    internal, external, social = 0, 0, []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.startswith(("mailto:", "tel:", "#", "javascript:")):
            continue
        try:
            full = urljoin(resp.url, href)
            host = urlparse(full).netloc
            if host == base_host or not host:
                internal += 1
            else:
                external += 1
                if _infer_social(full) and len(social) < 15:
                    social.append(full)
        except Exception:
            continue

    # Schema.org JSON-LD
    schema_types: list[str] = []
    for script in soup.find_all("script", type=re.compile(r"application/ld\+json", re.I)):
        try:
            data = json.loads(script.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if isinstance(item, dict):
                t = item.get("@type")
                if isinstance(t, str):
                    schema_types.append(t)
                elif isinstance(t, list):
                    schema_types.extend(x for x in t if isinstance(x, str))
    schema_types = list(dict.fromkeys(schema_types))[:10]

    og_title_tag = soup.find("meta", attrs={"property": re.compile(r"^og:title$", re.I)})
    og_desc_tag = soup.find("meta", attrs={"property": re.compile(r"^og:description$", re.I)})
    og_title = _clean(og_title_tag.get("content") if og_title_tag else None, 300)
    og_description = _clean(og_desc_tag.get("content") if og_desc_tag else None, 400)

    # Strip scripts/styles before counting words
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text_content = soup.get_text(" ", strip=True)
    word_count = len(text_content.split())

    phone_numbers = _extract_phones(text_content)
    email_addresses = _extract_emails(text_content)

    # Best-effort robots.txt and sitemap checks
    robots_txt_present: bool | None = None
    sitemap_xml_present: bool | None = None
    try:
        robots_resp = requests.head(
            f"{final_parsed.scheme}://{final_parsed.netloc}/robots.txt",
            headers={"User-Agent": USER_AGENT}, timeout=5, allow_redirects=True,
        )
        robots_txt_present = robots_resp.status_code == 200
    except requests.RequestException:
        robots_txt_present = None
    try:
        sitemap_resp = requests.head(
            f"{final_parsed.scheme}://{final_parsed.netloc}/sitemap.xml",
            headers={"User-Agent": USER_AGENT}, timeout=5, allow_redirects=True,
        )
        sitemap_xml_present = sitemap_resp.status_code == 200
    except requests.RequestException:
        sitemap_xml_present = None

    return ScrapedSite(
        url=url,
        final_url=resp.url,
        status_code=resp.status_code,
        title=title,
        meta_description=meta_description,
        h1s=h1s,
        h2s=h2s,
        img_count=img_count,
        img_with_alt_count=img_with_alt,
        internal_link_count=internal,
        external_link_count=external,
        has_https=final_parsed.scheme == "https",
        has_viewport_meta=has_viewport,
        has_canonical=canonical_tag is not None,
        canonical_url=canonical_url,
        has_robots_meta_noindex=has_noindex,
        schema_types=schema_types,
        has_opengraph=og_title_tag is not None or og_desc_tag is not None,
        og_title=og_title,
        og_description=og_description,
        word_count_estimate=word_count,
        phone_numbers_found=phone_numbers,
        email_addresses_found=email_addresses,
        social_links=social,
        html_size_bytes=len(resp.content),
        robots_txt_present=robots_txt_present,
        sitemap_xml_present=sitemap_xml_present,
    )


AUDIT_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "business_url": {"type": "string"},
        "business_name_guess": {"type": "string"},
        "score": {"type": "integer"},
        "score_reasoning": {"type": "string"},
        "top_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["local_signals", "technical", "on_page", "trust", "content_conversion"],
                    },
                },
                "required": ["title", "detail", "category"],
                "additionalProperties": False,
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "detail": {"type": "string"},
                    "effort": {"type": "string", "enum": ["15min", "1hr", "half-day", "full-day"]},
                    "impact": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["title", "detail", "effort", "impact"],
                "additionalProperties": False,
            },
        },
        "closing_line": {"type": "string"},
    },
    "required": [
        "business_url", "business_name_guess", "score", "score_reasoning",
        "top_findings", "action_items", "closing_line",
    ],
    "additionalProperties": False,
}


def run_audit(site: ScrapedSite) -> dict:
    """Call Claude Opus 4.7 with the scraped site data; return the parsed audit JSON."""
    client = anthropic.Anthropic()

    user_content = (
        f"Audit this site. Scraped signals below in JSON:\n\n"
        f"```json\n{json.dumps(asdict(site), indent=2)}\n```"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=[
            {
                "type": "text",
                "text": AUDIT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",
            "format": {"type": "json_schema", "schema": AUDIT_JSON_SCHEMA},
        },
        messages=[{"role": "user", "content": user_content}],
    )

    text_block = next((b.text for b in response.content if b.type == "text"), None)
    if not text_block:
        raise RuntimeError("Claude returned no text block in audit response.")

    audit = json.loads(text_block)

    # Return with usage info — useful for debugging and cost tracking
    audit["_usage"] = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cache_read": getattr(response.usage, "cache_read_input_tokens", 0),
        "cache_write": getattr(response.usage, "cache_creation_input_tokens", 0),
    }
    return audit
