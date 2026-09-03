"""Unit tests for the scraper's Cloudflare email-obfuscation handling.

Run from src/:  python -m pytest test_audit.py -q
"""
import requests

import audit
from audit import _decode_cfemail, scrape_site

# Captured 2026-09-03 from https://compoundstudio.io with the scraper's own
# User-Agent. Cloudflare Email Address Obfuscation rewrites a plain
# <a href="mailto:..."> into this markup for non-browser clients; the href
# fragment and the data-cfemail attribute both decode to the same address.
CF_HREF_HEX = "3b565a575a507b5854564b544e555f484f4e5f5254155254"
CF_SPAN_HEX = "89e4e8e5e8e2c9eae6e4f9e6fce7edfafdfcede0e6a7e0e6"
CF_EMAIL = "malak@compoundstudio.io"

CLOUDFLARE_HTML = f"""<!doctype html><html><head><title>Compound Studio</title></head>
<body>
<h1>Compound Studio</h1>
<a class="btn s" href="/cdn-cgi/l/email-protection#{CF_HREF_HEX}"><span class="__cf_email__" data-cfemail="{CF_SPAN_HEX}">[email&#160;protected]</span></a>
<script data-cfasync="false" src="/cdn-cgi/scripts/5c5dd728/cloudflare-static/email-decode.min.js"></script>
</body></html>"""


def test_decode_cfemail_known_samples():
    assert _decode_cfemail(CF_SPAN_HEX) == CF_EMAIL
    assert _decode_cfemail(CF_HREF_HEX) == CF_EMAIL
    # key 0x42 XOR "info@example.com"
    assert _decode_cfemail("422b2c242d02273a232f322e276c212d2f") == "info@example.com"


def test_decode_cfemail_rejects_garbage():
    assert _decode_cfemail("") is None
    assert _decode_cfemail("abc") is None          # odd length -> not valid hex bytes
    assert _decode_cfemail("zz11") is None         # not hex at all
    assert _decode_cfemail("4242424242") is None   # decodes to NUL bytes, not an email


class _FakeResponse:
    def __init__(self, html: str, url: str):
        self.content = html.encode("utf-8")
        self.url = url
        self.status_code = 200


def test_scrape_site_decodes_cloudflare_obfuscated_email(monkeypatch):
    url = "https://compoundstudio.io/"
    monkeypatch.setattr(audit, "_is_safe_host", lambda host: (True, ""))
    monkeypatch.setattr(audit.requests, "get", lambda *a, **k: _FakeResponse(CLOUDFLARE_HTML, url))

    def _no_head(*a, **k):
        raise requests.RequestException("offline in test")

    monkeypatch.setattr(audit.requests, "head", _no_head)

    site = scrape_site(url)

    assert site.error is None
    # href fragment + data-cfemail decode to the same address -> exactly one entry
    assert site.email_addresses_found == [CF_EMAIL]
