"""LoukaBuilds SEO Audit — Flask front-end."""
from __future__ import annotations

import logging
import os
import re
from datetime import datetime

from flask import Flask, Response, render_template, request

from audit import run_audit, scrape_site
from pdf_gen import render_pdf


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("seo-audit")


def _valid_url(candidate: str) -> bool:
    if not candidate or len(candidate) > 2048:
        return False
    # Tolerate missing scheme — we'll prepend https:// in scrape_site
    candidate = candidate.strip()
    if "://" not in candidate:
        candidate = "https://" + candidate
    return bool(re.match(r"^https?://[^\s/$.?#][^\s]*$", candidate, re.IGNORECASE))


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/audit")
def audit_html():
    url = (request.form.get("url") or "").strip()
    if not _valid_url(url):
        return render_template("index.html",
                               error="That doesn't look like a valid URL. Try again with something like https://your-business.com"), 400

    log.info("auditing url: %s", url)
    try:
        scrape = scrape_site(url)
        if scrape.error:
            log.warning("scrape error for %s: %s", url, scrape.error)
        audit = run_audit(scrape)
    except Exception as e:
        log.exception("audit pipeline failed for %s", url)
        return render_template("index.html",
                               error=f"Audit failed: {type(e).__name__}: {e}"), 500

    return render_template(
        "result.html",
        audit=audit,
        scrape_error=scrape.error,
        submitted_url=url,
        generated_at=datetime.now().strftime("%B %d, %Y"),
    )


@app.post("/audit.pdf")
def audit_pdf():
    url = (request.form.get("url") or "").strip()
    if not _valid_url(url):
        return Response("Invalid URL.", status=400)

    audit_json = request.form.get("audit_json")
    if audit_json:
        import json
        try:
            audit = json.loads(audit_json)
        except (json.JSONDecodeError, TypeError):
            audit = None
    else:
        audit = None

    if audit is None:
        log.info("auditing (pdf) url: %s", url)
        try:
            scrape = scrape_site(url)
            audit = run_audit(scrape)
        except Exception:
            log.exception("audit (pdf) pipeline failed for %s", url)
            return Response("Audit failed.", status=500)

    pdf_bytes = render_pdf(audit)
    slug = re.sub(r"[^a-z0-9]+", "-", (audit.get("business_name_guess") or "site").lower()).strip("-")
    filename = f"seo-audit-{slug or 'site'}.pdf"
    return Response(
        pdf_bytes, mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/healthz")
def healthz():
    return {"ok": True, "has_api_key": bool(os.environ.get("ANTHROPIC_API_KEY"))}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
