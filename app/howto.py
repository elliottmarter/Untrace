"""Ask an AI how to delete each "To delete" account, as a one-off background job.

The API key is typed into the Import page and lives only in this job's memory: it is never
written to the database, the logs or the environment, and it is dropped as soon as the job ends.
The only thing sent to the provider is each account's site and name, never usernames or notes.

Answers are stored per site (domain) in the `howto` table, so accounts on the same site share one
lookup, and re-importing doesn't lose them. Talks to each provider's HTTP API with the standard
library, so no extra packages are needed.
"""
import json
import re
import threading
import time
import urllib.error
import urllib.request

# Defaults are each provider's cheapest model that can search the web: a lookup is a short,
# simple question, so a bigger model mostly costs more (with Claude, Opus is ~40x Haiku).
PROVIDERS = {
    "anthropic": {"label": "Claude (Anthropic)", "model": "claude-haiku-5-5",
                  "key_hint": "sk-ant-…", "keys_url": "https://console.anthropic.com/settings/keys"},
    "openai": {"label": "ChatGPT (OpenAI)", "model": "gpt-5-mini",
               "key_hint": "sk-…", "keys_url": "https://platform.openai.com/api-keys"},
    "gemini": {"label": "Gemini (Google)", "model": "gemini-2.5-flash-lite",
               "key_hint": "AIza…", "keys_url": "https://aistudio.google.com/app/apikey"},
}

SYSTEM_PROMPT = """You help someone close online accounts they no longer use.
Search the web for the site's current, official way to delete an account and erase personal data:
a self-service deletion page, a privacy / data-protection email address, or both. Prefer the
site's own help, privacy policy or support pages over third-party articles.

Reply with ONLY a JSON object, no other text, in this shape:
{
  "summary": "One or two plain sentences on how to delete the account.",
  "method": "web" | "email" | "both" | "contact" | "unknown",
  "deletion_url": "https://… the page where the account can be deleted or a deletion request made, or null",
  "email": "the address to email a deletion / erasure request to, or null",
  "steps": ["Short step", "Short step"],
  "notes": "Anything to sort out first (open orders, subscriptions, balances), or null",
  "sources": ["https://… pages you used"]
}
Use "contact" when the only route is phoning or a general contact form. Only give a URL or email
address you actually found; never guess one. If you can't find how, use "unknown" and say so."""


def user_prompt(site, name):
    extra = f' (it is saved as "{name}" in my password manager)' if name and name.lower() != site else ""
    return f"How do I delete my {site} account{extra}?"


class HowtoError(RuntimeError):
    """A lookup failed. fatal=True stops the whole job (e.g. a bad API key)."""

    def __init__(self, message, fatal=False):
        super().__init__(message)
        self.fatal = fatal


def _post(url, body, headers, timeout=180):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", **headers},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8", "replace"))
            detail = (detail.get("error") or {}).get("message") or str(detail)[:200]
        except (ValueError, AttributeError):
            detail = exc.reason
        lowered = str(detail).lower()
        # A bad key or model name will fail every lookup, so stop the job rather than repeat it.
        fatal = exc.code in (401, 403, 404) or (exc.code == 400 and ("model" in lowered or "api key" in lowered))
        if exc.code == 429:
            raise HowtoError(f"Rate limited by the provider: {detail}") from None
        raise HowtoError(f"HTTP {exc.code}: {detail}", fatal=fatal) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HowtoError(f"Couldn't reach the provider: {getattr(exc, 'reason', exc)}") from None
    except ValueError:
        raise HowtoError("The provider sent back something that wasn't JSON.") from None


# Earlier defaults: a saved choice equal to one of these is treated as "use the default".
OLD_DEFAULTS = {"claude-opus-5-5", "gpt-5", "gemini-2.5-flash"}

# Web searches per site. Each search costs a fee and adds its results to the input tokens.
MAX_SEARCHES = 2


def ask_anthropic(key, model, prompt):
    # The newer search tool (it filters results before the model reads them) isn't on Haiku.
    tool = "web_search_20250305" if "haiku" in model else "web_search_20260209"
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {
            "model": model,
            "max_tokens": 4000,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": "low"},
            "tools": [{"type": tool, "name": "web_search", "max_uses": MAX_SEARCHES}],
            "messages": [{"role": "user", "content": prompt}],
        },
        {"x-api-key": key, "anthropic-version": "2023-06-01"},
    )
    if data.get("stop_reason") == "refusal":
        raise HowtoError("The model declined to answer.")
    blocks = data.get("content") or []
    found = [r.get("url") for b in blocks if b.get("type") == "web_search_tool_result"
             and isinstance(b.get("content"), list) for r in b["content"] if r.get("url")]
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text"), found


def ask_openai(key, model, prompt):
    data = _post(
        "https://api.openai.com/v1/responses",
        {"model": model, "instructions": SYSTEM_PROMPT, "input": prompt, "tools": [{"type": "web_search", "search_context_size": "low"}],
         "max_tool_calls": MAX_SEARCHES, "reasoning": {"effort": "low"}},
        {"Authorization": f"Bearer {key}"},
    )
    text, found = [], []
    for item in data.get("output") or []:
        for part in item.get("content") or []:
            if part.get("type") == "output_text":
                text.append(part.get("text", ""))
                found += [a.get("url") for a in part.get("annotations") or [] if a.get("url")]
    return "".join(text), found


def ask_gemini(key, model, prompt):
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"google_search": {}}],
        },
        {"x-goog-api-key": key},
    )
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [])
    chunks = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
    return text, [(c.get("web") or {}).get("uri") for c in chunks if (c.get("web") or {}).get("uri")]


ASK = {"anthropic": ask_anthropic, "openai": ask_openai, "gemini": ask_gemini}

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
METHODS = ("web", "email", "both", "contact", "unknown")


def _url(value):
    value = (value or "").strip() if isinstance(value, str) else ""
    return value if re.match(r"^https?://[^\s\"'<>]+$", value, re.I) else None


def _text(value, limit):
    return value.strip()[:limit] if isinstance(value, str) and value.strip() else None


def parse_answer(text, found_urls=()):
    """Pull the JSON object out of the reply and keep only well-formed fields.

    The answer is shown on the page and turned into links, so anything that isn't a plain
    http(s) URL or a valid email address is dropped.
    """
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise HowtoError("The answer didn't include the expected JSON.")
    try:
        raw = json.loads(text[start:end + 1])
    except ValueError:
        raise HowtoError("The answer's JSON couldn't be read.") from None
    if not isinstance(raw, dict):
        raise HowtoError("The answer's JSON wasn't an object.")
    email = (raw.get("email") or "").strip() if isinstance(raw.get("email"), str) else ""
    email = email.removeprefix("mailto:")
    sources = [u for u in (_url(s) for s in (raw.get("sources") or []) if isinstance(raw.get("sources"), list)) if u]
    for u in found_urls:  # fall back to the pages the search tool returned
        if len(sources) >= 5:
            break
        if _url(u) and u not in sources:
            sources.append(u)
    steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
    method = raw.get("method") if raw.get("method") in METHODS else "unknown"
    return {
        "summary": _text(raw.get("summary"), 600) or "",
        "method": method,
        "deletion_url": _url(raw.get("deletion_url")),
        "email": email if EMAIL_RE.match(email) else None,
        "steps": [s.strip()[:300] for s in steps if isinstance(s, str) and s.strip()][:8],
        "notes": _text(raw.get("notes"), 400),
        "sources": list(dict.fromkeys(sources))[:5],
    }


def lookup(provider, key, model, site, name):
    text, found = ASK[provider](key, model, user_prompt(site, name))
    return parse_answer(text, found)


# --- the background job -------------------------------------------------------

_lock = threading.Lock()
_job = None  # dict, see status()


def status():
    """A copy of the current (or last) job's progress, without the key."""
    with _lock:
        if not _job:
            return None
        return {k: v for k, v in _job.items() if k not in ("key", "cancel")}


def running():
    with _lock:
        return bool(_job and _job["state"] == "running")


def cancel():
    with _lock:
        if _job and _job["state"] == "running":
            _job["cancel"].set()


def start(provider, key, model, sites, save):
    """Run lookups for [(site, name), ...] in a background thread.

    save(site, answer_dict, provider, model) is called for each answer, in that thread.
    """
    global _job
    with _lock:
        if _job and _job["state"] == "running":
            return False
        _job = {
            "state": "running", "provider": provider, "model": model, "total": len(sites),
            "done": 0, "found": 0, "errors": [], "current": "", "started": time.time(),
            "key": key, "cancel": threading.Event(),
        }
        job = _job
    threading.Thread(target=_run, args=(job, sites, save), daemon=True).start()
    return True


def _run(job, sites, save):
    try:
        for site, name in sites:
            if job["cancel"].is_set():
                job["state"] = "cancelled"
                return
            job["current"] = site
            try:
                answer = None
                for attempt in range(3):
                    try:
                        answer = lookup(job["provider"], job["key"], job["model"], site, name)
                        break
                    except HowtoError as exc:
                        if exc.fatal or "Rate limited" not in str(exc) or attempt == 2:
                            raise
                        time.sleep(20 * (attempt + 1))
                save(site, answer, job["provider"], job["model"])
                job["found"] += 1
            except HowtoError as exc:
                job["errors"].append(f"{site}: {exc}")
                if exc.fatal:
                    job["state"] = "failed"
                    job["error"] = str(exc)
                    return
            except Exception as exc:  # keep going; one odd answer shouldn't stop the run
                job["errors"].append(f"{site}: {exc}")
            job["done"] += 1
        job["state"] = "done"
    finally:
        job["key"] = None  # forget the API key as soon as the job stops
        job["current"] = ""
        job["finished"] = time.time()
