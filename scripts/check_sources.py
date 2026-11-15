"""Watch the upstream snapshot this model was trained on and report drift.

The watcher never downloads training data into the repository, never retrains
and never republishes. It compares a recorded fingerprint of each upstream
source with what that source serves today, so that a retrain stays a deliberate
decision.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WATCH = ROOT / "sources" / "watch.json"
USER_AGENT = "aloysius109-source-watch/1.0 (+https://github.com/aloysius109/predictive-analytics)"
TIMEOUT = 120


def read(url):
    """Return (status, body). HTTP error responses are returned, not raised."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def digest(body):
    return hashlib.sha256(body).hexdigest()


def names(text, pattern):
    """Return the matched filenames and, from group 1, their release tags."""
    files, releases = set(), set()
    for match in re.finditer(pattern, text):
        files.add(match.group(0))
        if match.groups():
            releases.add(match.group(1))
    return sorted(files), sorted(releases)


def check_bytes(source, status, body):
    if status != 200:
        return "unreachable", f"HTTP {status} where the recorded file used to be served.", {}
    observed = {"sha256": digest(body), "bytes": len(body)}
    if observed["sha256"] == source.get("sha256"):
        return "unchanged", f"{len(body):,} bytes, sha256 unchanged.", observed
    return "changed", (f"Content changed: now sha256 `{observed['sha256']}` ({len(body):,} bytes), "
                       f"recorded `{source.get('sha256')}` ({source.get('bytes', 0):,} bytes)."), observed


def check_links(source, status, body):
    if status != 200:
        return "unreachable", f"HTTP {status} for the listing page.", {}
    files, releases = names(body.decode("utf-8", "replace"), source["pattern"])
    if not files:
        return "unreachable", "The page no longer lists any file matching the expected pattern; it may have moved.", {}
    observed = {"files": files}
    if releases:
        observed["latest_release"] = releases[-1]
    latest = observed.get("latest_release")
    added = [name for name in files if name not in source.get("files", [])]
    removed = [name for name in source.get("files", []) if name not in files]
    notes = []
    if latest and latest != source.get("latest_release"):
        notes.append(f"**New release `{latest}`**, recorded `{source.get('latest_release')}`.")
    if added:
        notes.append(f"{len(added)} new file(s): " + ", ".join(f"`{name}`" for name in added[:8])
                     + (" …" if len(added) > 8 else "") + ".")
    if removed:
        notes.append(f"{len(removed)} file(s) withdrawn: " + ", ".join(f"`{name}`" for name in removed[:8])
                     + (" …" if len(removed) > 8 else "") + ".")
    if notes:
        return "changed", " ".join(notes), observed
    return "unchanged", f"{len(files)} files listed, latest release `{latest}`.", observed


def check_status(source, status, body):
    expected = source.get("status", 200)
    if status == expected:
        return "unchanged", f"HTTP {status}, as recorded.", {}
    return "changed", f"HTTP {status}, recorded HTTP {expected}. {source.get('on_change', '')}".strip(), {}


CHECKS = {"bytes": check_bytes, "links": check_links, "status": check_status}


def inspect(source):
    result = {"id": source["id"], "label": source["label"], "url": source["url"]}
    try:
        status, body = read(source["url"])
    except Exception as error:  # DNS, TLS, timeout, connection reset
        return {**result, "state": "unreachable", "detail": f"Could not be fetched: {error}", "observed": {}}
    state, detail, observed = CHECKS[source["check"]](source, status, body)
    return {**result, "state": state, "detail": detail, "observed": observed}


def fingerprint(drift):
    """A stable identifier for which sources are in which state, ignoring sizes and dates."""
    state = ";".join(sorted(f"{result['id']}:{result['state']}" for result in drift))
    return hashlib.sha256(state.encode("utf-8")).hexdigest()[:12]


def report_markdown(results, title):
    stamp = datetime.now(timezone.utc).strftime("%d %B %Y")
    changed = [result for result in results if result["state"] == "changed"]
    unreachable = [result for result in results if result["state"] == "unreachable"]
    unchanged = [result for result in results if result["state"] == "unchanged"]
    lines = [f"## {title} — {stamp}", "",
             f"{len(unchanged)} of {len(results)} watched sources unchanged.", ""]
    for heading, group in (("Changed upstream", changed), ("Could not be checked", unreachable)):
        if group:
            lines.append(f"### {heading}")
            for result in group:
                lines.append(f"- **{result['label']}** — {result['detail']}")
                lines.append(f"  <{result['url']}>")
            lines.append("")
    if unchanged:
        lines.append("### Unchanged")
        lines += [f"- {result['label']} — {result['detail']}" for result in unchanged]
        lines.append("")
    if changed or unreachable:
        lines += ["### What this means", "",
                  "The model has not been retrained. A changed snapshot means the committed coefficients were fitted",
                  "on data that has since been revised, and retraining is a deliberate decision: it moves every",
                  "published probability. To adopt it, retrain with `python asylum_model.py train --data <new csv>`,",
                  "update `data/provenance.json`, re-run the tests and the report build, then re-baseline this watcher",
                  "with `python scripts/check_sources.py --update`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true",
                        help="Re-baseline the watch file to what the sources serve now")
    parser.add_argument("--strict", action="store_true", help="Exit 1 when a source has drifted")
    parser.add_argument("--report", type=Path, help="Also write the Markdown report to this file")
    args = parser.parse_args()

    watch = json.loads(WATCH.read_text(encoding="utf-8"))
    results = [inspect(source) for source in watch["sources"]]
    drift = [result for result in results if result["state"] != "unchanged"]
    report = report_markdown(results, watch.get("title", "Source watch"))
    if drift:
        report += f"\n<!-- source-watch:{fingerprint(drift)} -->\n"
    print(report)

    if args.report:
        args.report.write_text(report + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(report + "\n")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"drift={'true' if drift else 'false'}\n")

    if args.update:
        observed = {result["id"]: result for result in results}
        for source in watch["sources"]:
            result = observed[source["id"]]
            if result["state"] == "unreachable":
                print(f"Not re-baselining {source['id']}: it could not be checked.", file=sys.stderr)
                continue
            source.update(result["observed"])
        watch["checked_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        WATCH.write_text(json.dumps(watch, indent=2) + "\n", encoding="utf-8")
        print(f"Re-baselined {WATCH.relative_to(ROOT).as_posix()}.")

    if drift and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
