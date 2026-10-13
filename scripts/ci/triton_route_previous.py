#!/usr/bin/env python3
"""Fetch the durations of the newest earlier Triton route sweep, for --order-from.

  GH_TOKEN=... python3 scripts/ci/triton_route_previous.py OWNER/REPO OUT.json

Standard library only. Exits non-zero when there is none to fetch; the caller
then sweeps in path order.
"""

import io
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile

ARTIFACT = "triton-route-coverage"
TRIES = 5
API = os.environ.get("GITHUB_API_URL", "https://api.github.com")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_a, **_k):
        return None


def _api(url, token, follow=True):
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28"})
    opener = urllib.request.build_opener(*([] if follow else [_NoRedirect]))
    return opener.open(req, timeout=60)


def _durations(a, token):
    """The archive is fetched from its redirect without the token: the storage
    host signs the URL itself and rejects a second credential."""
    try:
        with _api(a["archive_download_url"], token, follow=False) as r:
            blob = r.read()
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise
        with urllib.request.urlopen(e.headers["Location"], timeout=300) as r:
            blob = r.read()
    results = json.loads(zipfile.ZipFile(io.BytesIO(blob)).read("results.json"))
    return [{"test": r["test"], "seconds": r["seconds"]} for r in results
            if isinstance(r.get("test"), str)
            and isinstance(r.get("seconds"), (int, float))]


def main(repo, out):
    """Newest first, and on to the next one when an artifact has no durations:
    a cancelled run uploads its artifact before the sweep wrote results.json."""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("no GH_TOKEN")
        return 1
    this_run = os.environ.get("GITHUB_RUN_ID", "")
    with _api(f"{API}/repos/{repo}/actions/artifacts?name={ARTIFACT}"
              f"&per_page=30", token) as r:
        artifacts = json.load(r)["artifacts"]
    usable = [a for a in artifacts if not a["expired"]
              and str((a.get("workflow_run") or {}).get("id")) != this_run]
    usable.sort(key=lambda a: a["created_at"], reverse=True)
    for a in usable[:TRIES]:
        run = (a.get("workflow_run") or {}).get("id")
        try:
            timed = _durations(a, token)
        except Exception as e:
            print(f"artifact {a['id']} (run {run}): {e!r}")
            continue
        if not timed:
            print(f"artifact {a['id']} (run {run}) has no test durations")
            continue
        with open(out, "w") as f:
            json.dump(timed, f, indent=1)
        print(f"{out}: {len(timed)} durations from artifact {a['id']} "
              f"(run {run}, {a['created_at']})")
        return 0
    print(f"no durations in the newest {min(TRIES, len(usable))} of "
          f"{len(usable)} unexpired {ARTIFACT} artifacts")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    try:
        sys.exit(main(sys.argv[1], sys.argv[2]))
    except Exception as e:
        print(f"previous sweep not fetched: {e!r}")
        sys.exit(1)
