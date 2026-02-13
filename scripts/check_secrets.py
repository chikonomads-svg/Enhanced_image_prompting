"""Simple pre-commit secret checker.

Scans staged files for common secret patterns and prevents commits if found.
This is a lightweight guard — for stronger protection consider using
pre-commit hooks like `detect-secrets` or `git-secrets`.
"""
import re
import sys
from pathlib import Path

PATTERNS = [
    re.compile(r"AZURE_OPENAI_API_KEY\s*=\s*['\"]?[A-Za-z0-9-_]{20,}['\"]?"),
    re.compile(r"TAVILY_API_KEY\s*=\s*['\"]?tvly-[A-Za-z0-9_-]{8,}['\"]?"),
    re.compile(r"(sk-|api_key\s*=|AZURE_).*['\"]?[A-Za-z0-9-_]{20,}['\"]?"),
]


def scan_file(path: Path):
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return None
    for p in PATTERNS:
        if p.search(text):
            return True
    return False


def main():
    # If no filenames passed, scan all files (fallback)
    files = sys.argv[1:] or [str(p) for p in Path('.').rglob('*') if p.is_file()]
    leaked = []
    for f in files:
        if scan_file(Path(f)):
            leaked.append(f)
    if leaked:
        print("Potential secrets found in the following files:")
        for l in leaked:
            print("  -", l)
        print('\nPlease remove secrets and use environment variables or .env (gitignored).')
        sys.exit(1)


if __name__ == '__main__':
    main()
