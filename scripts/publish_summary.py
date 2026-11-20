"""Publish the short article as a first-class page in the static site."""
import argparse
from html.parser import HTMLParser
import html
from pathlib import Path
import re
import shutil
from urllib.parse import unquote, urlsplit

GOATCOUNTER = 'https://aloysius109.goatcounter.com/count'


class ImageSources(HTMLParser):
    def __init__(self):
        super().__init__()
        self.sources = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            source = dict(attrs).get('src')
            if source:
                self.sources.append(source)


def publish_summary_html(source_html, output_root, canonical_url, images_dir=None):
    text = re.sub(r'(?i)\s*(?:&mdash;|&#8212;|—|–|-)?\s*Medium edition\b', '', source_html)
    canonical = f'<link rel="canonical" href="{html.escape(canonical_url, quote=True)}">'
    canonical_pattern = re.compile(r'<link\b(?=[^>]*\brel=["\']canonical["\'])[^>]*>', re.I)
    if canonical_pattern.search(text):
        text = canonical_pattern.sub(canonical, text, count=1)
    else:
        text = re.sub(r'</head\s*>', canonical + '\n</head>', text, count=1, flags=re.I)
    nav_styles = '''<style id="summary-navigation-style">
.summary-site-nav{display:flex;align-items:center;justify-content:space-between;gap:16px;max-width:860px;margin:0 auto;padding:16px 24px;border-bottom:1px solid #dfdfda;font:14px/1.4 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif}
.summary-site-nav a{color:#245b50;text-decoration:none;font-weight:650}
.summary-site-nav a:last-child{padding:9px 13px;border:1px solid #b8c9bf;border-radius:5px}
.summary-site-nav a:hover{text-decoration:underline;text-underline-offset:3px}
@media(max-width:600px){.summary-site-nav{padding:13px 18px;font-size:13px}}
</style>'''
    text = re.sub(r'</head\s*>', nav_styles + '\n</head>', text, count=1, flags=re.I)
    nav = ('<nav class="summary-site-nav" aria-label="Report navigation">'
           '<a href="https://aloysius109.github.io/">Kathryn Hopkins</a>'
           f'<a href="{html.escape(canonical_url, quote=True)}">Full report ↗</a></nav>')
    text, inserted = re.subn(r'<body\b[^>]*>', lambda match: match.group(0) + nav, text, count=1, flags=re.I)
    if not inserted:
        raise ValueError('Short article has no body element')
    if 'data-goatcounter=' not in text:
        analytics = f'<script data-goatcounter="{GOATCOUNTER}" async src="//gc.zgo.at/count.js"></script>'
        text = re.sub(r'</body\s*>', analytics + '\n</body>', text, count=1, flags=re.I)
    summary_dir = Path(output_root) / 'summary'
    if summary_dir.exists():
        shutil.rmtree(summary_dir)
    summary_dir.mkdir(parents=True)
    parser = ImageSources()
    parser.feed(text)
    copied = []
    for source in parser.sources:
        parsed = urlsplit(source)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        relative = Path(unquote(parsed.path))
        if relative.parts and relative.parts[0] == 'images':
            relative = Path(*relative.parts[1:])
        if not relative.parts or images_dir is None:
            if parsed.path and not parsed.path.startswith('/'):
                raise ValueError(f'No image source directory configured for {source}')
            continue
        source_path = Path(images_dir) / relative
        if not source_path.is_file():
            raise FileNotFoundError(f'Missing short-article image: {source_path}')
        target = summary_dir / 'images' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        copied.append(target)
    page = summary_dir / 'index.html'
    page.write_text(text, encoding='utf-8', newline='\n')
    if 'medium edition' in text.lower():
        raise ValueError('Published summary still refers to a Medium edition')
    return page, copied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--article', required=True, type=Path)
    parser.add_argument('--images', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--canonical', required=True)
    args = parser.parse_args()
    page, images = publish_summary_html(args.article.read_text(encoding='utf-8'), args.output, args.canonical, args.images)
    print(f'Published {page} with {len(images)} local images')


if __name__ == '__main__':
    main()