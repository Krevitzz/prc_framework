# main.py
import sys
from pathlib import Path

from .collector import collect_all
from .html_renderer import render_html

def main():
    root_path = sys.argv[1] if len(sys.argv) > 1 else '.'
    # Option pour collecter les appels (non implémentée complètement)
    collect_calls = True
    if '--no-collect-calls' in sys.argv:
        collect_calls = False

    print(f"[generate_architecture_v2] Scan : {root_path}")
    data = collect_all(root_path, collect_calls)

    # Écriture du HTML
    output_dir = Path(__file__).parent
    html_path = output_dir / 'architecture_v2.html'
    html = render_html(data)
    html_path.write_text(html, encoding='utf-8')
    print(f"  → {html_path}")

    # Optionnel : générer requirements.txt (similaire à l'ancien, mais sans versions)
    # On peut le faire plus tard si besoin.

    print("[generate_architecture_v2] OK")

if __name__ == '__main__':
    main()
