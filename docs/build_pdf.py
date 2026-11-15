"""Gera docs/relatorio.pdf a partir de docs/relatorio_transformacoes.md (links clicáveis).

    pip install markdown
    python docs/build_pdf.py --repo-url https://github.com/<usuario>/<repo>

Usa o Edge ou o Chrome já instalado (modo headless) para imprimir em PDF, então não
precisa de LaTeX nem de pandoc. Links relativos do markdown (outros arquivos do
repositório) viram links absolutos para o GitHub quando --repo-url é informado.
Também gera docs/arquitetura.png a partir do SVG.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

DOCS = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1c2530; }
h1 { font-size: 20pt; margin: 0 0 6pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #c9d1da; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; break-after: avoid; }
a { color: #1a56c4; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 10pt; font-size: 9pt; }
th, td { border: 1px solid #c9d1da; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eef2f7; }
tr { break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f2f4f7; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f2f4f7; padding: 6pt 8pt; border-radius: 4pt; font-size: 8.5pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
img { max-width: 100%; }
td img { max-width: 100%; display: block; }
blockquote { margin: 6pt 0; padding: 2pt 10pt; border-left: 3px solid #c9d1da; color: #44505f; }
"""

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_browser() -> str:
    for candidate in BROWSERS:
        if Path(candidate).exists():
            return candidate
    for name in ("msedge", "microsoft-edge", "google-chrome", "chromium", "chromium-browser"):
        if shutil.which(name):
            return shutil.which(name)
    sys.exit("Edge/Chrome não encontrado. Abra o HTML gerado e use Imprimir > Salvar como PDF.")


def headless(browser: str, output: Path, *args: str) -> None:
    """Roda o navegador e espera `output` aparecer.

    O Edge devolve o controle antes de terminar de gravar o arquivo, então não basta
    esperar o processo: o arquivo antigo é apagado e a função aguarda o novo estabilizar.
    """
    output.unlink(missing_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="relatorio-"))
    try:
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}", *args],
            check=True,
            capture_output=True,
            timeout=180,
        )
        deadline, last_size = time.time() + 120, -1
        while time.time() < deadline:
            size = output.stat().st_size if output.exists() else 0
            if size > 0 and size == last_size:
                return
            last_size = size
            time.sleep(1)
        sys.exit(f"O navegador não gerou {output}. Abra o HTML e use Imprimir > Salvar como PDF.")
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def absolutize_links(html: str, repo_url: str | None) -> str:
    """Links relativos para arquivos do repositório -> URL do GitHub (clicável no PDF)."""
    if not repo_url:
        return html
    base = repo_url.rstrip("/") + "/blob/main/docs/"

    def fix(match: re.Match) -> str:
        href = match.group(1)
        if re.match(r"^(https?:|mailto:|#)", href):
            return match.group(0)
        return f'href="{base}{href}"'

    return re.sub(r'href="([^"]+)"', fix, html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relatório em PDF com links clicáveis")
    parser.add_argument("--source", default=str(DOCS / "relatorio_transformacoes.md"))
    parser.add_argument("--out", default=str(DOCS / "relatorio.pdf"))
    parser.add_argument("--repo-url", help="URL do repositório público, para os links relativos")
    args = parser.parse_args()

    browser = find_browser()
    source = Path(args.source)

    svg = DOCS / "arquitetura.svg"
    if svg.exists():
        headless(
            browser,
            DOCS / "arquitetura.png",
            f"--screenshot={DOCS / 'arquitetura.png'}",
            "--window-size=1280,580",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            svg.as_uri(),
        )

    body = markdown.markdown(source.read_text(encoding="utf-8"), extensions=["tables", "fenced_code", "sane_lists"])
    body = absolutize_links(body, args.repo_url)
    html = (
        f'<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">'
        f"<title>Relatório de transformações</title><style>{CSS}</style></head><body>{body}</body></html>"
    )
    html_path = source.with_suffix(".html")  # mesma pasta: caminhos relativos das imagens funcionam
    html_path.write_text(html, encoding="utf-8")
    out = Path(args.out).resolve()
    headless(browser, out, f"--print-to-pdf={out}", "--no-pdf-header-footer", html_path.as_uri())
    print(f"PDF gerado em {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
