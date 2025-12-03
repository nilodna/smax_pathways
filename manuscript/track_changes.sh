#!/bin/bash

# Arquivos principais
OLD_FILE="original_main.tex"
NEW_FILE="main.tex"
DIFF_FILE="diff.tex"

# PDF viewer (Linux: xdg-open, Mac: open, Windows: start)
PDF_VIEWER="xdg-open"

# Lista de arquivos auxiliares a limpar
AUX_FILES=("diff.aux" "diff.log" "diff.bbl" "diff.blg" "diff.out" "diff.toc" "diff.lof" "diff.lot" "diff.fls" "diff.fdb_latexmk")

# Gerar arquivo com diferenças
echo "Gerando arquivo de diferenças com latexdiff..."
latexdiff "$OLD_FILE" "$NEW_FILE" > "$DIFF_FILE"

# Compilar diff.tex
echo "Compilando diff.tex..."
pdflatex "$DIFF_FILE"
bibtex diff
pdflatex "$DIFF_FILE"
pdflatex "$DIFF_FILE"

# Limpar arquivos auxiliares antigos
echo "Limpando arquivos auxiliares antigos..."
for f in "${AUX_FILES[@]}"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "Build concluído!"

