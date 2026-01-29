#!/bin/bash

# Arquivos principais
OLD_FILE="original_main_v2.tex"
NEW_FILE="main.tex"
DIFF="track_changes_SmaxPathways_v5"
DIFF_FILE="$DIFF.tex"

# PDF viewer (Linux: xdg-open, Mac: open, Windows: start)
PDF_VIEWER="xdg-open"

# Lista de arquivos auxiliares a limpar
AUX_FILES=("$DIFF.aux" "$DIFF.log" "$DIFF.bbl" "$DIFF.blg" "$DIFF.out" "$DIFF.toc" "$DIFF.lof" "$DIFF.lot" "$DIFF.fls" "$DIFF.fdb_latexmk")

# Gerar arquivo com diferenças
echo "Gerando arquivo de diferenças com latexdiff..."
latexdiff "$OLD_FILE" "$NEW_FILE" > "$DIFF_FILE"

# Compilar diff.tex
echo "Compilando diff.tex..."
pdflatex "$DIFF_FILE"
bibtex "$DIFF.aux"
pdflatex "$DIFF_FILE"
pdflatex "$DIFF_FILE"

# Limpar arquivos auxiliares antigos
echo "Limpando arquivos auxiliares antigos..."
for f in "${AUX_FILES[@]}"; do
    [ -f "$f" ] && rm -f "$f"
done

echo "Build concluído!"

