#!/bin/bash

# Nome do arquivo principal (sem extensão)
MAIN="main"
PDF_VIEWER="xdg-open"  # Linux; no MacOS use "open", no Windows "start"

# Lista de arquivos auxiliares para remover
AUX_FILES=("$MAIN.aux" "$MAIN.log" "$MAIN.bbl" "$MAIN.blg" "$MAIN.out" \
           "$MAIN.toc" "$MAIN.lof" "$MAIN.lot" "$MAIN.fls" "$MAIN.fdb_latexmk")

# Compilação inicial
echo "Compilando $MAIN.tex (1/3)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"

# Rodar bibliografia, se existir
if [ -f "$MAIN.aux" ]; then
    echo "Compilando bibliografia..."
    bibtex "$MAIN"
fi

# Compila novamente para atualizar referências
echo "Compilando $MAIN.tex (2/3)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"
echo "Compilando $MAIN.tex (3/3)..."
pdflatex -interaction=nonstopmode "$MAIN.tex"


# Limpar arquivos auxiliares
echo "Limpando arquivos auxiliares antigos..."
for f in "${AUX_FILES[@]}"; do
    [ -f "$f" ] && rm -f "$f"
done

## Abrir PDF
#if [ -f "$MAIN.pdf" ]; then
#    echo "Abrindo PDF..."
#    $PDF_VIEWER "$MAIN.pdf" &
#else
#    echo "PDF não encontrado! Algo deu errado na compilação."
#fi

# cp "main.pdf" "SmaxPathways_v3.pdf"

echo "Build concluído!"

