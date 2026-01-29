#/bin/bash

# run this code on Ocean

input_dir="$1"
output_file="${1}.mp4"

mkdir tmp_pngs

n=0;

for f in $(ls "$input_dir"/*.png | sort); do
	cp "$f" $(printf "tmp_pngs/frame_%04d.png" $n);   
	n=$((n+1)); 
done

# Create video
ffmpeg -framerate 2 -i tmp_pngs/frame_%04d.png \
    -vf "scale=iw:ih:flags=lanczos,pad=ceil(iw/2)*2:ceil(ih/2)*2" \
    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p "$output_file"


# clean tmp_pngs
rm -r tmp_pngs
