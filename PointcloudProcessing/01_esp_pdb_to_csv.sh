regex="^([[:digit:]]+)(_vtx\.pdb)"
for file in *.pdb; do
    if [[ $file =~ $regex ]]; then
	new_name="${BASH_REMATCH[1]}.csv"
	echo "processing: $file"
        cut -c 26-54,61-66 $file> tmp.tsv
	awk 'BEGIN {OFS = ","; print "x,y,z,esp"} ; NR > 2 {print substr($0,6,8),substr($0,14,8),substr($0,22,8),substr($0,30,8)}' tmp.tsv > tmp.csv
	mv "tmp.csv" "processed/"$new_name
	rm tmp.tsv
    fi
done
