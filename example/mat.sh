
# $1: plasmid fasta file
# $2: chr fasta file
# $3: chunk size (500)

# process plasmid
awk '/^>/ { print (NR==1 ? "" : RS) $0; next } { printf "%s", $0 } END { printf RS }' $1 > $1.tmp
sed -n '1~2!p' $1.tmp | awk 'NF' > $1.seq
tr '\n' '\r' < $1.seq | fold -w 500 | tr '\n\r' '\n' | fold -w 500 > $1.chunk
sed 's/.*/&,1/' $1.chunk > $1.final
sed -r '/^.{,500}$/d' $1.final > $1.final.long
sort -R $1.final.long | head -n 500 >$1.final.long.500

# process chr
awk '/^>/ { print (NR==1 ? "" : RS) $0; next } { printf "%s", $0 } END { printf RS }' $2 > $2.tmp
sed -n '1~2!p' $2.tmp | awk 'NF' > $2.seq
tr '\n' '\r' < $2.seq | fold -w 500 | tr '\n\r' '\n' | fold -w 500 > $2.chunk
sed 's/.*/&,0/' $2.chunk > $2.final
sed -r '/^.{,500}$/d' $2.final > $2.final.long
sort -R $2.final.long | head -n 500 >$2.final.long.500

echo "sequence,target" > train.csv
cat $1.final.long.500 $2.final.long.500 >> train.csv