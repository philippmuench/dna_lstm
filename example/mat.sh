
# $1: plasmid fasta file
# $2: chr fasta file

# process plasmid

awk '/^>/ { print (NR==1 ? "" : RS) $0; next } { printf "%s", $0 } END { printf RS }' $1 > $1.tmp
sed -n '1~2!p' $1.tmp | awk '/^>/ {print($0)}; /^[^>]/ {print(toupper($0))}' |awk 'NF' > $1.seq
rm $1.tmp
tr '\n' '\r' < $1.seq | fold -w 150 | tr '\n\r' '\n' | fold -w 150 > $1.chunk
rm $1.seq
sed 's/.*/&,1/' $1.chunk > $1.final
rm $1.chunk
sed -r '/^.{,150}$/d' $1.final > $1.final.long
rm $1.final
sort -R $1.final.long | grep -iv "n" | grep -iv "y" | grep -iv "r"  | grep -iv "k" | grep -iv "m"| grep -iv "s" | grep -iv "w" | awk 'length($0)>150' >$1.final.long.r
rm $1.final.long
nr=$(grep -c ">" "$1.final.long.r")
echo $nr


# process chr
awk '/^>/ { print (NR==1 ? "" : RS) $0; next } { printf "%s", $0 } END { printf RS }' $2 > $2.tmp
sed -n '1~2!p' $2.tmp | awk '/^>/ {print($0)}; /^[^>]/ {print(toupper($0))}' | awk 'NF' > $2.seq
rm $2.tmp
tr '\n' '\r' < $2.seq | fold -w 150 | tr '\n\r' '\n' | fold -w 150 > $2.chunk
rm $2.seq
sed 's/.*/&,0/' $2.chunk > $2.final
rm $2.chunk
sed -r '/^.{,150}$/d' $2.final > $2.final.long
rm $2.final
sort -R $2.final.long | grep -iv "n" | grep -iv "y" | grep -iv "r"  | grep -iv "k" | grep -iv "m"| grep -iv "s" | grep -iv "w" | awk 'length($0)>150' >$2.final.long.r
rm $2.final.long

cat $1.final.long.r $2.final.long.r > train.csv
sort -R train.csv > train_rand.csv
sed -i -e '1isequence,target\' train_rand.csv
rm $1.final.long.r
rm $2.final.long.r
