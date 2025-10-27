#!/bin/bash

rm -f Figure2.csv Figure3.csv

SIZE_F2=50
STEPS_F2=1000
T_F2=0.65

python ../lebo/core.py $STEPS_F2 $SIZE_F2 $T_F2 0 > /dev/null
FILE=$(ls -t LL-Output-* | head -n 1)
echo "MCS,Energy,Order" > Figure2.csv
awk '!/^#/ {print $1","$3","$4}' "$FILE" >> Figure2.csv

SIZE_F3=20
STEPS_F3=2000
BURN=$((STEPS_F3/2))

echo "Temperature,OrderMean,OrderStd" > Figure3.csv
for T in $(seq 0.0 0.05 1.60); do
  python ../lebo/core.py $STEPS_F3 $SIZE_F3 $T 0 > /dev/null
  FILE=$(ls -t LL-Output-* | head -n 1)
  read M SD < <(awk -v burn=$BURN '!/^#/ {if ($1>=burn){n++; s+=$4; ss+=$4*$4}} END{if(n>1){m=s/n; sd=sqrt((ss-s*s/n)/(n-1)); printf "%.6f %.6f", m, sd} else {printf "nan nan"}}' "$FILE")
  echo "$T,$M,$SD" >> Figure3.csv
done

echo "Figure2.csv and Figure3.csv written."
