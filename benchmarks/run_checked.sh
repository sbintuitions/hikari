#!/bin/sh
# Run a command and print the macOS Low Power Mode state before and after (it skews GPU benchmarks ~2.5x).
lpm() { pmset -g | awk '/lowpowermode/ {print $2}'; }
echo "[lowpowermode before: $(lpm)]"
"$@"
echo "[lowpowermode after: $(lpm)]"
