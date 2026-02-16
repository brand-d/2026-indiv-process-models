import sys
import ccobra

sys.argv.append("benchmark.json")
ccobra.benchmark.runner.entry_point()