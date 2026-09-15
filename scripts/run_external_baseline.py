import sys

from defectfirst.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run-external-baseline", *sys.argv[1:]]))
