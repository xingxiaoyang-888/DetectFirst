import sys

from defectfirst.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate-handoff", *sys.argv[1:]]))
