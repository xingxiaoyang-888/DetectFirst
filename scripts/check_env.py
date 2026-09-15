import sys

from defectfirst.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-env", *sys.argv[1:]]))
