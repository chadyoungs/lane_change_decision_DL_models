import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conscend.scenario_extraction import ConScenDExtractor
from configs.constant import DATASET_ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ConScenD-style OpenX scenarios from highD.")
    parser.add_argument(
        "--recordings",
        nargs="*",
        type=int,
        help="Specific highD recording numbers to process, e.g. --recordings 1 2 3",
    )
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Path to the highD dataset root directory.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "conscend"),
        help="Directory for generated metadata, OpenSCENARIO, and OpenDRIVE files.",
    )
    return parser.parse_args()


def main():
    arguments = parse_args()
    extractor = ConScenDExtractor(arguments.dataset_root, arguments.output_root)
    scenarios = extractor.run(arguments.recordings)
    print(f"Generated {len(scenarios)} ConScenD-style scenarios in {arguments.output_root}")


if __name__ == "__main__":
    main()
