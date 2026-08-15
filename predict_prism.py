from __future__ import absolute_import, division, print_function

import argparse
import runpy
import sys


COMMANDS = {
    "depth-c3vd": {
        "module": "reference.depth.depth_evaluate_max_norm",
        "help": "depth prediction/evaluation for C3VD or C3VD-style folders",
    },
    "depth-endomapper": {
        "module": "reference.depth.depth_evaluate_endomapper",
        "help": "EndoMapper-style depth prediction at full output resolution",
    },
    "depth-endomapper-288": {
        "module": "reference.depth.depth_evaluate_endomapper_288",
        "help": "EndoMapper-style depth prediction at 288x288 output resolution",
    },
    "pose-c3vd": {
        "module": "reference.pose_predict_feast_v1",
        "help": "pose prediction export for C3VD-style sequences",
    },
}


def parse_command(argv):
    parser = argparse.ArgumentParser(
        description="Unified PRISM prediction and evaluation entrypoint",
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments passed through to the selected prediction script",
    )
    return parser.parse_args(argv)


def main():
    args = parse_command(sys.argv[1:])
    module = COMMANDS[args.command]["module"]
    sys.argv = [module + ".py"] + args.args
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
