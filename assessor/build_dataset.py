import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Build assessor dataset")
    parser.add_argument("--episodes", type=int, help="Number of collection episodes.")
    parser.add_argument("--max-steps", type=int, help="Override the maximum simulation steps per episode.")
    parser.add_argument("--decision-step", type=int, help="Override the control decision interval.")
    parser.add_argument("--sumocfg", help="Path to the SUMO .sumocfg file.")
    parser.add_argument("--net-file", help="Path to the SUMO network file.")
    parser.add_argument("--graph-out", help="Path to save the exported graph pickle.")
    parser.add_argument("--dataset-out", help="Path to save the exported dataset pickle.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.episodes is not None:
        os.environ["CRITILIGHT_DC_NUM_EPISODES"] = str(args.episodes)
    if args.max_steps is not None:
        os.environ["CRITILIGHT_DC_MAX_EP_STEPS"] = str(args.max_steps)
    if args.decision_step is not None:
        os.environ["CRITILIGHT_DC_DECISION_STEP"] = str(args.decision_step)
    if args.sumocfg:
        os.environ["CRITILIGHT_DC_SUMO_CFG"] = args.sumocfg
    if args.net_file:
        os.environ["CRITILIGHT_DC_NET_FILE"] = args.net_file
    if args.graph_out:
        os.environ["CRITILIGHT_DC_GRAPH_OUT"] = args.graph_out
    if args.dataset_out:
        os.environ["CRITILIGHT_DC_DATASET_OUT"] = args.dataset_out
    from assessor.dataset_builder import build_dataset
    build_dataset()


if __name__ == "__main__":
    main()
