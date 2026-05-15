import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Run the CritiLight online decision pipeline.")
    parser.add_argument("--sumocfg", help="Path to the SUMO .sumocfg file.")
    parser.add_argument("--net-file", help="Path to the SUMO network file.")
    parser.add_argument("--model-path", help="Path to the runtime assessor checkpoint.")
    parser.add_argument("--graph-path", help="Path to the runtime assessor graph pickle.")
    parser.add_argument("--llm-backend", choices=["local", "remote"], help="Select the LLM backend.")
    parser.add_argument("--llm-model-path", help="Path to the local fine-tuned LLM directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.sumocfg:
        os.environ["CRITILIGHT_SUMO_CFG"] = args.sumocfg
    if args.net_file:
        os.environ["CRITILIGHT_NET_FILE"] = args.net_file
    if args.model_path:
        os.environ["CRITILIGHT_MODEL_PATH"] = args.model_path
    if args.graph_path:
        os.environ["CRITILIGHT_GRAPH_PATH"] = args.graph_path
    if args.llm_backend:
        os.environ["CRITILIGHT_LLM_BACKEND"] = args.llm_backend
    if args.llm_model_path:
        os.environ["CRITILIGHT_LLM_MODEL_PATH"] = args.llm_model_path
    from critilight.pipeline import run_critilight
    run_critilight()


if __name__ == "__main__":
    main()
