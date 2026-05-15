import traci
import time
from critilight.traffic_simulation.simulation import Env
from critilight.traffic_simulation.config import setup_sumo_environment
from critilight.llm_api import LLM
from critilight.metrics import Cal_Offline
from critilight.assessor_runtime import LightTrafficGAT
from critilight.output2prompt import *
import numpy as np
import json
import csv
import os
import pandas as pd
import torch
from pathlib import Path

setup_sumo_environment()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_legacy_path(value, default):
    raw = os.environ.get(value, default)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path

junctions_num = [0, 17, 19, 177]
device = torch.device("cpu")
NUM_NODES = int(os.environ.get("CRITILIGHT_NUM_NODES", "19"))
MODEL_PATH = _resolve_legacy_path("CRITILIGHT_MODEL_PATH", "assessor/checkpoints/traffic_gat_model_hz2.pth")
GRAPH_PATH = _resolve_legacy_path("CRITILIGHT_GRAPH_PATH", "assessor/checkpoints/traffic_graph_hz2.pkl")
SUMO_CFG_PATH = _resolve_legacy_path("CRITILIGHT_SUMO_CFG", "map/hangzhou/hangzhou.sumocfg")
NET_FILE_PATH = _resolve_legacy_path("CRITILIGHT_NET_FILE", "map/hangzhou/hangzhou.net.xml")
TRIPINFO_PATH = _resolve_legacy_path("CRITILIGHT_TRIPINFO_OUTPUT", "outputs/critilight/tripinfo.xml")
FCD_PATH = _resolve_legacy_path("CRITILIGHT_FCD_OUTPUT", "outputs/critilight/fcd_output.xml")
QUEUE_PATH = _resolve_legacy_path("CRITILIGHT_QUEUE_OUTPUT", "outputs/critilight/queue_output.xml")

model = LightTrafficGAT(
    num_nodes=NUM_NODES,
    pretrained_path=str(MODEL_PATH),
    graph_path=str(GRAPH_PATH),
    device=device,
)

performance_log = []
gamma = float(os.environ.get("CRITILIGHT_GAMMA", "0.5"))
sigma = float(os.environ.get("CRITILIGHT_SIGMA", "0.3"))

scenario_name = os.environ.get("CRITILIGHT_SCENARIO_NAME", "hangzhou2440")
method_name = os.environ.get("CRITILIGHT_METHOD_NAME", "CritiLight")
experiment_name = os.environ.get("CRITILIGHT_EXPERIMENT_NAME", "main")
note = f"sigma={sigma}"

def calculate_heterogeneous_risk_score(
        diff_sum,
        pre_decision,
        junctions_inlane_num,
        junctions_outlane_num,
        junctions_occupancy_num,
):

    score_dict = {}

    sorted_tls_ids = sorted(pre_decision.keys())

    for idx, tls_id in enumerate(sorted_tls_ids):
        mp_action = pre_decision.get(tls_id, 0)

        if isinstance(diff_sum, torch.Tensor):
            q_future = diff_sum[idx].item()
        elif isinstance(diff_sum, list):
            q_future = diff_sum[idx]
        else:
            q_future = 0.0

        try:
            occ_list = junctions_occupancy_num.get(tls_id, [])
            o_down = occ_list[mp_action] if mp_action < len(occ_list) else 0.0

            in_list = junctions_inlane_num.get(tls_id, [])
            n_in = in_list[mp_action] if mp_action < len(in_list) else 1.0

            out_list = junctions_outlane_num.get(tls_id, [])
            n_out = out_list[mp_action] if mp_action < len(out_list) else 1.0

        except IndexError:
            print(f"Error: Data mismatch for {tls_id}, action {mp_action}")
            score_dict[tls_id] = 0.0
            continue

        real_n_out = max(1.0, float(n_out))
        bottleneck_ratio = float(n_in) / real_n_out

        d_constraint = o_down * bottleneck_ratio

        score = q_future * (1.0 + gamma * d_constraint)

        score_dict[tls_id] = score

    return score_dict


def predict_and_filter_junctions(junction_state, pre_decision, queue_junctions_in, queue_junctions_out, junctions_inlane_num, junctions_outlane_num, junctions_occupancy_num, valid_phase_num):
    device = next(model.parameters()).device

    sorted_tls_ids = sorted(pre_decision.keys())

    max_phases = 4
    current_q_list = []
    mask_list = []

    for tls_id in sorted_tls_ids:
        real_qs = queue_junctions_in[tls_id]
        v_num = valid_phase_num[tls_id] // 2

        padded_qs = real_qs + [0] * (max_phases - len(real_qs))
        mask = [1] * v_num + [0] * (max_phases - v_num)

        current_q_list.append(padded_qs)
        mask_list.append(mask)

    current_queue_t = torch.tensor(current_q_list, dtype=torch.float32).to(device)
    mask_t = torch.tensor(mask_list, dtype=torch.float32).to(device)

    state_arr, action_arr, _ = model.proj_process(junction_state, pre_decision)

    state_t = torch.tensor(state_arr, dtype=torch.float32).to(device)
    action_t = torch.tensor(action_arr, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        pred_norm = model(state_t, action_t)

    pred_queue_real = pred2int(pred_norm, capacity=1)

    pred_valid = pred_queue_real * mask_t
    current_valid = current_queue_t * mask_t

    diff_t = pred_valid - current_valid
    diff_sum = torch.sum(diff_t, dim=1)
    scores = calculate_heterogeneous_risk_score(diff_sum, pre_decision, junctions_inlane_num, junctions_outlane_num,
                                                junctions_occupancy_num)

    total_junctions = len(sorted_tls_ids)
    k = max(1, int(total_junctions * sigma))

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_k_list = sorted_scores[:k]
    top_k_junctions = {tls_id: score for tls_id, score in top_k_list}

    prediction_result = {}

    if len(top_k_junctions) == 0:
        return top_k_junctions, prediction_result, scores

    for tls_id in top_k_junctions.keys():
        prediction_result[tls_id] = {}

        target_row_idx = sorted_tls_ids.index(tls_id)

        target_mask = mask_t[target_row_idx]
        target_current_q = current_queue_t[target_row_idx]

        num_phases = valid_phase_num.get(tls_id, 0)
        valid_phases = [i for i in range(num_phases) if i % 2 == 0]

        original_phase = pre_decision.get(tls_id, 0)

        for phase in valid_phases:
            pre_decision[tls_id] = phase // 2

            _, new_action_arr, _ = model.proj_process(junction_state, pre_decision)
            new_action_t = torch.tensor(new_action_arr, dtype=torch.float32).to(device)

            with torch.no_grad():
                new_pred_norm = model(state_t, new_action_t)

            new_pred_real = pred2int(new_pred_norm, capacity=1)
            target_pred_vec = new_pred_real[target_row_idx]

            target_pred_valid = target_pred_vec * target_mask
            target_current_valid = target_current_q * target_mask

            target_diff_vec = target_pred_valid - target_current_valid

            diff_score = torch.sum(target_diff_vec).item()

            prediction_result[tls_id][phase] = diff_score

        pre_decision[tls_id] = original_phase

    return top_k_junctions, prediction_result, scores


def execute_actions(env, final_actions):
    d_action = 30
    if len(final_actions) not in junctions_num:
        raise  ValueError(f"Not all junctions have action, current junctions num is {len(junctions_num)}")

    for controller in env.controllers:
        if controller.if_need():
            p_action = final_actions[controller.tls_id]
            controller.set_action(p_action, d_action)
            controller.yellow_serve_status = 'not'
            controller.last_action = p_action

def _get_valid_phases(tls_id):
        try:
            return len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
        except Exception:
            return 0


def start_sumo_simulation():
    start = time.perf_counter()

    for output_path in (TRIPINFO_PATH, FCD_PATH, QUEUE_PATH):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    traci.start([
        'sumo',
        '-c', str(SUMO_CFG_PATH),
        '--ignore-route-errors',
        '--tripinfo-output', str(TRIPINFO_PATH),
        '--tripinfo-output.write-unfinished',
        '--fcd-output', str(FCD_PATH),
        '--queue-output', str(QUEUE_PATH),
        '--time-to-teleport', '-1',
        '--collision.action', 'none',
    ])
    netfile = str(NET_FILE_PATH)
    env = Env(netfile)
    traffic_lights = env.traffic_lights
    graph = env.graph
    print(f"traffic lights: {traffic_lights}")

    LLM_ins = LLM()
    step = 0
    total_queue = 0

    valid_phase_num = {}
    for controller in env.controllers:
        valid_phase_num[controller.tls_id] = _get_valid_phases(controller.tls_id)

    while step < 1800:
        print(f"step is {step}")
        traci.simulationStep()

        queue_number = 0
        queue_junctions_in = {}
        queue_junctions_out = {}
        junctions_inlane_num = {}
        junctions_outlane_num = {}
        junctions_occupancy_num = {}
        junction_state = {}
        pre_actions = {}

        for controller in env.controllers:
            queue_number += controller.get_total_queue_length(distance=100)

            if controller.if_need():
                if len(controller.phases) % 2 == 1:
                    continue

                pressure, queue_in, queue_out, inlane_num, outlane_num, downstream_occupancy = controller.get_pressure(valid_phase_num)

                queue_junctions_in[controller.tls_id] = queue_in
                queue_junctions_out[controller.tls_id] = queue_out
                junctions_inlane_num[controller.tls_id] = inlane_num
                junctions_outlane_num[controller.tls_id] = outlane_num
                junctions_occupancy_num[controller.tls_id] = downstream_occupancy

                p_action = np.argmax(pressure)
                phase_queue_lengths, phase_movements = controller.get_phase_queue_and_vehicles(distance=100)

                even_key_dict = {k: v for k, v in phase_queue_lengths.items() if k % 2 == 0}

                structure = controller.format_structure_prompt()
                controller.structure = structure
                controller.previous_prompt, controller.prompt_for_reviewer_log = LLM_ins.action(even_key_dict,
                                                                                                phase_movements,
                                                                                                structure, controller)

                junction_state[controller.tls_id] = even_key_dict
                pre_actions[controller.tls_id] = p_action
                controller.prompt_for_reviewer_log_last_step = controller.prompt_for_reviewer_log

        if len(pre_actions) != 0:

            LLM_revise = {}

            top_k_junctions, prediction_result, scores = predict_and_filter_junctions(
                junction_state, pre_actions, queue_junctions_in, queue_junctions_out,
                junctions_inlane_num, junctions_outlane_num, junctions_occupancy_num, valid_phase_num
            )

            if len(prediction_result) != 0:
                for tls_id, phase_dict in prediction_result.items():
                    index = traffic_lights.index(tls_id)
                    controller = env.controllers[index]
                    structure = controller.structure
                    action_text = controller.action_text
                    neighbors = list(graph.neighbors(tls_id))

                    revise_phase = LLM_ins.revise_action(env.controllers, index, structure, action_text, pre_actions,
                                                         phase_dict, traffic_lights, neighbors, junction_state,
                                                         junctions_occupancy_num)
                    if revise_phase is not None:
                        LLM_revise[tls_id] = [int(pre_actions[tls_id])]
                        pre_actions[tls_id] = revise_phase // 2
                        LLM_revise[tls_id].append(revise_phase // 2)
                print(f"LLM_revise(pre_actions, new_actions): {LLM_revise}")

            execute_actions(env, pre_actions)

        step += 1
        num_controllers = len(env.controllers)
        if num_controllers > 0:
            total_queue += queue_number / num_controllers

    avg_queue = total_queue / step if step > 0 else 0
    traci.close()

    Cal_Offline(avg_queue, scenario_name, method_name, experiment_name, note, str(TRIPINFO_PATH),
                str(QUEUE_PATH))

    end = time.perf_counter()
    print(f"Elapsed time: {end - start:.3f} s")

def run_critilight():
    start_sumo_simulation()


if __name__ == "__main__":
    start = time.perf_counter()
    start_sumo_simulation()

