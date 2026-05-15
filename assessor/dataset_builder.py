import math
import os
import pickle
import time
from pathlib import Path

import numpy as np
import sumolib
import torch
import traci
import xml.etree.ElementTree as ET

from assessor.baselines.maxpressure import MaxPressureAgent
from assessor.env_wrapper import RL_EnvWrapper
from assessor.traffic_simulation.config import setup_sumo_environment
from assessor.traffic_simulation.simulation import Env

setup_sumo_environment()
PROJECT_ROOT = Path(__file__).resolve().parent


def _resolve_local_path(value, default):
    raw = os.environ.get(value, default)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


NUM_INTERSECTIONS = int(os.environ.get("CRITILIGHT_DC_NUM_INTERSECTIONS", "17"))
DECISION_STEP = int(os.environ.get("CRITILIGHT_DC_DECISION_STEP", "36"))
MAX_EP_STEPS = int(os.environ.get("CRITILIGHT_DC_MAX_EP_STEPS", "1800"))
NUM_EPISODES = int(os.environ.get("CRITILIGHT_DC_NUM_EPISODES", "1"))

SUMO_CMD = [
    'sumo',
    '-c', str(_resolve_local_path("CRITILIGHT_DC_SUMO_CFG", "../map/jinan/jinan.sumocfg")),
    '--ignore-route-errors',
    '--time-to-teleport', '-1',
    '--collision.action', 'none',
]
NET_FILE = str(_resolve_local_path("CRITILIGHT_DC_NET_FILE", "../map/jinan/jinan.net.xml"))


class TrainWrapper:
    def __init__(self, base_env, decision_step, net_file):
        self.env = base_env
        self.decision_step = decision_step
        self.net = sumolib.net.readNet(net_file)

        self.num_intersections = len(self.env.controllers)

        self.valid_phases = {tls_id: self._get_valid_phases(tls_id) // 2 for tls_id in self.env.traffic_lights}
        self.valid_nums = [self.valid_phases[tls_id] for tls_id in self.env.traffic_lights]

        self.controller_projection = {}

        self.lane_structure = self._build_lane_structure()
        print(f"lane_structure:{self.lane_structure}")

        self.phase2vec_mapping = self.build_phase_mapping_with_traci()
        print(f"phase2vec_mapping:{self.phase2vec_mapping}")

    def _get_valid_phases(self, tls_id):
            try:
                return len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
            except Exception:
                return 0

    def _build_lane_structure(self):
        mapping = {}

        for controller in self.env.controllers:
            tls_id = controller.tls_id
            node = self.net.getNode(tls_id)
            incoming_edges = node.getIncoming()

            direction_slots = {
                0: {0: [], 1: [], 2: []},
                1: {0: [], 1: [], 2: []},
                2: {0: [], 1: [], 2: []},
                3: {0: [], 1: [], 2: []}
            }


            SLOT_CENTERS = {
                0: 90,
                3: 180,
                2: 270,
                1: 0
            }

            edge_candidates = []
            occupied_slots = set()
            self.controller_projection[controller.tls_id] = {}

            for edge in incoming_edges:
                shape = edge.getShape()
                p1, p2 = shape[-2], shape[-1]
                angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
                if angle < 0: angle += 360

                edge_candidates.append({'edge': edge, 'angle': angle})

            for item in edge_candidates:
                angle = item['angle']
                edge = item['edge']

                if 45 <= angle < 135:
                    ideal_idx = 0
                elif 135 <= angle < 225:
                    ideal_idx = 3
                elif 225 <= angle < 315:
                    ideal_idx = 2
                else:
                    ideal_idx = 1

                target_idx = -1

                if ideal_idx not in occupied_slots:
                    target_idx = ideal_idx

                else:
                    min_dist = 360.0
                    best_backup_idx = -1

                    for candidate_idx, center_angle in SLOT_CENTERS.items():
                        if candidate_idx not in occupied_slots:
                            diff = abs(angle - center_angle)
                            if diff > 180:
                                diff = 360 - diff

                            if diff < min_dist:
                                min_dist = diff
                                best_backup_idx = candidate_idx

                    if best_backup_idx != -1:
                        target_idx = best_backup_idx
                    else:
                        raise ValueError(f"Failed to map incoming edge {edge.getID()} to a direction slot.")


                if target_idx != -1:
                    occupied_slots.add(target_idx)
                    self.controller_projection[controller.tls_id][angle] = target_idx

                for lane in edge.getLanes():
                    lane_id = lane.getID()

                    connections = lane.getOutgoing()

                    is_left = False
                    is_straight = False
                    is_right = False

                    for conn in connections:
                        dir_char = conn.getDirection()
                        if dir_char in ['s']:
                            is_straight = True
                        elif dir_char in ['l', 't', 'L']:
                            is_left = True
                        elif dir_char in ['r', 'R']:
                            is_right = True

                    if is_left:
                        direction_slots[target_idx][0].append(lane_id)
                    if is_straight:
                        direction_slots[target_idx][1].append(lane_id)
                    if is_right:
                        direction_slots[target_idx][2].append(lane_id)

            mapping[tls_id] = direction_slots

        return mapping

    def build_phase_mapping_with_traci(self):

        mapping = {}

        for controller in self.env.controllers:
            tls_id = controller.tls_id
            mapping[tls_id] = {}

            links = traci.trafficlight.getControlledLinks(tls_id)

            logic = traci.trafficlight.getAllProgramLogics(tls_id)[0]

            for phase_idx, phase in enumerate(logic.phases):
                if phase_idx % 2 != 0:
                    continue

                mapping[tls_id][phase_idx // 2] = {}
                state_str = phase.state
                vec = [0] * 12

                for i, char in enumerate(state_str):
                    if char.lower() == 'g':
                        if i < len(links) and len(links[i]) > 0:
                            controlled_link = links[i][0]

                            lane_in_id = controlled_link[0]
                            lane_out_id = controlled_link[1]

                            edge_in_shape = traci.lane.getShape(lane_in_id)
                            edge_out_shape = traci.lane.getShape(lane_out_id)

                            slot, turn = self._get_movement_type_by_shape(tls_id, edge_in_shape, edge_out_shape)
                            vec_idx = slot * 3 + turn
                            if 0 <= vec_idx < 12:
                                vec[vec_idx] = 1

                mapping[tls_id][phase_idx // 2] = vec
        return mapping

    def _get_movement_type_by_shape(self, tls_id, shape_in, shape_out):
        p1, p2 = shape_in[-2], shape_in[-1]
        angle_in = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        if angle_in < 0: angle_in += 360

        slot = -1
        for angle, target_idx in self.controller_projection[tls_id].items():
            if abs(angle_in - angle) <= 5:
                slot = target_idx
                break

        if slot == -1:
            raise ValueError(f"Slot error")

        p3, p4 = shape_out[0], shape_out[1]
        angle_out = math.degrees(math.atan2(p4[1] - p3[1], p4[0] - p3[0]))
        if angle_out < 0: angle_out += 360

        diff = (angle_out - angle_in) % 360

        if diff > 315 or diff < 45:
            turn = 1
        elif 45 <= diff < 135:
            turn = 0
        elif 225 <= diff < 315:
            turn = 2
        else:
            turn = 0

        return slot, turn

    def get_adj_matrix(self):
        controllers = self.env.controllers
        target_node_ids = [c.tls_id for c in controllers]

        node2index = {node_id: i for i, node_id in enumerate(target_node_ids)}
        num_nodes = len(target_node_ids)
        target_set = set(target_node_ids)

        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        for src_idx, controller in enumerate(controllers):
            source_id = controller.tls_id

            try:
                source_node = self.net.getNode(source_id)
            except KeyError:
                print(f"[Warning] sumolib cannot find node {source_id}; skipping outgoing-edge search.")
                continue

            search_queue = [(edge, 0) for edge in source_node.getOutgoing()]
            visited_edges = set()

            while search_queue:
                current_edge, depth = search_queue.pop(0)
                if current_edge in visited_edges: continue
                visited_edges.add(current_edge)

                if depth > 5: continue

                next_node = current_edge.getToNode()
                next_node_id = next_node.getID()

                if next_node_id in target_set:
                    if next_node_id != source_id:
                        dst_idx = node2index[next_node_id]
                        adj_matrix[dst_idx][src_idx] = 1

                else:
                    for next_edge in next_node.getOutgoing():
                        search_queue.append((next_edge, depth + 1))

        np.fill_diagonal(adj_matrix, 1.0)

        return adj_matrix, target_node_ids


    def reset(self):
        self.env.reset()
        state_batch, ids = self._collect_state()
        return state_batch, ids

    def step(self, actions: np.ndarray, ids: list, step=None, max_step=1800):
        assert len(actions) == len(ids), "actions and ids length mismatch"
        done = False

        for act, idx in zip(actions, ids):
            self.env.controllers[idx].set_action(int(act) * 2)
            self.env.controllers[idx].yellow_serve_status = 'not'

        for _ in range(self.decision_step):
            for i in self.env.controllers:
                i.if_need()
            traci.simulationStep()
            if step is not None:
                step = step + 1
                if step >= max_step:
                    done = True
                    break

        next_state, next_ids, prev_actions = self._collect_state(actions)
        if not done:
            done = traci.simulation.getMinExpectedNumber() <= 0
        return next_state, done, next_ids, prev_actions

    def _collect_state(self, raw_actions=None):

        state_batch = []
        node_ids = []

        actions = []

        for idx, ctrl in enumerate(self.env.controllers):
            if not getattr(ctrl, "if_need")():
                continue
            else:
                node_ids.append(idx)

        if len(node_ids) == 0:
            return state_batch, node_ids

        if len(node_ids) != NUM_INTERSECTIONS:
            raise ValueError(f"node_num not match num of intersections: {len(node_ids)} vs {NUM_INTERSECTIONS}")

        for idx, controller in enumerate(self.env.controllers):
            tls_id = controller.tls_id
            dir_slots = self.lane_structure.get(tls_id, {})

            feature_vec = []

            for dir_idx in [0, 1, 2, 3]:
                turn_slots = dir_slots.get(dir_idx, {})

                for turn_idx in [0, 1, 2]:
                    target_lanes = turn_slots.get(turn_idx, [])

                    if not target_lanes:
                        feature_vec.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                    else:
                        raw_qs = [traci.lane.getLastStepHaltingNumber(lid) for lid in target_lanes]
                        max_q = sum(raw_qs)
                        norm_q = max_q / 20

                        near_list, mid_list, far_list = self.get_phase_vehicles(target_lanes)
                        n_count = sum(near_list) / 20.0 if near_list else 0.0
                        m_count = sum(mid_list) / 20.0 if mid_list else 0.0
                        f_count = sum(far_list) / 20.0 if far_list else 0.0

                        feature_vec.extend([norm_q, n_count, m_count, f_count, len(raw_qs)])

            state_batch.append(feature_vec)

            if raw_actions is not None:
                current_phase_idx = raw_actions[idx]
                if tls_id in self.phase2vec_mapping:
                    mapping = self.phase2vec_mapping[tls_id]
                    current_signal_vec = mapping.get(current_phase_idx, [0] * 12)
                else:
                    current_signal_vec = [0] * 12

                actions.append(current_signal_vec)

        if raw_actions is not None:
            return np.array(state_batch, dtype=np.float32), node_ids, np.array(actions, dtype=np.float32)
        else:
            return np.array(state_batch, dtype=np.float32), node_ids

    def get_phase_vehicles(self, target_lanes, moving_speed_thresh=0):
        near_counter = []
        mid_counter = []
        far_counter = []

        for lane_id in target_lanes:
            length = traci.lane.getLength(lane_id)
            vids = traci.lane.getLastStepVehicleIDs(lane_id)

            seg_len = length / 3.0 if length > 0 else 1.0

            l_near = 0
            l_mid = 0
            l_far = 0

            for vid in vids:
                spd = traci.vehicle.getSpeed(vid)
                pos = traci.vehicle.getLanePosition(vid)

                if spd > moving_speed_thresh:

                    idx = min(int(pos // seg_len), 2)

                    if idx == 0:
                        l_far += 1
                    elif idx == 1:
                        l_mid += 1
                    else:
                        l_near += 1

            near_counter.append(l_near)
            mid_counter.append(l_mid)
            far_counter.append(l_far)

        return near_counter, mid_counter, far_counter

    def _collect_pressure(self):
        batch_states, need_ids, labels, masks = [], [], [], []
        for idx, ctrl in enumerate(self.env.controllers):
            if getattr(ctrl, "if_need")():
                lane_q = ctrl.get_lane_queue()

                valid_phase = self.valid_phases[ctrl.tls_id]
                phase_pressure_list = [0] * valid_phase
                label_list = [0] * valid_phase

                downstream_q = ctrl.get_downstream_queue()

                phase_lanes = ctrl.get_phase_lanes()

                for in_lane in ctrl.incoming_lanes:
                    q_in = lane_q.get(in_lane, 0)
                    out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                    q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                    for key, value in phase_lanes.items():
                        if in_lane in value and key % 2 == 0:
                            phase_pressure_list[key // 2] = phase_pressure_list[key // 2] + (q_in - q_out)
                            label_list[key // 2] = label_list[key // 2] + q_in

                phase_pressure_obs, label_obs, mask = ctrl.state_process(phase_pressure_list, label_list, max_lane=4)

                batch_states.append(phase_pressure_obs.reshape(-1))
                need_ids.append(idx)
                labels.append(label_obs)
                masks.append(mask)

        if len(need_ids) == 0:
            return batch_states, labels, masks

        if len(need_ids) != NUM_INTERSECTIONS:
            print(f"len(need_ids): {len(need_ids)}")
            raise ValueError(f"node_num not match num of intersections in calculate pressure")

        return np.stack(batch_states).astype(np.float32), labels, masks

def build_dataset():
    traci.start(SUMO_CMD)
    base_env = Env(net_file=NET_FILE)
    env = TrainWrapper(base_env, DECISION_STEP, NET_FILE)

    dataset = []
    print("Start collecting assessor training data...")

    for episode in range(NUM_EPISODES):
        print(f"===== Episode {episode + 1} Start =====")
        agent = MaxPressureAgent(NUM_INTERSECTIONS)
        epsilon = 0.3

        step = 0
        state_batch, ids = env.reset()
        pressure, labels, masks = env._collect_pressure()

        while step < MAX_EP_STEPS:
            if len(ids) == 0:
                traci.simulationStep()
                state_batch, ids = env._collect_state()
                step += 1
                pressure, labels, masks = env._collect_pressure()
                continue

            print(f"step is {step}")
            current_states = state_batch
            actions = agent.act(pressure, env.valid_nums, epsilon)
            next_state_batch, done, next_ids, prev_actions = env.step(actions, ids, step)

            pressure, labels, masks = env._collect_pressure()
            labels = np.array(labels, dtype=np.float32)
            masks = np.array(masks, dtype=np.float32)

            try:
                if not isinstance(current_states, np.ndarray):
                    current_states = np.array(current_states, dtype=np.float32)
                if not isinstance(prev_actions, np.ndarray):
                    prev_actions = np.array(prev_actions, dtype=np.float32)
                if not isinstance(labels, np.ndarray):
                    labels = np.array(labels, dtype=np.float32)

                if current_states.shape != (NUM_INTERSECTIONS, 60):
                    raise ValueError(f"State shape error! Expected ({NUM_INTERSECTIONS}, 60), got {current_states.shape}")
                if prev_actions.shape != (NUM_INTERSECTIONS, 12):
                    raise ValueError(f"Action shape error! Expected ({NUM_INTERSECTIONS}, 12), got {prev_actions.shape}")
                if labels.shape != (NUM_INTERSECTIONS, 4):
                    raise ValueError(f"Label shape error! Expected ({NUM_INTERSECTIONS}, 4), got {labels.shape}")
            except ValueError as e:
                print(f"[Step {step}] Sample validation failed: {e}")
                break

            dataset.append({
                "step": step,
                "state_features": current_states,
                "action_index": prev_actions,
                "prediction_labels": labels,
                "masks": masks,
            })

            state_batch, ids = next_state_batch, next_ids
            step += DECISION_STEP

            if done:
                print("All vehicles have left the network or the max step limit was reached.")
                break

        traci.close()

    adj_matrix, order_check = env.get_adj_matrix()
    adj_matrix_path = str(_resolve_local_path("CRITILIGHT_DC_GRAPH_OUT", "checkpoints/traffic_graph_jn1.pkl"))
    output_path = str(_resolve_local_path("CRITILIGHT_DC_DATASET_OUT", "datasets/traffic_data_jn.pkl"))
    Path(adj_matrix_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    static_data = {
        "adj_matrix": adj_matrix,
        "node_ids": order_check,
    }
    with open(adj_matrix_path, "wb") as handle:
        pickle.dump(static_data, handle)
    print(f"Graph saved to {adj_matrix_path}")

    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle)
    print(f"Dataset saved to {output_path}")


def main():
    build_dataset()


if __name__ == "__main__":
    main()
