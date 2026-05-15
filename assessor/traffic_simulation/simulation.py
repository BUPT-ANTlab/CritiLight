from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_local_path(value, default):
    raw = os.environ.get(value, default)
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


import traci
import sumolib
import numpy as np
from .utils import *
import torch
from collections import defaultdict
import networkx as nx


class Env:
    def __init__(self, net_file):
        self.net_file = net_file
        self.graph = nx.DiGraph()
        self.traffic_lights = {}
        self.traffic_lights_adjacency = {}
        self.tls_junction_mapping = {}
        self.controllers = []
        self.initialize_traffic_lights()
        self.initialize_map()

        self.depart_times = {}
        self.arrival_times = {}
        self.total_travel_time = 0
        self.num_arrived_vehicles = 0
        self.total_queue_length = 0
        self.total_waiting_time = 0

    def initialize_map(self):

        self.graph.clear()
        edges = traci.edge.getIDList()
        full_graph = nx.DiGraph()
        for edge in edges:
            if not edge.startswith(':'):
                from_node = traci.edge.getFromJunction(edge)
                to_node = traci.edge.getToJunction(edge)
                if from_node and to_node:
                    full_graph.add_edge(from_node, to_node, edge_id=edge)

        signal_nodes = set()
        for tls_id, junctions in self.tls_junction_mapping.items():
            signal_nodes.update(junctions)

        for node in signal_nodes:
            for other_node in signal_nodes:
                if node == other_node:
                    continue
                try:
                    shortest_path = nx.shortest_path(full_graph, source=node, target=other_node)
                    if len(shortest_path) == 2:
                        self.graph.add_edge(node, other_node)
                except nx.NetworkXNoPath:
                    continue

    def initialize_traffic_lights(self):

        self.traffic_lights = []

        traffic_light_ids = traci.trafficlight.getIDList()

        self.controllers = [
            TrafficSignalController(tls_id=tls_id, env_graph=self.graph)
            for tls_id in traffic_light_ids
        ]

        self.traffic_lights = traffic_light_ids

        for tls_id in traffic_light_ids:
            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            controlled_junctions = set()

            for link_group in controlled_links:
                for link in link_group:
                    incoming_lane = link[0]
                    incoming_edge = traci.lane.getEdgeID(incoming_lane)
                    from_junction = traci.edge.getFromJunction(incoming_edge)

                    if from_junction:
                        controlled_junctions.add(from_junction)
            self.traffic_lights_adjacency[tls_id] = list(controlled_junctions)

            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            controlled_junctions = set()
            for lane in controlled_lanes:
                edge = traci.lane.getEdgeID(lane)
                junction = traci.edge.getToJunction(edge)
                controlled_junctions.add(junction)
            self.tls_junction_mapping[tls_id] = controlled_junctions

    def get_controlled_edges_by_tls(self):

        controlled_edges = {}

        for tls_id in self.traffic_lights:
            edges_set = set()

            controlled_links = traci.trafficlight.getControlledLinks(tls_id)
            for link_group in controlled_links:
                for link in link_group:
                    incoming_lane = link[0]
                    incoming_edge = traci.lane.getEdgeID(incoming_lane)
                    edges_set.add(incoming_edge)

            controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
            for lane in controlled_lanes:
                edge = traci.lane.getEdgeID(lane)
                edges_set.add(edge)

            controlled_edges[tls_id] = list(edges_set)

        return controlled_edges

    def get_tls_own_junctions(self, net_file):
        net = sumolib.net.readNet(net_file)

        tls_junction_mapping = {}

        for junction in net.getNodes():
            junction_type = junction.getType()
            if junction_type == 'traffic_light':
                tls_id = junction.getID()
                tls_junction_mapping[tls_id] = junction

        return tls_junction_mapping

    def get_adjacency_matrix(self):
        adjacency = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            if node not in neighbors:
                neighbors.append(node)
            adjacency[node] = neighbors
        return adjacency

    def get_graph(self):
        return self.graph

    def reset(self):
        traci.start([
            'sumo',
            "-c", _resolve_local_path("CRITILIGHT_SUMO_CFG", "../map/hangzhou/hangzhou.sumocfg"),
            '--ignore-route-errors',
            '--tripinfo-output',
            _resolve_local_path("CRITILIGHT_TRIPINFO_OUTPUT", "../outputs/critilight/tripinfo.xml"),
            '--tripinfo-output.write-unfinished',
            '--fcd-output', _resolve_local_path("CRITILIGHT_FCD_OUTPUT", "../outputs/critilight/fcd_output.xml"),
            '--queue-output', _resolve_local_path("CRITILIGHT_QUEUE_OUTPUT", "../outputs/critilight/queue_output.xml"),
            '--time-to-teleport', '-1',
            '--collision.action', 'none',
        ])

        self.traffic_lights = {}
        self.traffic_lights_adjacency = {}
        self.tls_junction_mapping = {}
        self.controllers = []
        self.initialize_traffic_lights()
        self.initialize_map()
        self.depart_times = {}
        self.arrival_times = {}
        self.total_travel_time = 0
        self.num_arrived_vehicles = 0
        self.total_queue_length = 0
        self.total_waiting_time = 0

    def calculate_metrics(self, step):
        for veh_id in traci.simulation.getDepartedIDList():
            self.depart_times[veh_id] = traci.simulation.getTime()

        for veh_id in traci.simulation.getArrivedIDList():
            self.arrival_times[veh_id] = traci.simulation.getTime()
            travel_time = self.arrival_times[veh_id] - self.depart_times[veh_id]
            self.total_travel_time += travel_time
            self.num_arrived_vehicles += 1

        for veh_id in traci.vehicle.getIDList():
            self.total_waiting_time += traci.vehicle.getWaitingTime(veh_id)

        self.total_queue_length += sum(
            traci.edge.getLastStepHaltingNumber(edge_id) for edge_id in traci.edge.getIDList())

        average_travel_time = self.total_travel_time / self.num_arrived_vehicles if self.num_arrived_vehicles > 0 else 0
        average_queue_length = self.total_queue_length / step if step > 0 else 0
        average_waiting_time = self.total_waiting_time / step if step > 0 else 0

        return average_travel_time, average_queue_length, average_waiting_time

    def get_batch_state(self):

        need_action_controllers = []
        processed_features = []
        for orig_idx, controller in enumerate(self.controllers):
            if controller.if_need():
                need_action_controllers.append(orig_idx)

            length_feature_per_phase = controller.return_features()

            if length_feature_per_phase is None:
                continue

            logic_phases = controller.phases

            valid_phase_num = min(len(logic_phases), 12)
            phase_features = np.array(length_feature_per_phase[:valid_phase_num])

            padded = np.pad(
                phase_features,
                (0, 12 - valid_phase_num),
                'constant',
                constant_values=(-1.0 if valid_phase_num < 12 else 0)
            )

            processed_features.append(padded)

        if len(processed_features) > 0:
            state_tensor = torch.FloatTensor(np.array(processed_features)[:, :, None])
        else:
            state_tensor = torch.zeros(0, 12, 1)

        return state_tensor, need_action_controllers


class TrafficSignalController:
    def __init__(self, tls_id, env_graph):
        self.tls_id = tls_id
        self.links = traci.trafficlight.getControlledLinks(tls_id)
        self.controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)
        self.controlled_edges = self._get_incoming_edges()
        self.phases = traci.trafficlight.getAllProgramLogics(tls_id)[0].phases
        self.remaining_duration = 30

        self.phase_lane_mapping = self._build_phase_mapping()

        self.incoming_lanes = []
        self.get_income_lane()

        self.lane_in_out_map = {}
        self.build_in_out_lane_map()

        self.last_action = None
        self.last_state = None
        self.accumulated_reward = 0
        self.last_test_reward = 0
        self.current_phase = 0
        self.yellow_serve_status = 'not'

        self.previous_prompt = None
        self.previous_answer = None

        self.revise_prompt = None
        self.structure = None
        self.action_text = None
        self.up_down_proj = None

        self.structure_label = None
        self._init_structure_label()

        self.prompt_for_reviewer_log = ""
        self.prompt_for_reviewer_log_last_step = ""

    def _build_phase_mapping(self):
        mapping = {}
        for p_idx, phase in enumerate(self.phases):
            state = phase.state
            green_lanes = set()

            for i, char in enumerate(state):
                if char.lower() == 'g':
                    if i < len(self.links):
                        connections = self.links[i]
                        for conn in connections:
                            green_lanes.add(conn[0])

            mapping[p_idx] = list(green_lanes)
        return mapping

    def get_income_lane(self):
        self.incoming_lanes = []

        for lane in self.controlled_lanes:
            links = traci.lane.getLinks(lane)
            if links:
                self.incoming_lanes.append(lane)

        self.incoming_lanes = sorted(set(self.incoming_lanes))

    def build_in_out_lane_map(self):
        lane_in_out_map = {}

        for lane in self.incoming_lanes:
            links = traci.lane.getLinks(lane)
            if links:
                main_out_lane = links[0][0]
                lane_in_out_map[lane] = main_out_lane
            else:
                lane_in_out_map[lane] = None

        self.lane_in_out_map = lane_in_out_map
        return lane_in_out_map

    def get_downstream_queue(self):
        downstream_q = {}
        for in_lane, out_lane in self.lane_in_out_map.items():
            if out_lane:
                vehicles = traci.lane.getLastStepVehicleIDs(out_lane)
                lane_length = traci.lane.getLength(out_lane)
                threshold = max(lane_length - 100, 0)
                queue_count = 0
                for veh in vehicles:
                    pos = traci.vehicle.getLanePosition(veh)
                    speed = traci.vehicle.getSpeed(veh)
                    if pos >= threshold and speed <= 0:
                        queue_count += 1
                downstream_q[out_lane] = queue_count
            else:
                downstream_q[out_lane] = 0
        return downstream_q

    def get_lane_queue(self, distance=100):
        lane_queue_lengths = {}

        for lane in self.incoming_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            lane_length = traci.lane.getLength(lane)
            threshold = max(lane_length - distance, 0)

            queue_count = 0
            for veh in vehicles:
                pos = traci.vehicle.getLanePosition(veh)
                speed = traci.vehicle.getSpeed(veh)
                if pos >= threshold and speed <= 0:
                    queue_count += 1

            lane_queue_lengths[lane] = queue_count

        return lane_queue_lengths

    def get_phase_lanes(self):
        phase_lanes = {}
        for phase_index, phase in enumerate(self.phases):
            controlled_lanes = self.get_phase_controlled_lanes(phase_index)
            phase_lanes[phase_index] = list(controlled_lanes)

        return phase_lanes

    def get_pressure(self, valid_phases):
        lane_q = self.get_lane_queue()
        valid_phase = valid_phases[self.tls_id] // 2

        phase_pressure_list = [0] * valid_phase
        phase_queue_in_list = [0] * valid_phase
        phase_queue_out_list = [0] * valid_phase

        phase_in_lane_num = [0] * valid_phase
        phase_out_lane_num = [0] * valid_phase

        downstream_occupancy = [0] * valid_phase

        downstream_q = self.get_downstream_queue()

        phase_lanes = self.get_phase_lanes()

        for in_lane in self.incoming_lanes:
            q_in = lane_q.get(in_lane, 0)
            out_lane = self.lane_in_out_map.get(in_lane, None)
            try:
                current_occ = traci.lane.getLastStepOccupancy(out_lane)
            except traci.TraCIException:
                current_occ = 0.0

            q_out = downstream_q.get(out_lane, 0) if out_lane else 0

            for key, value in phase_lanes.items():
                if in_lane in value and key % 2 == 0:
                    idx = key // 2
                    phase_pressure_list[idx] = phase_pressure_list[idx] + (q_in - q_out)
                    phase_queue_in_list[idx] = phase_queue_in_list[idx] + q_in
                    phase_queue_out_list[idx] = phase_queue_out_list[idx] + q_out

                    phase_in_lane_num[idx] = phase_in_lane_num[idx] + 1
                    phase_out_lane_num[idx] = phase_out_lane_num[idx] + 1

                    if out_lane:
                        downstream_occupancy[idx] = max(downstream_occupancy[idx], current_occ)

        phase_pressure_obs = self.state_process(phase_pressure_list)

        return phase_pressure_obs, phase_queue_in_list, phase_queue_out_list, phase_in_lane_num, phase_out_lane_num, downstream_occupancy

    def state_process(self, lane_q, max_lane=4):
        if type(lane_q) is dict:
            queue_values = list(lane_q.values())
        elif type(lane_q) is list:
            queue_values = lane_q
        else:
            raise NotImplementedError

        processed = queue_values + [-100] * (max_lane - len(queue_values))

        return np.array(processed, dtype=np.float32)

    def _get_incoming_edges(self) -> set[str]:
        edges = set()
        for lane in self.controlled_lanes:
            try:
                eid = traci.lane.getEdgeID(lane)
                edges.add(eid)
            except traci.TraCIException:
                continue
        return edges

    def _init_structure_label(self):
        if self.tls_id == '71':
            self.structure_label = f"8phase, incoming_lanes 2*2*2*2, 71"
        elif self.tls_id in {'100', '101', '103', '104', '106', '108', '111', '114', '119', '122', '123'}:
            self.structure_label = f"6phase, incoming_lanes 2*2*2, Y"
        else:
            links = traci.trafficlight.getControlledLinks(self.tls_id)
            edge_stats = defaultdict(lambda: {'incoming_lanes': set(), 'outgoing_edges': set()})
            for link_group in links:
                for link in link_group:
                    in_edge = traci.lane.getEdgeID(link[0])
                    edge_stats[in_edge]['incoming_lanes'].add(link[0])
                    edge_stats[in_edge]['outgoing_edges'].add(traci.lane.getEdgeID(link[1]))
            for e in edge_stats:
                edge_stats[e]['incoming_lane_count'] = len(edge_stats[e]['incoming_lanes'])

            phase_count = len(self.phases)
            entries = sorted([v['incoming_lane_count'] for v in edge_stats.values()], reverse=True)
            code = "×".join(map(str, entries))
            self.structure_label = f"{phase_count}phase, incoming_lanes{code}"

    def format_structure_prompt(self) -> str:

        edge2lanes: dict[str, set[str]] = defaultdict(set)
        all_links = traci.trafficlight.getControlledLinks(self.tls_id)
        for per_phase_links in all_links:
            for in_lane_id, _, _ in per_phase_links:
                edge = traci.lane.getEdgeID(in_lane_id)
                edge2lanes[edge].add(in_lane_id)

        sorted_edges = sorted(edge2lanes.keys(),
                              key=lambda e: -len(edge2lanes[e]))

        road_ids = sorted_edges
        lane_counts = [len(edge2lanes[e]) for e in sorted_edges]

        hint_phase = len(self.phases) // 2

        states = getattr(self.phases[0], 'state', None)
        if states is not None:
            green_only = [
                ph for ph in self.phases
                if any(c == 'G' for c in ph.state) and 'y' not in ph.state
            ]
            if len(green_only) > 0:
                hint_phase = len(green_only)

        num_roads = len(road_ids)

        roads_str = ", ".join(road_ids)
        lanes_str = ", ".join(str(c) for c in lane_counts)

        prompt = (
            f"There are {num_roads} bidirectional roads connected to this intersection "
            f"(ID: {roads_str}), with {lanes_str} incoming lanes respectively. "
            f"The traffic light in this intersection operates with {hint_phase} signal phases."
        )
        return prompt

    def get_signal_state(self):

        current_phase = self.get_current_phase()
        queue_lengths = self.get_phase_queue_lengths()
        duration = self.remaining_duration
        elapsed_time = traci.simulation.getTime() - (traci.trafficlight.getNextSwitch(self.tls_id) - duration)

        return {
            "Traffic Signal ID": self.tls_id,
            "Current Phase": current_phase,
            "Remaining Duration": duration,
            "Elapsed Time": elapsed_time,
            "Queue Lengths": queue_lengths
        }

    def get_current_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def get_phase_controlled_lanes(self, phase_index):

        phase = self.phases[phase_index]
        controlled_lanes = []

        for i, lane in enumerate(self.controlled_lanes):
            if phase.state[i] in ["G", "g"]:
                controlled_lanes.append(lane)

        return controlled_lanes

    def get_phase_queue_lengths(self, distance=100):
        phase_queue_lengths = {}
        for phase_index, phase in enumerate(self.phases):
            controlled_lanes = self.get_phase_controlled_lanes(phase_index)
            phase_queue_lengths[phase_index] = {}

            for lane in controlled_lanes:
                vehicles_in_lane = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = int(traci.lane.getLength(lane))
                if lane_length > distance:
                    threshold = lane_length - distance
                else:
                    threshold = 0
                queue_count = 0
                for vehicle_id in vehicles_in_lane:
                    vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
                    vehicle_speed = traci.vehicle.getSpeed(vehicle_id)
                    if vehicle_position >= threshold and vehicle_speed == 0:
                        queue_count += 1
                phase_queue_lengths[phase_index][lane] = queue_count

        return phase_queue_lengths

    def get_total_queue_length(self, distance=100.0):

        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.getPhases()
        links = traci.trafficlight.getControlledLinks(self.tls_id)

        total_queue = 0

        for ph_idx, ph in enumerate(phases):
            if ph_idx % 2 != 0:
                continue

            state = ph.state
            for sig_idx, char in enumerate(state):
                if sig_idx >= len(links):
                    continue
                if char in ('G', 'g'):
                    for (in_lane, out_lane, via) in links[sig_idx]:
                        if not in_lane or not out_lane:
                            continue

                        vids = traci.lane.getLastStepVehicleIDs(in_lane)
                        lane_length = traci.lane.getLength(in_lane)
                        thresh = max(0.0, lane_length - distance)

                        for vid in vids:
                            pos = traci.vehicle.getLanePosition(vid)
                            spd = traci.vehicle.getSpeed(vid)
                            if spd <= 0.1 and pos >= thresh:
                                total_queue += 1

        return total_queue

    def get_phase_queue_and_vehicles(self, distance=100.0, moving_speed_thresh=1.0):
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.getPhases()
        links = traci.trafficlight.getControlledLinks(self.tls_id)

        phase_queues = {}
        phase_movements = {}

        for ph_idx, ph in enumerate(phases):
            if ph_idx % 2 != 0:
                continue

            moves = []
            state = ph.state
            for sig_idx, char in enumerate(state):
                if sig_idx >= len(links):
                    continue
                if char in ('G', 'g'):
                    for (in_lane, out_lane, via) in links[sig_idx]:
                        if in_lane and out_lane:
                            moves.append((in_lane, out_lane))
            phase_movements[ph_idx] = moves

            lane_stats = {}
            for in_lane, _ in moves:
                vids = traci.lane.getLastStepVehicleIDs(in_lane)
                length = traci.lane.getLength(in_lane)
                seg = length / 3.0 if length > 0 else 1.0
                thresh = max(0.0, length - distance)

                q = 0
                far = mid = near = 0
                for vid in vids:
                    pos = traci.vehicle.getLanePosition(vid)
                    spd = traci.vehicle.getSpeed(vid)
                    if spd <= 0.1 and pos >= thresh:
                        q += 1
                    elif spd > moving_speed_thresh:
                        idx = min(int(pos // seg), 2)
                        if idx == 0:
                            far += 1
                        elif idx == 1:
                            mid += 1
                        else:
                            near += 1

                lane_stats[in_lane] = {
                    'queue_length': q,
                    'moving_far': far,
                    'moving_mid': mid,
                    'moving_near': near,
                }

            phase_queues[ph_idx] = lane_stats

        return phase_queues, phase_movements

    def get_phase_and_vehicles(self, distance=100):

        phase_data = {}
        for phase_index, phase in enumerate(self.phases):
            controlled_lanes = self.get_phase_controlled_lanes(phase_index)
            phase_data[phase_index] = {}

            for lane in controlled_lanes:
                vehicles_on_lane = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = traci.lane.getLength(lane)
                start_position_threshold = max(0, lane_length - distance)

                vehicles_in_zone_ids = []
                for vehicle_id in vehicles_on_lane:
                    vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
                    if vehicle_position >= start_position_threshold:
                        vehicles_in_zone_ids.append(vehicle_id)
                phase_data[phase_index][lane] = {
                    'queue_length': len(vehicles_in_zone_ids),
                    'vehicle_ids': vehicles_in_zone_ids
                }
        return phase_data

    def return_test_reward(self):
        phase_queue_lengths = self.get_phase_queue_lengths()
        total = 0
        for phase_dict in phase_queue_lengths.values():
            total += sum(phase_dict.values())
        return total

    def return_features(self):
        current_time = traci.simulation.getTime()

        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration

        elapsed_time = current_time - self.last_phase_change_time

        if elapsed_time >= self.current_phase_duration:

            phase_queue_lengths = self.get_phase_queue_lengths(distance=100)

            if phase_queue_lengths:
                max_queues_per_phase = [
                    max(lane_counts.values(), default=0)
                    for lane_counts in phase_queue_lengths.values()
                ]
            return max_queues_per_phase
        else:
            return None

    def set_action(self, p_action, d_action, yellow=None):
        new_phase = p_action * 2
        if d_action:
            new_duration = d_action
        else:
            new_duration = 30

        current_time = traci.simulation.getTime()
        if yellow:
            new_phase = p_action + 1
            new_duration = 6
            self.yellow_serve_status = 'served'

        self.set_phase(new_phase, duration=new_duration)
        self.remaining_duration = new_duration
        self.last_phase_change_time = current_time
        self.current_phase_duration = new_duration
        self.current_phase = p_action

    def report_traffic_light_info(self, tls_id, current_phase, duration, current_step):
        print(
            f"Current Step: {current_step}, Last Phase: {current_phase}, Last Duration: {duration}, Traffic Light ID: {tls_id} is changing for new phase")

    def if_need(self):
        current_time = traci.simulation.getTime()
        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration
        elapsed_time = current_time - self.last_phase_change_time

        if elapsed_time >= self.current_phase_duration and self.yellow_serve_status == 'served':

            return True
        elif elapsed_time >= self.current_phase_duration and self.yellow_serve_status == 'not':

            self.set_action(self.current_phase, -1, True)
            return False

    def control_signal_logic(self, current_step):
        current_time = traci.simulation.getTime()

        if not hasattr(self, 'last_phase_change_time'):
            self.last_phase_change_time = current_time
            self.current_phase_duration = self.remaining_duration

        elapsed_time = current_time - self.last_phase_change_time

        if elapsed_time >= self.current_phase_duration:
            current_phase = self.get_current_phase()
            duration = self.remaining_duration
            self.report_traffic_light_info(self.tls_id, current_phase, duration, current_step)

            phase_queue_lengths = self.get_phase_queue_lengths(distance=100)

            if phase_queue_lengths:
                max_queue_length = max(
                    (max(lane_counts.values()) for lane_counts in phase_queue_lengths.values() if lane_counts),
                    default=0
                )
            else:
                max_queue_length = 0

            new_duration = 30

            new_phase = select_phase_with_max_queue(phase_queue_lengths)

            self.set_phase(new_phase, duration=new_duration)
            self.remaining_duration = new_duration

            self.last_phase_change_time = current_time
            self.current_phase_duration = new_duration

    def set_phase(self, phase_index, duration):
        traci.trafficlight.setPhase(self.tls_id, phase_index)
        traci.trafficlight.setPhaseDuration(self.tls_id, duration)
        self.remaining_duration = duration


