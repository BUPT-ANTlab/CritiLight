import numpy as np
import traci

class RL_EnvWrapper:
    def __init__(self, base_env, feature_dim: int = 24 + 8 + 4, decision_step: int = 36, top_k: int = 2,):
        self.env = base_env
        self.top_k = top_k
        self.feature_dim = feature_dim
        self.decision_step = decision_step

        self.num_intersections = len(self.env.controllers)
        self._init_adjacency()

        self._prev_queues = np.zeros(self.num_intersections, dtype=np.float32)
        self.valid_phases = {tls_id: self._get_valid_phases(tls_id) // 2 for tls_id in self.env.traffic_lights}
        # print(f"self.valid_phases:{self.valid_phases}")
        self.valid_nums = [self.valid_phases[tls_id] for tls_id in self.env.traffic_lights]
        # print(f"traffic_lights: {self.env.traffic_lights}")
        # print(f"self.valid_nums:{self.valid_nums}")

    def _init_adjacency(self):
        rng = np.random.default_rng(42)
        self.adj = {}
        for i in range(self.num_intersections):
            others = [j for j in range(self.num_intersections) if j != i]
            if len(others) >= self.top_k:
                self.adj[i] = rng.choice(others, size=self.top_k, replace=False).tolist()
            else:
                self.adj[i] = others

    def _get_features(self, ctrl, method):
        lane_q = ctrl.get_lane_queue()
        lane_obs = ctrl.state_process(lane_q)

        try:
            phase_raw = ctrl.get_current_phase()
            if phase_raw % 2 == 0:
                phase_no_yellow = phase_raw
            else:
                phase_no_yellow = phase_raw - 1
        except Exception:
            print(f"phase_no_yellow")
            phase_no_yellow = 0

        phase_obs = np.zeros(8, dtype=np.float32)
        phase_obs[int(phase_no_yellow)] = 1

        intersection_obs = self.get_intersection_obs(ctrl.tls_id)

        if method in ["colight", "efficient_colight", "advanced_colight"]:
            return np.concatenate([lane_obs, phase_obs, intersection_obs])
        elif method in ["attendlight"]:
            phase_list = [[] for _ in range(4)]

            phase_lane_dicts = ctrl.get_phase_queue_lengths()
            # print(f"phase_lane_dicts: {phase_lane_dicts}, {ctrl.tls_id}")

            for key, value1 in phase_lane_dicts.items():
                if key % 2 == 0:
                    for value2 in value1.values():
                        phase_list[int(key // 2)].append(value2)

            phase_lane_obs = ctrl.state_process(phase_list, max_lane=8)

            # print(f"phase_lane_obs:{phase_lane_obs}")
            return np.array(phase_lane_obs)
        elif method in ["mplight"]:
            pressure_list = []
            downstream_q = ctrl.get_downstream_queue()

            for in_lane in ctrl.incoming_lanes:
                q_in = lane_q.get(in_lane, 0)
                out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                pressure_list.append(q_in - q_out)

            pressure_obs = ctrl.state_process(pressure_list)
            pressure_obs = np.array(pressure_obs, dtype=np.float32)
            return np.concatenate([pressure_obs, phase_obs, intersection_obs])
        elif method in ["presslight"]:
            incoming_pressure_list = []
            outgoing_pressure_list = []

            downstream_q = ctrl.get_downstream_queue()

            for in_lane in ctrl.incoming_lanes:
                q_in = lane_q.get(in_lane, 0)
                incoming_capacity = ctrl.incoming_capacity.get(in_lane, 0)
                q_in_cap = q_in / incoming_capacity
                incoming_pressure_list.append(q_in_cap)

                out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                outgoing_capacity = ctrl.outgoing_capacity.get(out_lane, 0)
                q_out_cap = q_out / outgoing_capacity
                outgoing_pressure_list.append(q_out_cap)

            incoming_pressure_obs = ctrl.state_process(incoming_pressure_list)
            outgoing_pressure_obs = ctrl.state_process(outgoing_pressure_list)

            incoming_pressure_obs = np.array(incoming_pressure_obs, dtype=np.float32)
            outgoing_pressure_obs = np.array(outgoing_pressure_obs, dtype=np.float32)

            return np.concatenate([incoming_pressure_obs, outgoing_pressure_obs, phase_obs, intersection_obs])
        elif method in ["fixedtime", "maxpressure"]:
            valid_phase = self.valid_phases[ctrl.tls_id]
            phase_pressure_list = [0] * valid_phase

            downstream_q = ctrl.get_downstream_queue()

            phase_lanes = ctrl.get_phase_lanes()

            for in_lane in ctrl.incoming_lanes:
                q_in = lane_q.get(in_lane, 0)
                out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                for key, value in phase_lanes.items():
                    if in_lane in value and key % 2 == 0:
                        phase_pressure_list[key // 2] = phase_pressure_list[key // 2] + (q_in - q_out)

            phase_pressure_obs = ctrl.state_process(phase_pressure_list, max_lane=4)

            return phase_pressure_obs
        else:
            raise ValueError(f"Unknown method: {method}")

    def build_phase_lane_feats(self, lane_q, phase_lanes):
        phase_lane_feats = []

        for phase_id in sorted(phase_lanes.keys()):
            lanes = phase_lanes[phase_id]

            seen = set()
            unique_lanes = []
            for l in lanes:
                if l not in seen:
                    seen.add(l)
                    unique_lanes.append(l)

            feats = [lane_q[l] for l in unique_lanes if l in lane_q]
            phase_lane_feats.append(feats)

        return phase_lane_feats

    def get_intersection_obs(self, tls_id):
        intersection_phase_dict = {
            "0001": ['1', '109', '116', '118', '120', '13', '14', '142', '15', '161', '166', '17', '172', '176', '177',
                    '18', '20', '25', '26', '51', '7', '72', '73', '8', '84', '97'],
            "0010": ['10', '105', '11', '110', '112', '113', '117', '121', '124', '126', '162', '163', '165', '173',
                     '174', '19', '2', '21', '22', '23', '24', '28', '29', '3', '30', '4', '5', '52', '53', '54', '56',
                     '57', '58', '74', '75', '76', '78', '79', '80', '82', '83', '9', '98', '99'],
            "0011": ['100', '101', '103', '104', '106', '108', '111', '114', '119', '122', '123'],
            "0100": ['102', '115', '12', '125', '27', '55', '77', '85', '86', '87', '88', '89', '91', '92', '93'],
            "0101": ['107', '128', '129', '130', '135', '147', '155', '159', '164', '168', '169', '61', '62', '63',
                     '65', '66', '67', '81'],
            "0110": ['127', '131', '133', '134', '136', '137', '139', '140', '143', '144', '146', '149', '151', '152',
                    '154', '156', '157', '16', '171', '175', '31', '42', '43', '44', '45', '47', '48', '49', '59', '69',
                    '70'],
            "0111": ['132', '145', '46', '64'],
            "1000": ['138', '148', '158', '170', '68'],
            "1001": ['141', '150', '160', '167', '60'],
            "1010": ['153', '41'],
            "1011": ['32', '33', '34', '35', '36', '38', '39', '40', '6'],
            "1100": ['37', '90'],
            "1101": ['50'],
            "1110": ['71'],
            "1111": ['94', '95', '96']
        }

        for key, value in intersection_phase_dict.items():
            if tls_id in value:
                return np.array([int(c) for c in key], dtype=np.float32)

        key = "0000"
        print(f"encoder tls_id {tls_id} error")
        return np.array([int(c) for c in key], dtype=np.float32)


    def reset(self, method):
        self.env.reset()
        self._prev_queues.fill(0)
        return self._collect_state(method)

    def _collect_state(self, method):
        batch_states, need_ids = [], []
        for idx, ctrl in enumerate(self.env.controllers):
            if getattr(ctrl, "if_need")():
                own_feat = self._get_features(ctrl, method)

                batch_states.append(own_feat.reshape(-1))
                need_ids.append(idx)

        if batch_states:
            return np.stack(batch_states).astype(np.float32), need_ids
        else:
            return np.zeros((0, self.top_k + 1, self.feature_dim), dtype=np.float32), []

    def step(self, actions: np.ndarray, ids: list, method=None, step=None, max_step=1800):
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

        rewards = self.calculate_reward(ids, method)

        next_state, next_ids = self._collect_state(method)
        if not done:
            done = traci.simulation.getMinExpectedNumber() <= 0
        return next_state, np.array(rewards, dtype=np.float32), done, next_ids

    def calculate_reward(self, ids, method):
        if method in ["colight", "efficient_colight", "advanced_colight", "attendlight"]:
            rewards = []
            for idx in ids:
                queue_now = self.env.controllers[idx].return_queue_length()
                delta = (self._prev_queues[idx] - queue_now) / 177
                rewards.append(delta)
                self._prev_queues[idx] = queue_now
            return rewards
        elif method in ["mplight"]:
            rewards = []
            for ctrl in self.env.controllers:
                lane_q = ctrl.get_lane_queue()
                downstream_q = ctrl.get_downstream_queue()
                pressure_list = []

                for in_lane in ctrl.incoming_lanes:
                    q_in = lane_q.get(in_lane, 0)
                    out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                    q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                    pressure_list.append(q_in - q_out)

                reward = -np.sum(pressure_list) / 177
                rewards.append(reward)
            return rewards
        elif method in ["presslight"]:
            rewards = []
            for ctrl in self.env.controllers:
                lane_q = ctrl.get_lane_queue()
                downstream_q = ctrl.get_downstream_queue()

                pressure_list = []

                for in_lane in ctrl.incoming_lanes:
                    q_in = lane_q.get(in_lane, 0)
                    incoming_capacity = ctrl.incoming_capacity.get(in_lane, 0)
                    q_in_cap = q_in / incoming_capacity

                    out_lane = ctrl.lane_in_out_map.get(in_lane, None)
                    q_out = downstream_q.get(out_lane, 0) if out_lane else 0
                    outgoing_capacity = ctrl.outgoing_capacity.get(out_lane, 0)
                    q_out_cap = q_out / outgoing_capacity

                    pressure_list.append(q_in_cap - q_out_cap)

                reward = -np.sum(pressure_list) / 177
                rewards.append(reward)
            return rewards
        elif method in ["fixedtime", "maxpressure"]:
            rewards = []
            return rewards
        else:
            print("calculate reward error")
            rewards = []
            return rewards


    def _get_valid_phases(self, tls_id):
            try:
                return len(traci.trafficlight.getAllProgramLogics(tls_id)[0].phases)
            except Exception:
                return 0
