import json
import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

class LLM:
    """Unified decision interface for CritiLight.

    Prompt generation, response parsing and logging stay here.
    Replace only the backend layer if you want to switch between an API model
    and a local fine-tuned LLM.
    """

    def __init__(self, log_dir=None, backend=None):
        log_root = Path(log_dir) if log_dir else PROJECT_ROOT / "outputs" / "critilight"
        log_root.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.normal_log_path = log_root / f"llm_log_{timestamp}.jsonl"
        self.error_log_path = log_root / f"llm_error_{timestamp}.jsonl"
        self.normal_log_path.write_text("", encoding="utf-8")
        self.error_log_path.write_text("", encoding="utf-8")
        self.backend = backend or self._build_backend()

    def _build_backend(self):
        backend_name = os.environ.get("CRITILIGHT_LLM_BACKEND", "remote").lower()
        if backend_name == "remote":
            from critilight.remote_llm_backend import RemoteLLMBackend
            return RemoteLLMBackend()
        if backend_name == "local":
            from critilight.local_llm_backend import LocalLLMBackend
            return LocalLLMBackend()
        raise ValueError(f"Unsupported LLM backend: {backend_name}")

    def action(self, phase_queue_lengths, phase_movements, structure, controller):
        action_text, up_down_proj = self.format_queue_data_for_display(phase_queue_lengths, phase_movements)
        controller.action_text = action_text
        controller.up_down_proj = up_down_proj

        prompt = f'''Role: You are a Traffic Signal Control AI.

Objective: Based on the real-time traffic representation and commonsense knowledge provided, determine the next traffic signal phase to activate. The signal duration is fixed at 30 seconds.

Real-Time Traffic Representation:
- Intersection Topology:
{structure}
- Action Space:
{action_text}

Commonsense Knowledge:
1. THINK STEP BY STEP: Analyze the available information carefully before choosing a phase.
2. MAXIMIZE THROUGHPUT: Select the proper phase to minimize congestion and waiting time.
3. EARLY QUEUE URGENCY: Prioritize vehicles queued near the stop line.
4. DOWNSTREAM BLOCKAGE CAUTION: Do not release traffic into a nearly full downstream link.
5. WAIT TIME FAIRNESS: Avoid starving a movement for too long.
6. LANE RULES: Vehicles pass one at a time per lane and follow queue order.

Task:
Select an appropriate traffic signal phase and explain why.

Output Format:
<response>
    <reason>INSERT_REASON_HERE</reason>
    <signal>INSERT_PHASE_NUMBER_HERE</signal>
</response>
'''
        return prompt, prompt

    def revise_action(self, controllers, index, structure, action_text, pre_actions, phase_dict, traffic_lights, neighbors, junction_state, junctions_occupancy_num):
        current_ctrl = controllers[index]
        prediction_text, neighbors_text, downstream_text = self.format_revise_text(
            current_ctrl,
            controllers,
            pre_actions,
            phase_dict,
            traffic_lights,
            neighbors,
            junction_state,
            junctions_occupancy_num,
        )

        prompt = f'''Role: You are a Traffic Signal Control AI.

Objective: Based on the current state, prediction scores, downstream status and adjacent intersections, choose the next traffic signal phase. The signal duration is fixed at 30 seconds.

Real-Time Traffic Representation:
- Intersection Topology:
{structure}
- Action Space:
{action_text}
- Downstream Status:
{downstream_text}
- Traffic Flow Prediction:
{prediction_text}
- Adjacent Intersections Status:
{neighbors_text}

Commonsense Knowledge:
1. THINK STEP BY STEP: Analyze the available information before choosing a phase.
2. MAXIMIZE THROUGHPUT: Reduce congestion and waiting time.
3. DOWNSTREAM BLOCKAGE CAUTION: Prefer movements whose downstream links can still receive flow.
4. NETWORK COORDINATION: Consider likely inflow from adjacent intersections.
5. BALANCE FORECAST AND ACTUAL DATA: Use prediction as reference, but trust current traffic state first.
6. FALLBACK STRATEGY: If options are similar, prefer the movement with the longest queue.

Output Format:
<response>
    <reason>INSERT_REASON_HERE</reason>
    <signal>INSERT_PHASE_NUMBER_HERE</signal>
</response>
'''

        answer = self.infer(prompt)
        if answer is not None:
            match = re.search(r"<signal>(\d+)</signal>", answer.strip())
            if match:
                chosen_phase = int(match.group(1))
                self._log_normal_interaction(prompt, answer)
                return chosen_phase
        self._log_error_interaction(prompt, answer)
        return None

    def infer(self, prompt):
        return self.backend.infer(prompt)

    def format_revise_text(self, current_ctrl, controllers, pre_actions, phase_dict, traffic_lights, neighbors, junction_state, junctions_occupancy_num):
        prediction_lines = []
        for phase_id, score in sorted(phase_dict.items()):
            prediction_lines.append(f"Phase {phase_id}: predicted queue delta score = {score:.2f}")
        prediction_text = "\n".join(prediction_lines) if prediction_lines else "No prediction scores available."

        neighbor_lines = []
        for neighbor_id in neighbors:
            if neighbor_id not in traffic_lights:
                continue
            neighbor_action = pre_actions.get(neighbor_id)
            if neighbor_action is None:
                continue
            neighbor_state = junction_state.get(neighbor_id, {})
            total_queue = 0
            for phase_info in neighbor_state.values():
                for lane_stats in phase_info.values():
                    total_queue += lane_stats.get("queue_length", 0)
            neighbor_lines.append(f"Neighbor {neighbor_id}: selected phase {neighbor_action}, total incoming queue {total_queue}")
        neighbors_text = "\n".join(neighbor_lines) if neighbor_lines else "No adjacent information."

        downstream = junctions_occupancy_num.get(current_ctrl.tls_id, [])
        downstream_text = "\n".join(
            f"Phase {phase_id}: downstream occupancy = {value:.2f}"
            for phase_id, value in enumerate(downstream)
        ) if downstream else "No downstream occupancy information."
        return prediction_text, neighbors_text, downstream_text

    def format_queue_data_for_display(self, phase_queues, phase_movements):
        phase_lines = []
        up_down_proj = {}
        for phase_id in sorted(phase_queues.keys()):
            lane_entries = []
            up_down_proj[phase_id] = {}
            for lane_id, queue_length in phase_queues[phase_id].items():
                lane_entries.append(f"{lane_id}: queue={queue_length}")

            movement_data = phase_movements.get(phase_id, {})
            if isinstance(movement_data, dict):
                movement_iter = movement_data.items()
            else:
                movement_iter = movement_data

            for movement in movement_iter:
                if not isinstance(movement, (list, tuple)) or len(movement) < 2:
                    continue
                lane_id, out_lane = movement[0], movement[1]
                up_down_proj[phase_id][lane_id] = out_lane
            lanes_text = "; ".join(lane_entries) if lane_entries else "no active lanes"
            phase_lines.append(f"Phase {phase_id}: {lanes_text}")
        return "\n".join(phase_lines), up_down_proj

    def _log(self, path, data):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _log_normal_interaction(self, prompt, response):
        self._log(self.normal_log_path, {"prompt": prompt, "response": response, "timestamp": time.time()})

    def _log_error_interaction(self, prompt, response):
        self._log(self.error_log_path, {"prompt": prompt, "response": response, "timestamp": time.time()})
