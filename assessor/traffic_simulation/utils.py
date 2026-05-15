import traci


def select_phase_with_max_queue(phase_queue_lengths):
    phase_max_queues = {
        phase: max(queues.values(), default=0)
        for phase, queues in phase_queue_lengths.items()
    }
    max_queue = max(phase_max_queues.values())
    candidate_phases = [phase for phase, value in phase_max_queues.items() if value == max_queue]
    return min(candidate_phases)
