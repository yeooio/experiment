"""Three disjoint lanes sharing one GPU; training protocol is unchanged."""
LANES = ('cs2_other', 'oxford', 'cs2_transformer')

def lane_tasks(tasks, lane):
    if lane not in LANES:
        raise ValueError(lane)
    for task in tasks:
        if lane == 'oxford' and task['dataset'] == 'oxford':
            yield task
        elif lane == 'cs2_other' and task['dataset'] == 'cs2' and task['model'] != 'Transformer':
            yield task
        elif lane == 'cs2_transformer' and task['dataset'] == 'cs2' and task['model'] == 'Transformer':
            yield task
