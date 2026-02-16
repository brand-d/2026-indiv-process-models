from . import actions

def _state_to_str(state):
    return "{}/{}/{}".format(state["phase"], state["task"], state["state"])

operations = actions.conversions.getProcesses() \
            + actions.nvc_rules.getProcesses() \
            + actions.quantifiers.getProcesses() \
            + actions.directions.getProcesses() \
            + actions.extensions.getProcesses() \
            + actions.conclusion_tests.getProcesses() \
            + actions.preferences.getProcesses()

operations_with_fol = operations + [actions.FOLSubprocess()]


def plan(task, target, use_fol=False, tolerance=0, verbose=True):
    if isinstance(target, str):
        target = [target]
    target = set(target)

    if verbose:
        print("Finding operation sequences:")
        print("    Task:", task)
        print("    Target:", target)

    initial_state = {
        "phase": actions.base.Phase.TASK_INTERPRETATION,
        "task": task,
        "state": task
    }
    
    initial_node = {
        "cost": 0,
        "state": initial_state,
        "path": []
    }

    open_list = [initial_node]
    closed_list = set()

    solutions = []
    optimal_cost = -1

    operations_to_use = operations
    if use_fol:
        operations_to_use = operations_with_fol

    while open_list:
        open_list.sort(key=lambda x: x["cost"])

        current_node = open_list.pop(0)
        closed_list.add(_state_to_str(current_node["state"]))

        cost = current_node["cost"]
        state = current_node["state"]
        path = current_node["path"]

        if optimal_cost > 0 and cost >= optimal_cost + tolerance:
            break

        for operation in operations_to_use:
            if not operation.is_applicable_in_phase(state["phase"]):
                continue
            next_state = operation.apply(state)

            if set(next_state["state"]) == target:
                if optimal_cost < 0:
                    optimal_cost = cost + 1
                solutions.append({
                    "cost": cost + 1,
                    "path": [x for x in path] + [operation]
                })
            elif _state_to_str(next_state) not in closed_list:
                open_list.append({
                    "cost": cost + 1,
                    "state": next_state,
                    "path": [x for x in path] + [operation]
                })
    if verbose:
        print("Found {} solutions".format(len(solutions)))
        if solutions:
            print("    Cost:", solutions[0]["cost"])

    return solutions
