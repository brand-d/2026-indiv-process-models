import ProcessModel
import sys

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('Usage: python subprocess_planner.py syllog "Concl1, Concl2,..." [0 | 1, to disable/enable FOL]')
        exit()
    syllog = sys.argv[1]
    targets = sys.argv[2]
    use_fol = False
    if len(sys.argv) == 4:
        use_fol = int(sys.argv[3]) == 1

    targets = [x.replace(" ", "") for x in targets.split(",")]
    solutions = ProcessModel.plan(syllog, targets, use_fol=use_fol, tolerance=0)

    for idx, solution in enumerate(solutions):
        solution_names = [x.get_name() for x in solution["path"]]
        print(idx + 1, ":", " -> ".join(solution_names))