import gurobipy as gp
from gurobipy import GRB
import time

def test_limit():
    n = 2000
    m = 1000

    model = gp.Model("HardTimeLimitTest")

    x = model.addVars(n, vtype=GRB.BINARY, name="x")

    import numpy as np
    obj_coeffs = np.random.rand(n)
    model.setObjective(sum(obj_coeffs[i] * x[i] for i in range(n)), GRB.MAXIMIZE)

    for j in range(m):
        weights = np.random.rand(n)
        model.addConstr(sum(weights[i] * x[i] for i in range(n)) <= n * 0.25)

    limit_seconds = 10
    print(f"\n[test start] set TimeLimit = {limit_seconds} s...")
    model.setParam('TimeLimit', limit_seconds)
    model.setParam('MIPGap', 0.0001)

    start_time = time.time()
    model.optimize()
    end_time = time.time()

    actual_duration = end_time - start_time
    print("\n" + "="*50)
    print(f"actual run time: {actual_duration:.2f} s")
    print(f"Gurobi status code: {model.Status} (9 = TIME_LIMIT)")

    if abs(actual_duration - limit_seconds) < 2:
        print(">>> Conclusion: TimeLimit is effective!")
    else:
        print(">>> Conclusion: TimeLimit may not be effective, or the problem was solved too quickly.")
    print("="*50 + "\n")

if __name__ == "__main__":
    test_limit()
