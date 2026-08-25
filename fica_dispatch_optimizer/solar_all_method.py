import numpy as np
import pandapower as pp
import pandapower.networks as ppnw
from pandapower.pypower.makePTDF import makePTDF
from pandapower.pd2ppc import _pd2ppc
import os
import gurobipy as gp
from gurobipy import GRB
from solar_scenario_gen import Solar_sce_gen
from scipy.linalg import norm
import time
from datetime import datetime
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
plt.style.use('default')
plt.rcParams.update({
    'font.size': 13,
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'legend.fontsize': 13,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    "mathtext.fontset": "cm",
})

def select_envelope_samples(total_RE_error_train, t_g_list, m0):
    N_WDR, T = total_RE_error_train.shape
    key_sets = {}
    for (t, g) in t_g_list:
        a = total_RE_error_train[:, t]
        b = total_RE_error_train[:, t-1]


        s1 = np.abs(a - b)
        s2 = np.abs(a)
        s3 = np.abs(b)

        scores = np.maximum.reduce([s1, s2, s3])

        m = min(m0, N_WDR)
        if m < N_WDR:
            idx_top = np.argpartition(-scores, m-1)[:m]
            idx_top = idx_top[np.argsort(-scores[idx_top])]
        else:
            idx_top = np.arange(N_WDR, dtype=int)
        key_sets[(t, g)] = np.asarray(idx_top, dtype=int)
    return key_sets
def check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test,
              P_line_limit, gen_bus_list, WT_bus_list, Solar_pred=None, Solar_error_scenarios_test=None, Solar_bus_list=None,
              gen_ramp_rate=None):

    PTDF[np.abs(PTDF) < 1e-5] = 0
    PTDF_gen = PTDF[:, gen_bus_list].T
    PTDF_wind = PTDF[:, WT_bus_list].T
    PTDF_solar = PTDF[:, Solar_bus_list].T if Solar_bus_list is not None else None
    PTDF_load = PTDF.T

    P_res = []
    total_RE_error = WT_error_scenarios_test.sum(axis=-1)
    if Solar_error_scenarios_test is not None:
        total_RE_error = total_RE_error + Solar_error_scenarios_test.sum(axis=-1)

    for t in range(T):
        for g in range(num_gen):
            gen_power_adjusted = gen_power_all[t, g] - total_RE_error[:, t] * gen_alpha_all[t, g]
            P_res.append(gen_power_adjusted <= gen_cap_individual[g])
            P_res.append(gen_power_adjusted >= gen_pmin_individual[g])
    L_res = []
    for t in range(T):
        for l in range(num_branch):
            total_error_expanded = total_RE_error[:, t]

            line_flow_gen = (gen_power_all[t] @ PTDF_gen[:, l]
                            - gen_alpha_all[t] @ PTDF_gen[:, l] * total_error_expanded)

            line_flow_wind = (WT_pred[t] + WT_error_scenarios_test[:, t]) @ PTDF_wind[:, l]
            line_flow = line_flow_gen + line_flow_wind

            if Solar_pred is not None and PTDF_solar is not None:
                line_flow_solar = (Solar_pred[t] + Solar_error_scenarios_test[:, t]) @ PTDF_solar[:, l]
                line_flow = line_flow + line_flow_solar

            line_flow_load = load_bus_all[t] @ PTDF_load[:, l]
            line_flow = line_flow - line_flow_load

            L_res.append(line_flow <= P_line_limit[l])
            L_res.append(line_flow >= -P_line_limit[l])

    R_res = []
    if gen_ramp_rate is not None:
        for t in range(1, T):
            for g in range(num_gen):
                actual_power_t = gen_power_all[t, g] - gen_alpha_all[t, g] * total_RE_error[:, t]
                actual_power_t_1 = gen_power_all[t-1, g] - gen_alpha_all[t-1, g] * total_RE_error[:, t-1]
                ramp = actual_power_t - actual_power_t_1
                R_res.append(ramp <= gen_ramp_rate[g])
                R_res.append(ramp >= -gen_ramp_rate[g])

    if len(R_res) > 0:
        res = np.vstack(P_res + L_res + R_res).T
    else:
        res = np.vstack(P_res + L_res).T
    satisfied_rate = np.mean(np.all(res, axis=1))

    P_res_arr = np.vstack(P_res).T
    L_res_arr = np.vstack(L_res).T
    P_satisfy = np.mean(np.all(P_res_arr, axis=1))
    L_satisfy = np.mean(np.all(L_res_arr, axis=1))
    print(f"[check_JCC] Pmax/Pmin satisfy rate: {P_satisfy*100:.1f}%, line-flow satisfy rate: {L_satisfy*100:.1f}%", end="")
    if len(R_res) > 0:
        R_res_arr = np.vstack(R_res).T
        R_satisfy = np.mean(np.all(R_res_arr, axis=1))
        print(f", ramp satisfy rate: {R_satisfy*100:.1f}%")
    else:
        print("")

    return satisfied_rate

def dual_norm_constr(prob, lhs, rhs, norm_ord=2):
    if norm_ord == 1:
        return [lhs >= rhs, lhs >= -rhs]
    elif norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= rhs @ rhs, lhs_anc == lhs]
    elif norm_ord == np.inf:
        rhs_anc = prob.addMVar(rhs.shape, lb=0, ub=GRB.INFINITY)
        return [lhs >= rhs_anc.sum(), rhs_anc >= rhs, rhs_anc >= -rhs]

def dual_norm_constr_exact_method(prob, lhs, rhs_list, norm_ord=2):
    if norm_ord == 2:
        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        return [lhs_anc * lhs_anc >= gp.quicksum([exp @ exp for exp in rhs_list]), lhs_anc == lhs]
    elif norm_ord == 1:
        return [lhs >= rhs for rhs in rhs_list] + [lhs >= -rhs for rhs in rhs_list]
    else:
        raise NotImplementedError(f'Only 2-norm is implemented, but got {norm_ord}.')

def solve_PD(T, num_gen, num_WT, num_branch, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_train,
              P_line_limit, gen_bus_list, WT_bus_list, N_WDR, epsilon, theta, MIPGap, rng, bigM,
              gen_cost, gen_cost_quadra, gurobi_seed, method="EIFICA",
              njobs = 1, log_file_name = None, thread = 16, norm_ord = 2,
              num_Solar=0, Solar_pred=None, Solar_error_scenarios_train=None, Solar_bus_list=None,
              gen_ramp_rate=None, time_limit=14400,
              debug_log=False):
    PTDF[np.abs(PTDF) < 1e-5] = 0
    t_start_total = time.time()

    random_var_scenario_index = rng.choice(WT_error_scenarios_train.shape[0], N_WDR, replace=False)
    WT_error_scenarios_train = WT_error_scenarios_train[random_var_scenario_index, :, :]

    total_RE_error_train = WT_error_scenarios_train.sum(axis=-1)
    if Solar_error_scenarios_train is not None:
        Solar_error_scenarios_train = Solar_error_scenarios_train[random_var_scenario_index, :, :]
        total_RE_error_train = total_RE_error_train + Solar_error_scenarios_train.sum(axis=-1)

    N_WDR_indices = np.arange(N_WDR)

    t_g_list = [(t, g) for t in range(1, T) for g in range(num_gen)]

    k = int(np.floor(N_WDR * epsilon))
    m0 = max(2 * k, int(np.ceil(0.05 * N_WDR)))
    m1 = max(5, int(np.ceil(0.01 * N_WDR)))
    _env_m0 = os.environ.get('EIFICA_M0', '').strip()
    _env_m1 = os.environ.get('EIFICA_M1', '').strip()
    if _env_m0:
        m0 = int(_env_m0)
    if _env_m1:
        m1 = int(_env_m1)
    max_iter = int(os.environ.get('EIFICA_MAX_ITER', 30))
    tol = 1e-3

    if method == 'EIFICA':
        print(f"[EIFICA] N_WDR={N_WDR}, epsilon={epsilon}, k={k}, m0={m0}, m1={m1}, max_iter={max_iter}")
        key_sets = select_envelope_samples(total_RE_error_train, t_g_list, m0)
    elif method == 'FICA':
        print(f"[FICA] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}
    else:
        print(f"[{method}] N_WDR={N_WDR}, epsilon={epsilon}, theta={theta}")
        key_sets = {}

    iteration_log = []

    prev_alpha = None
    final_solution = None
    final_prob = None

    for it in range(max_iter + 1):
        iter_time_start = time.time()
        prob = gp.Model('ED')
        gen_power_all = prob.addMVar((T, num_gen), lb=-GRB.INFINITY, ub=GRB.INFINITY)
        gen_alpha_all = prob.addMVar((T, num_gen), lb=0, ub=1)

        for t in range(T):
            total_renewable = WT_pred[t, :].sum()
            if Solar_pred is not None:
                total_renewable = total_renewable + Solar_pred[t, :].sum()
            prob.addConstr(gen_power_all[t, :].sum() + total_renewable == load_bus_all[t, :].sum())

            has_RE_uncertainty = (WT_pred[t, :].sum() > 1e-6)
            if Solar_pred is not None:
                has_RE_uncertainty = has_RE_uncertainty or (Solar_pred[t, :].sum() > 1e-6)
            if has_RE_uncertainty:
                prob.addConstr(gen_alpha_all[t, :].sum() == 1)
            else:
                prob.addConstr(gen_alpha_all[t, :] == 0)

            prob.addConstr(gen_power_all[t, :] <= gen_cap_individual)
            prob.addConstr(gen_power_all[t, :] >= gen_pmin_individual)

        if gen_ramp_rate is not None:
            for t in range(1, T):
                prob.addConstr(gen_power_all[t, :] - gen_power_all[t-1, :] <= gen_ramp_rate)
                prob.addConstr(gen_power_all[t-1, :] - gen_power_all[t, :] <= gen_ramp_rate)

        s = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
        r = prob.addMVar(N_WDR, lb=0, ub=GRB.INFINITY)
        if method == 'ExactLHS':
            z = prob.addMVar(N_WDR, vtype=GRB.BINARY)
            prob.addConstr(bigM * (1 - z) >= s - r)
            prob.addConstr(gp.quicksum(z) <= N_WDR * epsilon)
            bAx_list = []

        PTDF[np.abs(PTDF) < 1e-5] = 0
        PTDF_gen = PTDF[:, gen_bus_list].T
        PTDF_wind = PTDF[:, WT_bus_list].T
        PTDF_solar = PTDF[:, Solar_bus_list].T if Solar_bus_list is not None else None
        PTDF_load = PTDF.T

        for t in range(T):
            for g in range(num_gen):
                if method == 'CVAR':
                    b_Ax = gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] >= s - r[N_WDR_indices])
                elif method == 'ExactLHS':
                    bAx_list.append(gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0)))
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] - gen_power_all[t, g] + bigM * z[N_WDR_indices] >= 0)
                elif method in ['EIFICA', 'FICA']:
                    b_Ax = gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    k_local = int(np.floor(N_WDR * epsilon))
                    random_elements = total_RE_error_train[:,t]
                    q_p_plus_base = np.sort(random_elements)[k_local]
                    q_p_minus_base = np.sort(random_elements)[N_WDR-k_local-1]
                    N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                    N_p_minus = np.where(random_elements > q_p_minus_base)[0]
                    if len(N_p_plus) > 0:
                        prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_p_plus,t:t+1].T - gen_power_all[t, g] >= s - r[N_p_plus])
                    if len(N_p_minus) > 0:
                        prob.addConstr(gen_cap_individual[g] + gen_alpha_all[t, g] * total_RE_error_train[N_p_minus,t:t+1].T - gen_power_all[t, g] >= s - r[N_p_minus])
                    prob.addConstr(q_p_plus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)
                    prob.addConstr(q_p_minus_base * gen_alpha_all[t, g] + gen_cap_individual[g] - gen_power_all[t, g] >= s)

        for t in range(T):
            for g in range(num_gen):
                if method == 'CVAR':
                    b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] >= s - r[N_WDR_indices])
                elif method == 'ExactLHS':
                    bAx_list.append(-gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0)))
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_WDR_indices,t] + gen_power_all[t, g] + bigM * z[N_WDR_indices] >= 0)
                elif method in ['EIFICA', 'FICA']:
                    b_Ax = -gen_alpha_all[t, g] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0))
                    prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))
                    k_local = int(np.floor(N_WDR * epsilon))
                    random_elements = total_RE_error_train[:,t]
                    q_p_plus_base = np.sort(random_elements)[k_local]
                    q_p_minus_base = np.sort(random_elements)[N_WDR-k_local-1]
                    N_p_plus = np.where(random_elements < q_p_plus_base)[0]
                    N_p_minus = np.where(random_elements > q_p_minus_base)[0]
                    if len(N_p_plus) > 0:
                        prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_p_plus,t:t+1].T + gen_power_all[t, g] >= s - r[N_p_plus])
                    if len(N_p_minus) > 0:
                        prob.addConstr(-gen_pmin_individual[g] - gen_alpha_all[t, g] * total_RE_error_train[N_p_minus,t:t+1].T + gen_power_all[t, g] >= s - r[N_p_minus])
                    prob.addConstr(-q_p_plus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)
                    prob.addConstr(-q_p_minus_base * gen_alpha_all[t, g] - gen_pmin_individual[g] + gen_power_all[t, g] >= s)

        if method == 'ExactLHS':
            bAx_list = [bAx * N_WDR * theta for bAx in bAx_list]
            prob.addConstrs(
                constr for constr in dual_norm_constr_exact_method(
                    prob, epsilon * N_WDR * s - gp.quicksum(r), bAx_list, norm_ord=norm_ord
                )
            )

        PTDF_gen = PTDF[:, gen_bus_list]
        PTDF_wind = PTDF[:, WT_bus_list]
        PTDF_solar = PTDF[:, Solar_bus_list] if Solar_bus_list is not None else None
        PTDF_load = PTDF

        t_l_list = [(t, l) for t in range(T) for l in range(num_branch)]

        bAx_line_list = [] if method == 'ExactLHS' else None

        for t, l in t_l_list:
            if method == 'ExactLHS':
                b_Ax_wind = -PTDF_wind[l]
                b_Ax_solar = -PTDF_solar[l] if (PTDF_solar is not None and num_Solar > 0) else np.zeros(0)
                b_Ax_combined = np.concatenate([b_Ax_wind, b_Ax_solar])
                b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0)) + b_Ax_combined
                bAx_line_list.append(b_Ax)

                line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T
                line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T
                line_flow_load = PTDF_load[l] @ load_bus_all[t]

                if PTDF_solar is not None and Solar_error_scenarios_train is not None:
                    line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind - line_flow_solar + line_flow_load + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind - line_flow_solar + line_flow_load + bigM * z[N_WDR_indices] >= 0)
                else:
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind + line_flow_load + bigM * z[N_WDR_indices] >= s - r[N_WDR_indices])
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind + line_flow_load + bigM * z[N_WDR_indices] >= 0)
            else:
                b_Ax_wind = -PTDF_wind[l]
                b_Ax_solar = -PTDF_solar[l] if (PTDF_solar is not None and num_Solar > 0) else np.zeros(0)
                b_Ax_combined = np.concatenate([b_Ax_wind, b_Ax_solar])
                b_Ax = PTDF_gen[l] @ gen_alpha_all[t] * np.ones(num_WT + (num_Solar if num_Solar > 0 else 0)) + b_Ax_combined
                prob.addConstrs(constr for constr in dual_norm_constr(prob, epsilon * N_WDR * s - r.sum(), theta * N_WDR * b_Ax, norm_ord=norm_ord))

                line_flow_gen = PTDF_gen[l] @ gen_power_all[t] - PTDF_gen[l] @ gen_alpha_all[t] * total_RE_error_train[N_WDR_indices,t:t+1].T
                line_flow_wind = PTDF_wind[l] @ WT_pred[t] + PTDF_wind[l] @ WT_error_scenarios_train[N_WDR_indices,t].T
                line_flow_load = PTDF_load[l] @ load_bus_all[t]

                if PTDF_solar is not None and Solar_error_scenarios_train is not None:
                    line_flow_solar = PTDF_solar[l] @ Solar_pred[t] + PTDF_solar[l] @ Solar_error_scenarios_train[N_WDR_indices,t].T
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind - line_flow_solar + line_flow_load >= s - r[N_WDR_indices])
                else:
                    prob.addConstr(P_line_limit[l] - line_flow_gen - line_flow_wind + line_flow_load >= s - r[N_WDR_indices])

        if gen_ramp_rate is not None:
            ramp_constr_count = 0
            for (t, g) in t_g_list:
                if method == 'EIFICA':
                    sel_idx = key_sets.get((t, g), None)
                    if sel_idx is None or len(sel_idx) == 0:
                        sel_idx = N_WDR_indices
                    sel_idx = np.asarray(sel_idx, dtype=int)
                else:
                    sel_idx = N_WDR_indices

                alpha_t = gen_alpha_all[t, g]
                alpha_t_minus_1 = gen_alpha_all[t-1, g]
                delta_P = gen_power_all[t, g] - gen_power_all[t-1, g]

                if method in ['FICA', 'CVAR', 'EIFICA']:
                    num_RE = num_WT + (num_Solar if num_Solar > 0 else 0)
                    lhs_expr = epsilon * N_WDR * s - r.sum()
                    if norm_ord == 1:
                        prob.addConstr(lhs_expr >= theta * N_WDR * alpha_t)
                        prob.addConstr(lhs_expr >= -theta * N_WDR * alpha_t)
                        prob.addConstr(lhs_expr >= theta * N_WDR * alpha_t_minus_1)
                        prob.addConstr(lhs_expr >= -theta * N_WDR * alpha_t_minus_1)
                    elif norm_ord == 2:
                        lhs_anc = prob.addMVar(1, lb=0, ub=GRB.INFINITY)
                        prob.addConstr(lhs_anc * lhs_anc >= num_RE * (alpha_t * alpha_t + alpha_t_minus_1 * alpha_t_minus_1))
                        prob.addConstr(lhs_anc == lhs_expr / (theta * N_WDR))

                prob.addConstr(gen_ramp_rate[g] - delta_P
                               + alpha_t * total_RE_error_train[sel_idx, t]
                               - alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                               >= s - r[sel_idx])

                prob.addConstr(gen_ramp_rate[g] + delta_P
                               - alpha_t * total_RE_error_train[sel_idx, t]
                               + alpha_t_minus_1 * total_RE_error_train[sel_idx, t-1]
                               >= s - r[sel_idx])
                ramp_constr_count += 2 * len(sel_idx)
            if debug_log:
                print(f"Iter {it}: added ~{ramp_constr_count} ramp constraints (method={method})")

        FC = gen_cost * gen_power_all + gen_cost_quadra * gen_power_all ** 2
        prob.setObjective(FC.sum(), GRB.MINIMIZE)
        prob.setParam('MIPGap', MIPGap)
        prob.setParam('Seed', gurobi_seed)
        prob.setParam('Threads', thread)
        if time_limit is not None and time_limit > 0:
            prob.setParam('TimeLimit', time_limit)
        if log_file_name is not None:
            prob.setParam('LogFile', log_file_name)
            for tt in range(T):
                for gg in range(num_gen):
                    try:
                        gen_power_all[tt, gg].start = prev_p[tt, gg]
                        gen_alpha_all[tt, gg].start = prev_a[tt, gg]
                    except Exception:
                        pass

        prob.optimize()

        if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
            raise ValueError(f'Iter {it}: Solver failed with status {prob.status}')

        sol_gen_power = gen_power_all.X.copy()
        sol_gen_alpha = gen_alpha_all.X.copy()
        final_solution = {'gen_power_all': sol_gen_power, 'gen_alpha_all': sol_gen_alpha, 'obj': prob.objVal}
        final_prob = prob

        if method != 'EIFICA':
            print(f"[{method}] optimization finished in one pass.")
            break

        if prev_alpha is None:
            alpha_diff = np.inf
        else:
            alpha_diff = np.max(np.abs(sol_gen_alpha - prev_alpha))
        prev_alpha = sol_gen_alpha.copy()

        sol_s = float(s.X[0])
        viol_tol = 1e-4
        any_added = False
        max_violation = 0.0
        per_constraint_added = {}
        total_key_samples_before = sum(len(v) for v in key_sets.values())
        for (t, g) in t_g_list:
            xi_t = total_RE_error_train[:, t]
            xi_t1 = total_RE_error_train[:, t-1]
            delta_P = sol_gen_power[t, g] - sol_gen_power[t-1, g]
            ramp_term = sol_gen_alpha[t, g] * xi_t - sol_gen_alpha[t-1, g] * xi_t1
            lhs_upper = gen_ramp_rate[g] - delta_P + ramp_term
            lhs_lower = gen_ramp_rate[g] + delta_P - ramp_term
            rho = sol_s - np.minimum(lhs_upper, lhs_lower)
            cur_set = set(key_sets.get((t, g), np.array([], dtype=int)).tolist())
            viol_idx = np.where(rho > viol_tol)[0]
            viol_idx = np.array([i for i in viol_idx if i not in cur_set], dtype=int)
            if len(viol_idx) > 0:
                max_violation = max(max_violation, float(rho[viol_idx].max()))
                if len(viol_idx) > m1:
                    sel = np.argsort(-rho[viol_idx])[:m1]
                    viol_idx = viol_idx[sel]
                new_set = cur_set.union(set(viol_idx.tolist()))
                added_count = len(new_set) - len(cur_set)
                if added_count > 0:
                    any_added = True
                    per_constraint_added[(t, g)] = added_count
                    key_sets[(t, g)] = np.array(sorted(new_set), dtype=int)
        total_key_samples_after = sum(len(v) for v in key_sets.values())

        iter_time = time.time() - iter_time_start
        iter_log_entry = {
            'iter': it,
            'obj': final_solution['obj'],
            'alpha_diff': float(alpha_diff),
            'any_added': any_added,
            'max_violation': float(max_violation),
            'total_key_samples_before': int(total_key_samples_before),
            'total_key_samples_after': int(total_key_samples_after),
            'per_constraint_added_sample_counts': per_constraint_added,
            'iter_time_s': float(iter_time)
        }
        iteration_log.append(iter_log_entry)

        print(f"[Iter {it}] obj={final_solution['obj']:.4f} alpha_diff={alpha_diff:.3e} "
              f"max_viol={max_violation:.3e} added={total_key_samples_after-total_key_samples_before} "
              f"key_total={total_key_samples_after} time={iter_time:.1f}s")

        if not any_added:
            print(f"[EIFICA] converged at iter {it}: no full-sample ramp violation (max_viol={max_violation:.3e})")
            break
        if it == max_iter:
            print(f"[EIFICA] WARNING: hit max_iter={max_iter}, still max_viol={max_violation:.3e} (NOT fully converged)")
    solve_time = time.time() - t_start_total
    results = {
        'prob': final_prob,
        'gen_power_all': final_solution['gen_power_all'],
        'gen_alpha_all': final_solution['gen_alpha_all'],
        'obj_value': float(final_solution['obj']),
        'solve_time': float(solve_time),
        'iteration_log': iteration_log,
        'key_sets': key_sets,
        'status': final_prob.status if final_prob is not None else None,
        'method': method,
        'epsilon': epsilon,
        'theta': theta,
        'N_WDR': N_WDR
    }
    return results

def solve_PD_instance(num_gen=38, num_WT=10, num_Solar=0, Tstart=0, norm_ord=1, T=24,
                     method='EIFICA', N_WDR=100, epsilon=0.05, theta=1.5e-1,
                     load_scaling_factor=1, solar_mode='auto', show_plot=True,
                     network_name='case24_ieee_rts', seed=0,
                     time_limit=14400, MIPGap=0.001):
    N_samples_train = 1000
    N_samples_test = 5000
    thread = 4

    gurobi_seed = 0

    gen_cap_total_prop = 1

    bigM =1e5
    log_file_name = None

    network_dict = {'case118': ppnw.case118(),
                    'case300': ppnw.case300(),
                    'case24_ieee_rts': ppnw.case24_ieee_rts(),
                    'case5': ppnw.case5(),
                    'case4gs': ppnw.case4gs(),
                    'case_ieee30': ppnw.case_ieee30()}

    rng = np.random.RandomState(0)
    rng_sample = np.random.RandomState(seed)
    rng_fixed = np.random.RandomState(0)

    network = network_dict[network_name]

    load_location = os.path.join(os.getcwd(), 'data', 'UK_norm_load_curve_highest.npy')
    network_load = np.load(load_location)
    network_load = np.mean(np.vstack([network_load[::2],
                                      network_load[1::2]]), axis=0)
    network_load = np.tile(network_load, 2)
    network_load = network_load[Tstart:Tstart+T]

    pp.rundcpp(network)
    _, ppci = _pd2ppc(network)
    bus_info = ppci['bus']
    branch_info = ppci['branch']
    PTDF = makePTDF(ppci["baseMVA"], bus_info, branch_info,
                    using_sparse_solver=False)

    num_branch = len(branch_info)

    load_bus_size = bus_info[:, 2] * load_scaling_factor

    load_total = np.sum(load_bus_size)
    load_bus_all = load_bus_size.reshape(1, -1) * network_load.reshape(-1, 1)

    gen_cap_total = load_total * gen_cap_total_prop
    gen_cap_individual = gen_cap_total / num_gen
    gen_cap_individual = rng_fixed.uniform(0.6, 1.4, num_gen) * gen_cap_individual
    gen_pmin_individual = 0.1 * gen_cap_individual

    gen_ramp_rate = 0.6*gen_cap_individual

    gen_cost = rng.uniform(23.13, 57.03, num_gen)
    gen_cost_quadra = rng.uniform(0.002, 0.008, num_gen)



    bus_list = np.arange(bus_info.shape[0])
    gen_bus_list = rng_fixed.choice(bus_list, num_gen, replace=True)
    WT_bus_list = rng_fixed.choice(bus_list, num_WT, replace=True)
    Solar_bus_list = rng_fixed.choice(bus_list, num_Solar, replace=True) if num_Solar > 0 else None

    P_line_limit = np.abs(ppci['branch'][:, 5])
    P_line_limit = np.clip(P_line_limit, 0, 2 * load_total)

    if num_WT > 0:
        raise NotImplementedError("Wind is not supported; run with num_WT=0.")
    else:
        print("\n=== Wind Power Disabled (num_WT=0) ===")
        num_WT = 1
        WT_bus_list = np.array([0])
        WT_pred = np.zeros((T, num_WT))
        WT_error_scenarios_train = np.zeros((N_samples_train, T, num_WT))
        WT_error_scenarios_test = np.zeros((N_samples_test, T, num_WT))

    if num_Solar > 0:
        Solar_total = 0.45 * load_total
        Solar_individual = Solar_total / num_Solar

        Solar_pred, Solar_error_scenarios, Solar_full_scenarios = Solar_sce_gen(num_Solar, N_samples_train + N_samples_test)
        Solar_pred = Solar_pred[Tstart:Tstart+T] * Solar_individual
        Solar_error_scenarios = Solar_error_scenarios[:, Tstart:Tstart+T] * Solar_individual
        Solar_full_scenarios = Solar_full_scenarios[:, Tstart:Tstart+T] * Solar_individual

        for t in range(T):
            hour_of_day = (Tstart + t) % 24
            if hour_of_day < 6 or hour_of_day >= 18:
                Solar_pred[t, :] = 0
                Solar_error_scenarios[:, t, :] = 0
                Solar_full_scenarios[:, t, :] = 0

        Solar_error_scenarios_train = Solar_error_scenarios[:N_samples_train]
        Solar_error_scenarios_test = Solar_error_scenarios[N_samples_train:]

        print(f"\n=== Solar Integration ===")
        print(f"Solar total capacity: {Solar_total:.2f} MW ({0.45*100:.0f}% of load)")
        print(f"Individual solar station: {Solar_individual:.2f} MW")
        print(f"Daylight hours: 6:00-18:00 (time window: {Tstart}-{Tstart+T})")
    else:
        Solar_pred = None
        Solar_error_scenarios_train = None
        Solar_error_scenarios_test = None
        Solar_bus_list = None

    input_param_dict = {'T': T, 'num_gen': num_gen, 'num_WT': num_WT, 'num_branch': num_branch,
                        'load_bus_all': load_bus_all, 'PTDF': PTDF, 'gen_cap_individual': gen_cap_individual,
                        'gen_pmin_individual': gen_pmin_individual, 'WT_pred': WT_pred,
                        'WT_error_scenarios_train': WT_error_scenarios_train, 'P_line_limit': P_line_limit,
                        'gen_bus_list': gen_bus_list, 'WT_bus_list': WT_bus_list, 'N_WDR': N_WDR, 'epsilon': epsilon,
                        'thread': thread,
                        'theta': theta, 'method': method, 'MIPGap': MIPGap, 'gen_cost': gen_cost,
                        'gen_cost_quadra': gen_cost_quadra, 'bigM': bigM, 'gurobi_seed': gurobi_seed,
                        'log_file_name': log_file_name, 'rng': rng_sample, "norm_ord": norm_ord,
                        'num_Solar': num_Solar, 'Solar_pred': Solar_pred,
                        'Solar_error_scenarios_train': Solar_error_scenarios_train, 'Solar_bus_list': Solar_bus_list,
                        'gen_ramp_rate': gen_ramp_rate, 'time_limit': time_limit}
    solve_results = solve_PD(**input_param_dict)

    prob = solve_results['prob']
    gen_power_all = solve_results['gen_power_all']
    gen_alpha_all = solve_results['gen_alpha_all']

    if hasattr(gen_power_all, 'X'):
        gen_power_all = gen_power_all.X
    if hasattr(gen_alpha_all, 'X'):
        gen_alpha_all = gen_alpha_all.X

    if prob.status not in [GRB.Status.OPTIMAL, GRB.Status.TIME_LIMIT, GRB.Status.SUBOPTIMAL]:
        raise ValueError('The problem does not have a feasible solution.')

    t_solve = solve_results.get('solve_time', prob.Runtime)

    satisfied_rate = check_JCC(T, num_gen, num_branch, gen_power_all, gen_alpha_all, load_bus_all, PTDF, gen_cap_individual,
              gen_pmin_individual, WT_pred, WT_error_scenarios_test, P_line_limit, gen_bus_list, WT_bus_list,
              Solar_pred, Solar_error_scenarios_test, Solar_bus_list, gen_ramp_rate)

    print('------------------------------------')
    print(f'{network_name}, {num_gen} generators, {T}-step horizon')
    print(f'Risk level {epsilon}, radius {theta}, N_WDR {N_WDR}')
    print('')
    print(f'the objective value is {prob.objVal}, the out-of-sample JCC rate is {satisfied_rate*100}%')
    print(f'The method used is {method}')
    print(f'The computing time for solving the dispatch is {t_solve} seconds')
    print('')
    print('------------------------------------')
    if show_plot:
        plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                      WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost,
                      Solar_pred, Solar_error_scenarios_test, gen_ramp_rate)

    results = {
        'prob': prob,
        'gen_power_all': gen_power_all,
        'gen_alpha_all': gen_alpha_all,
        'obj_value': prob.objVal,
        'solve_time': t_solve,
        'satisfied_rate': satisfied_rate,
        'status': prob.status,
        'network_name': network_name,
        'iteration_log': solve_results.get('iteration_log'),
        'final_key_total': int(sum(len(v) for v in solve_results.get('key_sets', {}).values())) if solve_results.get('key_sets') else 0,
        'num_gen': num_gen,
        'T': T,
        'method': method,
        'epsilon': epsilon,
        'theta': theta,
        'N_WDR': N_WDR
    }
    return results

def plot_all_gen(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost):
    rng = np.random.RandomState(0)
    num_plot_gen = min(5, num_gen)
    top_pick = max(int(0.6 * num_gen), num_plot_gen)
    plot_gen_index = rng.choice(np.argsort(gen_cap_individual)[:top_pick], num_plot_gen, replace=False)
    num_plot_sce = 3
    fig, axs = plt.subplots(num_plot_gen, num_plot_sce, figsize=(5*num_plot_sce, 2 * num_plot_gen))
    for i in range(3):
        ax = axs[:, i]
        for ig, g in enumerate(plot_gen_index):
            x = np.arange(T)
            ax[ig].step(x, gen_power_all[:, g], label='first-stage')
            ax[ig].step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * WT_error_scenarios_test[i].sum(axis=-1), label='actual')
            ax[ig].set_xlabel('hour')
            ax[ig].axhline(gen_pmin_individual[g], color='black', linestyle='--')
            ax[ig].axhline(gen_cap_individual[g], color='black', linestyle='--')
            ax[ig].legend()
            ax[ig].set_title(f'scenario {i}, {method}, generator {g}, eps {epsilon}, theta {theta}')
    plt.tight_layout()
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}.png')
    plt.savefig(save_name, dpi=300)
    plt.show()

def plot_paper(num_gen, gen_power_all, gen_alpha_all, gen_cap_individual, gen_pmin_individual, WT_pred,
                  WT_error_scenarios_test, method, epsilon, theta, network_name, T, gen_cost,
                  Solar_pred=None, Solar_error_scenarios_test=None, gen_ramp_rate=None):
    alpha_std = np.std(gen_alpha_all, axis=0)
    plot_gen_index = np.argsort(alpha_std)[-2:][::-1]
    print(f"\n=== Plotting Generators with Highest AGC Variation ===")
    print(f"Selected: Gen {plot_gen_index[0]} (std={alpha_std[plot_gen_index[0]]:.6f}), Gen {plot_gen_index[1]} (std={alpha_std[plot_gen_index[1]]:.6f})")

    num_plot_sce = 1
    rng = np.random.RandomState(10)
    scenario_set = rng.choice(WT_error_scenarios_test.shape[0], num_plot_sce, replace=False)

    print(f"\n=== Selected Scenario (Random) ===")
    sce_i = scenario_set[0]
    print(f"Selected scenario index: {sce_i}")

    wt_error = WT_error_scenarios_test.sum(axis=-1)[sce_i]
    print(f"WT error in this scenario: mean={wt_error.mean():.4f}, std={wt_error.std():.4f}, range=[{wt_error.min():.4f}, {wt_error.max():.4f}]")

    total_error = wt_error
    if Solar_error_scenarios_test is not None:
        solar_error = Solar_error_scenarios_test.sum(axis=-1)[sce_i]
        print(f"Solar error in this scenario: mean={solar_error.mean():.4f}, std={solar_error.std():.4f}, range=[{solar_error.min():.4f}, {solar_error.max():.4f}]")
        total_error = total_error + solar_error
    print(f"Total RE error in this scenario: mean={total_error.mean():.4f}, std={total_error.std():.4f}, range=[{total_error.min():.4f}, {total_error.max():.4f}]")


    alpha_abs_sum = np.abs(gen_alpha_all).sum(axis=0)
    alpha_std_all = np.std(gen_alpha_all, axis=0)
    non_zero_gens = np.where(alpha_abs_sum > 1e-6)[0]

    top_std_gens = np.argsort(alpha_std_all)[-10:][::-1]


    num_rows = 0
    if Solar_pred is not None:
        num_rows += 1
    num_rows += len(plot_gen_index) * 3

    fig, axs = plt.subplots(num_rows, num_plot_sce, figsize=(10*num_plot_sce, 3.15 * num_rows))
    if num_plot_sce <= 1:
        axs = axs[..., None]

    row_idx = 0


    if Solar_pred is not None and Solar_error_scenarios_test is not None:
        for i in range(num_plot_sce):
            sce_i = scenario_set[i]
            ax_s = axs[row_idx, i]
            x = np.arange(T)
            ax_s.step(x, Solar_pred.sum(axis=-1), label='forecast',  where='post')
            ax_s.step(x, Solar_pred.sum(axis=-1) + Solar_error_scenarios_test.sum(axis=-1)[sce_i], label='actual', where='post')
            ax_s.set_title(f'Solar Farm')
            ax_s.set_ylabel('Solar (MW)')
            ax_s.set_xlim(-0.5, T-0.5)
            ax_s.set_xticks(np.arange(T))
            ax_s.set_xticklabels(np.arange(T))
            ax_s.set_xlabel('Hour')
            ax_s.legend()
            ax_s.grid(True, linestyle='--', alpha=0.3)
        row_idx += 1
    gen_start_row = row_idx
    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3, i]
            x = np.arange(T)
            ax.step(x, gen_power_all[:, g], label='first-stage', where='post')
            total_RE_error_test = WT_error_scenarios_test.sum(axis=-1)[sce_i]
            if Solar_error_scenarios_test is not None:
                total_RE_error_test = total_RE_error_test + Solar_error_scenarios_test.sum(axis=-1)[sce_i]
            ax.step(x, gen_power_all[:, g] - gen_alpha_all[:, g] * total_RE_error_test, label='actual', where='post')
            ax.set_title(f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh')
            ax.set_ylabel(f'Gen (MW)')
            ax.axhline(gen_pmin_individual[g], color='black', linestyle='--')
            ax.axhline(gen_cap_individual[g], color='black', linestyle='--')
            ax.set_xlim(-0.5, T-0.5)
            ax.set_xticks(np.arange(T))
            ax.set_xticklabels(np.arange(T))
            ax.set_xlabel('Hour')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)

    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3 + 1, i]
            x = np.arange(T) + 0.5
            ax.bar(x, gen_alpha_all[:, g], width=0.8, label='AGC', color='violet', align='center')
            ax.set_title(f'Gen {g}, Cost {gen_cost[g]:.2f} USD/MWh')
            ax.set_ylabel('AGC Factor')
            ax.set_ylim(-1, 1)
            ax.set_xlim(-0.5, T-0.5)
            ax.set_xticks(np.arange(T))
            ax.set_xticklabels(np.arange(T))
            ax.set_xlabel('Hour')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()

    for i in range(num_plot_sce):
        sce_i = scenario_set[i]
        total_RE_error_test = WT_error_scenarios_test.sum(axis=-1)[sce_i]
        if Solar_error_scenarios_test is not None:
            total_RE_error_test = total_RE_error_test + Solar_error_scenarios_test.sum(axis=-1)[sce_i]

        for ig, g in enumerate(plot_gen_index):
            ax = axs[gen_start_row + ig*3 + 2, i]

            first_stage_ramp = np.diff(gen_power_all[:, g])

            actual_power = gen_power_all[:, g] - gen_alpha_all[:, g] * total_RE_error_test
            second_stage_ramp = np.diff(actual_power)

            x = np.arange(1, T) + 0.5

            width = 0.35
            ax.bar(x - width/2, first_stage_ramp, width, label='First-stage', color='steelblue', alpha=0.7)
            ax.bar(x + width/2, second_stage_ramp, width, label='Actual', color='darkorange', alpha=0.7)

            ramp_limit = gen_ramp_rate[g] if gen_ramp_rate is not None else gen_cap_individual[g]
            ax.axhline(ramp_limit, color='red', linestyle='--', linewidth=1.5, label=f'Limit (±{ramp_limit:.1f} MW)')
            ax.axhline(-ramp_limit, color='red', linestyle='--', linewidth=1.5)

            ax.set_ylim(-ramp_limit * 1.2, ramp_limit * 1.2)

            ax.set_title(f'Gen {g} Ramping, Cap={gen_cap_individual[g]:.1f} MW')
            ax.set_ylabel('Ramp (MW/h)')
            ax.set_xlim(-0.5, T-0.5)
            ax.set_xticks(np.arange(0, T))
            ax.set_xticklabels(np.arange(0, T))
            ax.set_xlabel('Hour')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.85))

    plt.subplots_adjust(left=0.08, right=0.95, top=0.97, bottom=0.03, hspace=0.35, wspace=0.3)
    save_dir = os.path.join(os.getcwd(), 'figure', 'test')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_name = os.path.join(save_dir, f'{network_name}_{num_gen}gen_T{T}_{method}_eps{epsilon}_theta{theta}_{timestamp}.pdf')
    plt.savefig(save_name, dpi=300)
    plt.show()

if __name__ == '__main__':
    # ---- experiment switches ----
    method = 'EIFICA'                  # 'FICA' / 'EIFICA' / 'CVAR'
    network_name = 'case24_ieee_rts'   # 'case24_ieee_rts' / 'case118'
    N_WDR = 30                         # number of training scenarios
    epsilon = 0.03                     # JCC risk level
    theta = 0.06                       # Wasserstein radius
    seed = 0                           # random seed (resamples training scenarios)
    num_gen = 38                       # number of generators
    num_Solar = 5                      # number of solar farms
    num_WT = 0                         # number of wind turbines
    Tstart = 0                         # start hour of the horizon
    T = 24                             # horizon length
    solar_mode = 'auto'
    norm_ord = 1                       # dual norm order: 1 / 2 / np.inf
    load_scaling_factor = 1            # scale the base load
    eifica_direct = False              # True -> EIFICA direct (envelope-only); False -> iterative
    show_plot = True                   # save / show the dispatch figure
    time_limit = 14400                 # solver time limit (s)
    MIPGap = 0.001                     # solver MIP gap
    # -----------------------------

    if eifica_direct:
        os.environ['EIFICA_MAX_ITER'] = '0'

    solve_PD_instance(num_gen=num_gen, num_WT=num_WT, num_Solar=num_Solar, Tstart=Tstart,
                     norm_ord=norm_ord, T=T, method=method, N_WDR=N_WDR,
                     epsilon=epsilon, theta=theta, load_scaling_factor=load_scaling_factor,
                     solar_mode=solar_mode, network_name=network_name, seed=seed,
                     show_plot=show_plot, time_limit=time_limit, MIPGap=MIPGap)