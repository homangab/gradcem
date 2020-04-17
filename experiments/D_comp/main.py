import numpy as np
import torch
from mpc.test_energy import test_energy2d, NavigateGTEnv
import matplotlib.pyplot as plt
from mpc.cem import CEM
from mpc.grad import GradPlan
from mpc.gradcem import GradCEMPlan
from tqdm import tqdm

def run(planner, env):
    planner.set_env(env)
    plan = planner.forward(1, return_plan=True)
    env.reset_state(1)
    r = env.rollout(plan)
    # print(r.size())
    return r.item()

def run_mult(planner, env, N):
    rs = []
    for i in tqdm(range(N)):
        rs.append(run(planner,env))
    rs = np.array(rs)
    m = np.mean(rs)
    std = np.std(rs)
    return m, std

if __name__ == "__main__":
    # Compare CEM to GradPlan as we increase D
    comp_device = torch.device('cuda:0')
    H = 50

    K = 20
    tK = 4
    opt_iter = 10
    cem_planner = CEM(H, opt_iter, K, tK, None, device=comp_device)

    # K = 20
    # opt_iter = 10
    grad_planner = GradPlan(H, opt_iter, K, None, device=comp_device)

    gradcem_planner = GradCEMPlan(H, opt_iter, K, tK, None, device=comp_device)

    B = 1
    grad_return = []
    cem_return = []
    grad_std = []
    cem_std = []

    gradcem_return = []
    gradcem_std = []
    for D in range(2,21):
        print(D)
        env = NavigateGTEnv(B, D, test_energy2d, comp_device, control='force', sparse_r_step=H-1, dt=2.5/H)

        print("Running grad+cem planner")
        m, std = run_mult(gradcem_planner, env,20)
        gradcem_return.append(m)
        gradcem_std.append(std)

        m, std = run_mult(grad_planner, env,20)
        grad_return.append(m)
        grad_std.append(std)

        m, std = run_mult(cem_planner, env,20)
        cem_return.append(m)
        cem_std.append(std)

    # print(grad_return)
    # print(grad_std)
    # print(cem_return)
    # print(cem_std)
    grad_return = np.array(grad_return)
    grad_std = np.array(grad_std)
    cem_return = np.array(cem_return)
    cem_std = np.array(cem_std)
    gradcem_return = np.array(gradcem_return)
    gradcem_std = np.array(gradcem_std)
    # results = np.stack((grad_return, grad_std, cem_return, cem_std))
    results = np.stack((grad_return, grad_std, cem_return, cem_std, gradcem_return, gradcem_std))

    np.save("results_new.npy", results)
