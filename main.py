import numpy as np
import torch
from mpc.test_energy import test_energy2d, NavigateGTEnv
import matplotlib.pyplot as plt
from mpc.cem import CEM
from mpc.grad import GradPlan
from mpc.gradcem import GradCEMPlan
from mpc.svgd import SVGDPlan

if __name__ == "__main__":

    # config = {
    #         "planner": "CEM"
    # }
    # config = {
    #         "planner": "GradPlan"
    # }
    # config = {
    #         "planner": "GradCEMPlan"
    # }
    config = {
            "planner": "SVGDPlan"
    }

    B = 1
    # comp_device = torch.device('cuda:0')
    comp_device = torch.device('cpu')
    H = 70
    env = NavigateGTEnv(B, 3, test_energy2d, comp_device, control='force', sparse_r_step=H-1, dt=2.0/H, obstacles_env=True)

    planner = None
    if config["planner"] == "CEM":
        K = 100
        tK = 20
        opt_iter = 10
        planner = CEM(H, opt_iter, K, tK, env, device=comp_device)
    elif config["planner"] == "GradCEMPlan":
        K = 100
        tK = 20
        opt_iter = 10
        planner = GradCEMPlan(H, opt_iter, K, tK, env, device=comp_device)
    elif config["planner"] == "GradPlan":
        K = 20
        opt_iter = 10
        planner = GradPlan(H, opt_iter, K, env, device=comp_device)
    elif config["planner"] == "SVGDPlan":
        K = 20
        opt_iter = 10
        planner = SVGDPlan(H, opt_iter, K, env, device=comp_device)

    # actions = planner.forward(B, return_plan=True)
    plans = planner.forward(B, return_plan_each_iter=True)
    # actions = actions.cpu().numpy()

    # Visualize
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    ax.set_aspect('equal')

    env.reset_state(1)
    env.draw_env_2d_proj(ax)

    for actions in plans[0:]:
        env.reset_state(1)
        rs, ss = env.rollout(actions, return_traj=True)
        # ps = [s[0].cpu().numpy() for s in ss]
        # ps = np.array(ps)
        # ps = ps.squeeze()
        env.draw_traj_2d_proj(ax, ss)

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)

    plt.savefig("./experiments/tmp/main.png")
