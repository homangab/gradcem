import torch
from torch import jit
from torch import nn, optim

# class CEM:
#     def __init__(self, candidates, top_candidates):


class CEM():  # jit.ScriptModule):
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, env, device):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K, self.top_K = samples, top_samples
        self.device = device

    def set_env(self, env):
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple CEMs should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.H, B, 1, self.a_size, device=self.device)
        a_std = torch.ones(self.H, B, 1, self.a_size, device=self.device)

        plan_each_iter = []
        for _ in range(self.opt_iters):
            self.env.reset_state(B*self.K)
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            # Sample actions (T x (B*K) x A)
            actions = (a_mu + a_std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)

            # Returns (B*K)
            returns = self.env.rollout(actions)

            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
            topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
            # Update belief with new means and standard deviations
            a_mu = best_actions.mean(dim=2, keepdim=True)
            a_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())
                # plan_each_iter.append(a_mu.squeeze(dim=2).data.clone())

        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return a_mu.squeeze(dim=2)
        else:
            # Return first action mean µ_t
            return a_mu.squeeze(dim=2)[0]

if __name__ == "__main__":
    from test_energy import get_test_energy2d_env
    B = 1
    K = 100
    tK = 10
    t_env = get_test_energy2d_env(B*K)
    H = 1
    planner = CEM(H, 10, K, tK, t_env, device=torch.device('cpu'))
    action = planner.forward(B)
    action = action.cpu().numpy()

    import matplotlib.pyplot as plt
    N = 30
    x = torch.linspace(-1,1,N)
    y = torch.linspace(-1,1,N)
    X, Y = torch.meshgrid(x,y)
    actions_grid = torch.stack((X,Y),dim=-1)
    # print(actions_grid)
    energies = t_env.func(actions_grid.reshape(-1,2))

    plt.pcolormesh(X.numpy(), Y.numpy(), -energies.reshape(N,N).numpy(), cmap="coolwarm")
    plt.contour(X.numpy(), Y.numpy(), -energies.reshape(N,N).numpy(), cmap="seismic")
    plt.scatter(action[:, 0], action[:, 1])
    plt.show()
