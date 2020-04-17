import torch
from torch import jit
from torch import nn, optim


class GradCEMPlan():  # jit.ScriptModule):
    def __init__(self, planning_horizon, opt_iters, samples, top_samples, env, device, grad_clip=True):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.top_K = top_samples
        self.device = device
        self.grad_clip = grad_clip

    def set_env(self, env):
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False):
        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        a_mu = torch.zeros(self.H, B, 1, self.a_size, device=self.device)
        a_std = torch.ones(self.H, B, 1, self.a_size, device=self.device)

        # Sample actions (T x (B*K) x A)
        actions = (a_mu + a_std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)
        actions = torch.tensor(actions, requires_grad=True)

        # optimizer = optim.SGD([actions], lr=0.1, momentum=0)
        optimizer = optim.RMSprop([actions], lr=0.1)
        plan_each_iter = []
        for _ in range(self.opt_iters):
            self.env.reset_state(B*self.K)

            optimizer.zero_grad()

            # Returns (B*K)
            returns = self.env.rollout(actions)
            tot_returns = returns.sum()
            (-tot_returns).backward()

            # grad clip
            # Find norm across batch
            if self.grad_clip:
                epsilon = 1e-6
                max_grad_norm = 1.0
                actions_grad_norm = actions.grad.norm(2.0,dim=2,keepdim=True)+epsilon
                # print("before clip", actions.grad.max().cpu().numpy())

                # Normalize by that
                actions.grad.data.div_(actions_grad_norm)
                actions.grad.data.mul_(actions_grad_norm.clamp(min=0, max=max_grad_norm))
                # print("after clip", actions.grad.max().cpu().numpy())

            optimizer.step()

            _, topk = returns.reshape(B, self.K).topk(self.top_K, dim=1, largest=True, sorted=False)
            topk += self.K * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.H, B, self.top_K, self.a_size)
            a_mu = best_actions.mean(dim=2, keepdim=True)
            a_std = best_actions.std(dim=2, unbiased=False, keepdim=True)

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())

            # There must be cleaner way to do this
            k_resamp = self.K-self.top_K
            _, botn_k = returns.reshape(B, self.K).topk(k_resamp, dim=1, largest=False, sorted=False)
            botn_k += self.K * torch.arange(0, B, dtype=torch.int64, device=self.device).unsqueeze(dim=1)

            resample_actions = (a_mu + a_std * torch.randn(self.H, B, k_resamp, self.a_size, device=self.device)).view(self.H, B * k_resamp, self.a_size)
            actions.data[:, botn_k.view(-1)] = resample_actions.data

        actions = actions.detach()
        # Re-fit belief to the K best action sequences
        _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
        best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size)

        if return_plan_each_iter:
            return plan_each_iter
        if return_plan:
            return best_plan
        else:
            return best_plan[0]

if __name__ == "__main__":
    from test_energy import get_test_energy2d_env
    B = 1
    K = 100
    top_K = 10
    t_env = get_test_energy2d_env(B*K)
    H = 1
    planner = GradCEMPlan(H, 10, K, top_K, t_env, device=torch.device('cpu'))
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
