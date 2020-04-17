import torch
from torch import jit
from torch import nn, optim, autograd

import numpy as np

def npy(tensor):
    return tensor.cpu().detach().numpy()

def squared_dist(x):
    assert len(x.size()) == 2
    norm = (x ** 2).sum(1).view(-1, 1)
    dn = (norm + norm.view(1, -1)) - 2.0 * (x @ x.t())
    return dn

def rbf_kernel(x, h=None):
    """
    Returns the full kernel matrix for input x
    x: NxC
    Output
    K: NxN where Kij = k(x_i, x_j)
    dK: NxC where dKi = sum_j grad_j Kij
    """
    n, c = x.size()
    sq_dist_mat = squared_dist(x)

    if h is None:
        # Apply median trick for h
        h = torch.clamp(torch.median(sq_dist_mat)/np.log(n+1),1e-3, float('inf')).detach()

    K = torch.exp(-sq_dist_mat/h)
    dK = -(torch.matmul(K, x) - torch.sum(K, dim=-1,keepdim=True)*x)/h

    return K, dK


class SVGDPlan():  # jit.ScriptModule):
    """ Plan with Stein Variational Gradient Descent """
    def __init__(self, planning_horizon, opt_iters, samples, env, device, grad_clip=True):
        super().__init__()
        self.set_env(env)
        self.H = planning_horizon
        self.opt_iters = opt_iters
        self.K = samples
        self.device = device
        self.grad_clip = grad_clip

    def set_env(self, env):
        self.env = env
        if self.env is not None:
            self.a_size = env.a_size

    # @jit.script_method
    def forward(self, batch_size, return_plan=False, return_plan_each_iter=False, alpha=1.0):
        # TODO enable batching by batch the distance matrix computation
        assert batch_size == 1

        # Here batch is strictly if multiple Plans should be performed!
        B = batch_size

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        flat_a_mu = torch.zeros(1, self.H * self.a_size, device=self.device)
        flat_a_std = torch.ones(1, self.H * self.a_size, device=self.device)

        # actions = (a_mu + a_std * torch.randn(self.H, B, self.K, self.a_size, device=self.device)).view(self.H, B * self.K, self.a_size)
        # Sample actions ((B*K) x (T,A))
        flat_actions = (flat_a_mu + flat_a_std * torch.randn(B*self.K, self.H*self.a_size, device=self.device))
        # TODO: debug
        # flat_actions = flat_actions*0
        flat_actions = torch.tensor(flat_actions, requires_grad=True)

        # Dummy op to init grad
        flat_actions.sum().backward()

        # optimizer = optim.SGD([actions], lr=0.1, momentum=0)
        optimizer = optim.RMSprop([flat_actions], lr=0.1)
        plan_each_iter = []
        for _ in range(self.opt_iters):
            self.env.reset_state(B*self.K)

            optimizer.zero_grad()

            # Get log prob gradient
            # Returns (B*K)
            # Use p propto exp(r) -> logp = r + const
            # Need actions in H,B*K,A format
            actions = flat_actions.view(B*self.K, self.H, self.a_size).transpose(0,1)
            returns = self.env.rollout(actions)
            tot_returns = returns.sum()
            grad_scores = autograd.grad(-tot_returns, flat_actions, retain_graph=True)
            assert len(grad_scores) == 1
            grad_scores = grad_scores[0].detach()
            # (-tot_returns).backward()

            print('grad_score', npy(grad_scores))
            print(grad_scores.size())
            # grad clip
            # Find norm across batch
            if self.grad_clip:
                epsilon = 1e-6
                max_grad_norm = 1.0
                grad_scores_norm = grad_scores.norm(2.0,dim=-1,keepdim=True)+epsilon
                # print("before clip", actions.grad.max().cpu().numpy())

                # Normalize by that
                grad_scores.data.div_(grad_scores_norm)
                grad_scores.data.mul_(grad_scores_norm.clamp(min=0, max=max_grad_norm))
                # print("after clip", actions.grad.max().cpu().numpy())

            print('grad_score', npy(grad_scores))

            # Get the kernel matrix and the summed kernel gradients
            # TODO: handle batching
            K, dK = rbf_kernel(flat_actions.view(self.K, self.H*self.a_size))
            print("K", npy(K))
            print("dK", npy(dK))

            # Form SVGD gradient
            svgd = (torch.matmul(K, grad_scores) + alpha * dK).detach()
            print("svgd", npy(svgd))

            # Assign gradient
            # flat_actions.grad = (svgd.clone())
            with torch.no_grad():
                flat_actions.grad.set_(svgd.clone())

            optimizer.step()

            if return_plan_each_iter:
                _, topk = returns.reshape(B, self.K).topk(1, dim=1, largest=True, sorted=False)
                actions = flat_actions.view(B*self.K, self.H, self.a_size).transpose(0,1)
                best_plan = actions[:, topk[0]].reshape(self.H, B, self.a_size).detach()
                plan_each_iter.append(best_plan.data.clone())

        actions = flat_actions.view(B*self.K, self.H, self.a_size).transpose(0,1)
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

    # torch.manual_seed(0)
    B = 1
    K = 10
    # to_K = 10
    t_env = get_test_energy2d_env(B*K)
    H = 1
    opt_iters = 10
    planner = SVGDPlan(H, opt_iters, K, t_env, device=torch.device('cpu'))
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
