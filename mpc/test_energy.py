import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def get_test_energy2d_env(batch_size):
    return FuncMinGTEnv(batch_size, 2, test_energy2d)

def test_energy2d(action_batch):
    assert action_batch.dim() == 2
    assert action_batch.size()[1] == 2

    opt_point = torch.tensor([[1.0, 1.0]],
                    requires_grad=False,
                    device=action_batch.device)

    return ((action_batch-opt_point)**2).sum(-1)

def get_test_energy(opt_point):
    def test_energy(query_batch):
        assert query_batch.dim() == 2
        assert query_batch.size(1) == opt_point.size(-1)

        return ((query_batch-opt_point)**2).sum(-1)
    return test_energy

# def time_param_curve()

class BatchRepulseCircle:
    def __init__(self, origins, radius, batch_dims=[0,1], k=1.0):
        self.B = origins.size(0)
        self.origins = origins
        self.device = self.origins.device
        self.radius = radius
        self.k = k
        self.batch_dims = batch_dims

    def force(self, x):
        # print(x.size())
        x = x.unsqueeze(-2)
        # print(x.size())
        # print(self.origins.size())
        # print(self.B)
        self.device = self.origins.device
        # print(self.batch_dims)
        contact_vector = (x-self.origins)[..., torch.arange(self.B, device=self.device, dtype=torch.long).view(-1,1), self.batch_dims]
        # print('cv', contact_vector.size())
        # print(contact_vector.size())
        dist = contact_vector.norm(dim=-1,keepdim=True)
        # print(dist.size())
        penetration = (self.radius - dist).clamp(0,self.radius)
        # penetration = torch.max(torch.min((self.radii - dist),0),self.origins)
        # print(penetration.size())
        force = self.k*(contact_vector)*(penetration/(dist+1e-6)+1e-16).pow(0.3)
        # print(force.size())
        tot_force = force.sum(dim=1)
        # print(tot_force.size())
        shape = tuple(tot_force.size()[:-1])
        # print(shape)
        return torch.cat((tot_force, torch.zeros(shape+(x.size(-1)-2,), dtype=torch.float, device=x.device)),dim=-1)

class RepulseCircle:
    def __init__(self, origin, radius, k=1.0, dims=[0,1]):
        self.origin = origin
        self.radius = radius
        self.k = k
        self.dims = dims

    def force(self, x):
        contact_vector = (x-self.origin)[..., self.dims]
        # print(contact_vector.size())
        dist = contact_vector.norm(dim=-1,keepdim=True)
        penetration = (self.radius - dist).clamp(0,self.radius)
        force = self.k*(contact_vector)*(penetration/(dist+1e-6)+1e-16).pow(0.3)
        shape = tuple(force.size()[:-1])
        return torch.cat((force, torch.zeros(shape+(x.size(-1)-2,), dtype=torch.float, device=x.device)),dim=-1)

class NavigateGTEnv():
    def __init__(self, batch_size, input_size, batched_func, device, control='force', mass=1.0, sparse_r_step=None, dt=0.05, obstacles_env=False, num_obs=12):
        """
            batch_size B: number of agents to simulate in parallel
            input_size A: number of dimensions for the space the agent is operating in
            batched_func func: a batched function for what the reward should be given a position
            device: torch device
            control: type of control can be {'vel', 'accel', 'force'}
            mass: if control type is force, this is the mass of the agent
        """
        self.device = device
        self.a_size = input_size
        self.s_size = input_size
        self.func = batched_func
        self.control = control
        self.mass = mass
        self.state = None
        self.dt = dt
        self.reset_state(batch_size)
        self.obstacles_env = obstacles_env
        self.num_obs = num_obs

        self.primary_axis = torch.ones(self.s_size)
        # self.primary_axis = self.primary_axis/self.primary_axis.norm()

        self.opt_point = torch.tensor(self.primary_axis,
                    requires_grad=False,
                    device=device)

        #TODO: hack for now
        self.func = get_test_energy(self.opt_point)

        if obstacles_env:
            # self.obstacles = []
            # obstacle_list = [((0.5,0.5),0.06), ((0.3,0.3),0.08), ((0.05,0.2),0.08), ((0.2,0.05),0.1), ((0.5,0.25),0.06), ((0.25,0.5),0.06), ((0.8,0.25),0.1), ((0.25,0.8),0.1), ((0.5,0.75),0.1), ((0.75,0.5),0.1), ((0.5,-0.1),0.15), ((-0.1,0.5),0.15)]
            origin_list = []
            num_obs = self.num_obs
            print(num_obs)
            density = 0.8
            radius = density/num_obs
            for x_pos in np.linspace(-0.7,1.3,num_obs):
                for y_pos in np.linspace(-0.7,1.3,num_obs):
                    x_pos = x_pos + np.random.uniform(-0.1/num_obs, 0.1/num_obs)
                    y_pos = y_pos + np.random.uniform(-0.1/num_obs, 0.1/num_obs)
                    origin_list.append([x_pos, y_pos]+[0]*(self.s_size-2))
            circ_origins = torch.tensor(origin_list, device=self.device)

            # for (point, rad) in obstacle_list:
            #     # Pad it out to the right dimensionality
            #     circ_origin = torch.tensor(point+(0.,)*(self.s_size-2),
            #             requires_grad=False,
            #             device=device)
            #     self.obstacles.append(RepulseCircle(circ_origin, rad, 100.0))
            self.obstacle = BatchRepulseCircle(circ_origins, radius, batch_dims=torch.tensor([0,1], dtype=torch.long, device=self.device).view(1,2).expand(circ_origins.size(0),2), k=300.0)
        else:
            circ_origin = torch.tensor(self.primary_axis*0.5,
                requires_grad=False,
                device=device)
            self.obstacle = RepulseCircle(circ_origin, 0.5, k=40.0)

        # Step on which sparse reward is given (if None dense reward given at each time step)
        self.sparse_r_step=sparse_r_step

    def reset_state(self, batch_size):
        """
            Starts new "episode" for all the agents
            resets the state of the environment to the initial state
            batch_size B: sets the batch_size for this run
        """
        self.t_step = 0
        self.B = batch_size
        if(self.state is None or batch_size != self.state[0].size(0)):
            self.pos = torch.tensor(torch.zeros((batch_size, self.s_size), dtype=torch.float), device=self.device, requires_grad=False)
            self.vel = torch.tensor(torch.zeros((batch_size, self.s_size), dtype=torch.float), device=self.device, requires_grad=False)
            self.state = [self.pos, self.vel]
            # self.done = torch.tensor(torch.zeros((batch_size), dtype=torch.float), requires_grad=False)
        # Detach from graph
        self.state[0] = self.state[0].detach()
        self.state[1] = self.state[1].detach()
        # Reset to zero
        self.state[0].fill_(0)
        self.state[1].fill_(0)

    def rollout(self, actions, return_traj=False):
        # Uncoditional action sequence rollout
        # actions: shape: TxBxA (time, batch, action)
        assert actions.dim() == 3
        assert actions.size(1) == self.B, "{}, {}".format(actions.size(1), self.B)
        assert actions.size(2) == self.a_size
        T = actions.size(0)
        rs = []
        ss = []

        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        for i in range(T):
            _, r, done = self.step(actions[i])
            rs.append(r)
            ss.append(self.state)
            total_r = total_r + r
            if(done):
                break
        if return_traj:
            return rs, ss
        else:
            return total_r

    def step(self, action):
        self.state = self.sim(self.state, action)
        o = self.calc_obs(self.state)
        r = self.calc_reward(self.state, action)
        return o, r, False

    def sim(self, state, action):
        # Symplectic euler
        next_state = [None, None]
        # Velocity control
        if self.control == 'vel':
            next_state[1] = nn.Tanh()(action)
        elif self.control == 'accel':
            next_state[1] = state[1] + self.dt * nn.Tanh()(action)
        elif self.control == 'force':
            if self.obstacles_env:
                # fext = self.obstacles[0].force(state[0])
                # for i in range(1,len(self.obstacles)):
                #     fext = fext + self.obstacles[i].force(state[0])
                fext = self.obstacle.force(state[0])
            else:
                fext = self.obstacle.force(state[0])
            next_state[1] = state[1] + self.dt * (nn.Tanh()(action) + fext)/self.mass
        else:
            raise NotImplementedError()
        next_state[0] = state[0] + self.dt * next_state[1]
        self.t_step += 1
        return next_state

    def calc_obs(self, state):
        return None

    def calc_reward(self, state, action):
        if self.sparse_r_step is not None:
            if self.sparse_r_step == self.t_step:
                return -self.func(state[0])
            else:
                return 0

        return -self.func(state[0])

    # def visualize_2d(self, )
    @staticmethod
    def repulsive_circle_force(x, origin, radius, k=1.0):
        contact_vector = (x-origin)
        # print(contact_vector.size())
        dist = contact_vector.norm(dim=-1,keepdim=True)
        penetration = (radius - dist).clamp(0,radius)
        force = k*(contact_vector)*(penetration/(dist+1e-6))
        return force

    def draw_env_2d_proj(self, ax):
        # Project onto principal dimension
        N = 30
        x = torch.linspace(-0.5,1.5,N)
        y = torch.linspace(-0.5,1.5,N)
        X, Y = torch.meshgrid(x,y)
        pos_grid = torch.stack((X,Y),dim=-1)
        shape = tuple(pos_grid.size()[:-1])
        pos_grid = torch.cat((pos_grid, torch.ones(shape+(self.s_size-2,))), dim=-1)
        energies = self.func(pos_grid.reshape(-1,self.s_size).to(self.device))
        ax.pcolormesh(X.numpy(), Y.numpy(), -energies.reshape(N,N).cpu().numpy(), cmap="coolwarm")
        ax.contour(X.numpy(), Y.numpy(), -energies.reshape(N,N).cpu().numpy(), cmap="seismic")

        # Draw repulsive circle

        if self.obstacles_env:
            for i in range(self.obstacle.origins.size(0)):
                circle = plt.Circle(self.obstacle.origins[i].cpu().numpy().squeeze(), self.obstacle.radius, fill=False, edgecolor='k')
                ax.add_artist(circle)

            # for i in range(len(self.obstacles)):
            #     circle = plt.Circle(self.obstacles[i].origin.cpu().numpy().squeeze(), self.obstacles[i].radius, fill=False, edgecolor='k')
            #     ax.add_artist(circle)
        else:
            circle = plt.Circle(self.obstacle.origin.cpu().numpy().squeeze(), self.obstacle.radius, fill=False, edgecolor='lightgreen')
            ax.add_artist(circle)

    def draw_traj_2d_proj(self, ax, states):
        # Project onto principal dimension
        ps = [s[0].cpu().numpy() for s in states]
        ps = np.array(ps)
        ps = ps.squeeze()
        ax.plot(ps[:, 0], ps[:, 1],'ko-',linewidth=0.5,markersize=1)


class FuncMinGTEnv():
    def __init__(self, batch_size, input_size, batched_func):
        self.a_size = input_size
        self.func = batched_func
        self.state = None
        self.reset_state(batch_size)

    def reset_state(self, batch_size):
        self.B = batch_size

    def rollout(self, actions):
        # Uncoditional action sequence rollout
        # TxBxA
        T = actions.size(0)
        total_r = torch.zeros(self.B, requires_grad=True, device=actions.device)
        for i in range(T):
            _, r, done = self.step(actions[i])
            total_r = total_r + r
            if(done):
                break
        return total_r

    def step(self, action):
        self.state = self.sim(self.state, action)
        o = self.calc_obs(self.state)
        r = self.calc_reward(self.state, action)
        # always done after first step
        return o, r, True

    def sim(self, state, action):
        return state

    def calc_obs(self, state):
        return None

    def calc_reward(self, state, action):
        return -self.func(action)



# class PMEnv():
#     def __init__():
#         dt = 0.1
#         max_v = 2.0
#         max_a = 1.0
#         start_p = torch.tensor([[-1.0,0.0]], requires_grad=False)
#         start_v = torch.tensor([[0.0,0.0]], requires_grad=False)
#         cur_p = None
#         cur_v = None

#     def reset(batch_size):
#         cur_p = (torch.start_p.clone())


#     def batch_step(actions):
#         pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    N = 30
    x = torch.linspace(-1,1,N)
    y = torch.linspace(-1,1,N)
    X, Y = torch.meshgrid(x,y)
    actions_grid = torch.stack((X,Y),dim=-1)
    print(actions_grid)
    energies = test_energy2d(actions_grid.reshape(-1,2))

    plt.pcolormesh(X.numpy(), Y.numpy(), energies.reshape(N,N).numpy(), cmap="coolwarm")
    plt.contour(X.numpy(), Y.numpy(), energies.reshape(N,N).numpy(), cmap="seismic")
    plt.show()

