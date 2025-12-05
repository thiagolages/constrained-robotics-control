import numpy as np
import uaibot as ub

class DroneControl():
    def __init__(self, *args, **kwargs):
        self.param_k = kwargs.get('param_k')
        self.param_eta = kwargs.get('param_eta')
        self.param_eta_v = kwargs.get('param_eta_v')
        self.param_v_max = kwargs.get('param_v_max')
        self.param_a_max = kwargs.get('param_a_max')
        self.param_radius = kwargs.get('param_radius')
        self.param_height = kwargs.get('param_height')
        self.param_n_robots = kwargs.get('param_n_robots')
        self.param_dist_interm = kwargs.get('param_dist_interm')
        self.param_dist_final = kwargs.get('param_dist_final')
        self.param_h_dist = kwargs.get('param_h_dist')
        self.param_eps_dist = kwargs.get('param_eps_dist')
        self.param_delta = kwargs.get('param_delta')
        self.param_min_dist_drone_tg = kwargs.get('param_min_dist_drone_tg')
        self.param_min_dist_tg = kwargs.get('param_min_dist_tg')
        self.param_t_max = kwargs.get('param_t_max')
        self.param_xmin = kwargs.get('param_xmin')
        self.param_xmax = kwargs.get('param_xmax')
        self.param_ymin = kwargs.get('param_ymin')
        self.param_ymax = kwargs.get('param_ymax')
        self.param_zmin = kwargs.get('param_zmin')
        self.param_zmax = kwargs.get('param_zmax')
        self.param_k_min = kwargs.get('param_k_min')
        self.param_k_max = kwargs.get('param_k_max')
        self.dt = kwargs.get('dt')

    #################################################

    def dfun(self, _q, _i, _j):
        
        qi = _q[3*_i:3*(_i+1),:]
        qj = _q[3*_j:3*(_j+1),:]

        return np.linalg.norm(qi-qj)-(2*self.param_radius)

    def jac_dfun(self, _q, _i, _j):
        
        n = self.param_n_robots
        jac_dist = np.zeros((1,3*n))
        # print("jac_dist.shape before = {}".format(jac_dist.shape))
        
        qi = _q[3*_i:3*(_i+1),:]
        qj = _q[3*_j:3*(_j+1),:]
        
        norm_ij = np.linalg.norm(qi-qj)
        
        # Convert matrix to array and flatten to 1D before assignment
        # .A1 converts matrix to 1D array, or use .A.flatten() or np.array(...).flatten()
        jac_dist[0,3*_i:3*(_i+1)] = np.array((qi-qj).T/norm_ij).flatten()
        jac_dist[0,3*_j:3*(_j+1)] = np.array((qj-qi).T/norm_ij).flatten()

        # print("jac_dist.shape after = {}".format(jac_dist.shape))
        
        return jac_dist

    def dofun(self, _q, _i, _pc):
        qi = _q[3*_i:3*(_i+1),:]
        
        ball_i = ub.Ball(htm=ub.Utils.trn(qi), radius=self.param_radius)
        
        return ball_i.compute_dist(obj = _pc, h=self.param_h_dist, eps=self.param_eps_dist)[2]-self.param_delta

    def dofun_real(self, _q, _i, _pc):
        qi = _q[3*_i:3*(_i+1),:]
        
        ball_i = ub.Ball(htm=ub.Utils.trn(qi), radius=self.param_radius)
        
        return ball_i.compute_dist(obj = _pc)[2]

    def jac_dofun(self, _q, _i, _pc):
        
        qi = _q[3*_i:3*(_i+1),:]
        ball_i =  ub.Ball(htm=ub.Utils.trn(qi), radius=self.param_radius)
        pball, ppc, dist, _ = ball_i.compute_dist(obj = _pc, h=self.param_h_dist, eps=self.param_eps_dist)
        

        n = self.param_n_robots
        jac_dist = np.zeros((1,3*n))
        # Convert matrix to array and flatten to 1D before assignment
        jac_dist[0,3*_i:3*(_i+1)] = np.array((pball-ppc).T/(1e-5+dist+0.03)).flatten()

        return jac_dist


    ############################
            
    def control_fun(self, _q, _dotq, _all_tg, _pc, _mov_obs_pos=None, _mov_obs_vel=None, _mov_obs_radius=0.5):
        
        n = self.param_n_robots
        
        #Create the objective function
        H = 2*np.identity(3*n)
        f = np.zeros((3*n,1))
        
        error = 0
        
        for i in range(n):
            qi = _q[3*i:3*(i+1),:]
            dotqi = _dotq[3*i:3*(i+1),:]
            #f[3*i:3*i+3,:] = ( 2*self.param_k*dotqi+ (self.param_k*self.param_k)*(qi-_all_tg[i]) )
            curr_error = np.linalg.norm(qi-_all_tg[i])
            dist_max = 2.0
            new_k = self.param_k_min + (self.param_k_max - self.param_k_min)*curr_error/(dist_max - self.param_dist_interm)
            f[3*i:3*i+3,:] = ( 2*new_k*dotqi+ (new_k*new_k)*(qi-_all_tg[i]) )
            error+= np.linalg.norm(qi-_all_tg[i])
            
        f = 2*f
        
        
        #Create the constraints
        A = np.matrix(np.zeros((0,3*n)))
        b = np.matrix(np.zeros((0,1)))
        
        min_dist_agents = 1e6
        min_dist_obs = 1e6
        

        #Create the inter-agent collision constraints
        for i in range(n):
            for j in range(0,i):
                
                
                # print("_q.shape = {}".format(_q.shape))
                # print("_dotq.shape = {}".format(_dotq.shape))
                dist = self.dfun(_q, i, j)
                # print("dist.shape = {}".format(dist.shape))
                jac_dist = self.jac_dfun(_q, i, j)
                # print("jac_dist.shape = {}".format(jac_dist.shape))
                dist_dot = (self.dfun(_q+self.dt*_dotq, i, j)-self.dfun(_q-self.dt*_dotq, i, j))/(2*self.dt)
                jac_dfun = self.jac_dfun(_q+self.dt*_dotq, i, j)
                # print("jac_dfun.shape = {}".format(jac_dfun.shape))
                dist_hess = ( (self.jac_dfun(_q+self.dt*_dotq, i, j)-self.jac_dfun(_q-self.dt*_dotq, i, j))/(2*self.dt))*_dotq

                
                #Implement extra safety
                dist_hess = min(dist_hess, 0)
                
                A = np.vstack((A, jac_dist))
                b = np.vstack((b, -2*self.param_eta*dist_dot - (self.param_eta*self.param_eta)*dist-dist_hess ) )
                
                min_dist_agents = min(dist, min_dist_agents)
                
        #Create the constraints with collisions with the environment
        for i in range(n):
                dist = self.dofun(_q, i, _pc)
                jac_dist = self.jac_dofun(_q, i, _pc)
                dist_dot = (self.dofun(_q+self.dt*_dotq, i, _pc)-self.dofun(_q-self.dt*_dotq, i, _pc))/(2*self. dt)
                dist_hess = ( (self.jac_dofun(_q+self.dt*_dotq, i, _pc)-self.jac_dofun(_q-self.dt*_dotq, i, _pc))/(2*self.dt))*_dotq
                
                #Implement extra safety
                dist_hess = min(dist_hess, 0)
                
                A = np.vstack((A, jac_dist))
                b = np.vstack((b, -2*self.param_eta*dist_dot - (self.param_eta*self.param_eta)*dist-dist_hess ) )
                
                min_dist_obs = min(self.dofun_real(_q, i, _pc), min_dist_obs)


        idm = np.identity(3*n)
        onev = np.ones((3*n,1))
                    
        A =   np.vstack((A, idm, -idm)) 
        b =   np.vstack((b, -self.param_a_max*onev, -self.param_a_max*onev)) 
            
        A =   np.vstack((A, idm, -idm)) 
        b =   np.vstack((b, -2*self.param_eta_v*(_dotq+self.param_v_max*onev), -2*self.param_eta_v*(self.param_v_max*onev-_dotq) ))
        
        
        for i in range(n):
            jac_dist = np.zeros((1,3*n))
            jac_dist[0,3*i+2] = -1.0
            
            A =   np.vstack((A, jac_dist))
            b =   np.vstack((b, -2*self.param_eta*(-_dotq[3*i+2,-1])-(self.param_eta*self.param_eta)*( self.param_zmax+0.2 - _q[3*i+2,-1] ) )) 
            
            jac_dist[0,3*i+2] = 1.0
            
            A =   np.vstack((A, jac_dist))
            b =   np.vstack((b, -2*self.param_eta*(_dotq[3*i+2,-1])-(self.param_eta*self.param_eta)*(_q[3*i+2,-1]-self.param_radius) )) 
        
        # Add constrainsts to moving obstacle
        if _mov_obs_pos is not None and _mov_obs_vel is not None:
            # for drone_idx in range(n):
            #     i = 3*drone_idx
            #     print(f"i = ")
            #     print(f"_q.shape = {_q.shape}")
            #     print(f"_q[i:i+3] = {_q[i:i+3]}")
            #     print(f"_q[i:i+3].shape = {_q[i:i+3].shape}")
            #     print(f"_mov_obs_pos = {_mov_obs_pos}")
            #     print(f"_mov_obs_pos.shape = {_mov_obs_pos.shape}")
            #     print(f"_mov_obs_vel = {_mov_obs_vel}")
            #     print(f"_mov_obs_vel.shape = {_mov_obs_vel.shape}")
            #     curr_q = _q[i:i+3]
            #     print(f"curr_q = {curr_q}")
            #     print(f"curr_q.shape = {curr_q.shape}")
            #     _mov_obs_pos = np.matrix(_mov_obs_pos).T
            #     B_mov_fun = np.linalg.norm(curr_q - _mov_obs_pos) - self.param_radius - self.param_delta
            #     print(f"B_mov_fun = {B_mov_fun}")
            #     print(f"B_mov_fun.shape = {B_mov_fun.shape}")
            #     grad_B_mov_fun = (curr_q - _mov_obs_pos).T/np.linalg.norm(curr_q - _mov_obs_pos)
            #     print(f"grad_B_mov_fun = {grad_B_mov_fun}")
            #     print(f"grad_B_mov_fun.shape = {grad_B_mov_fun.shape}")
            # Repeat the _mov_obs_pos as a (3*n,1) matrix by stacking n times vertically
            _mov_obs_pos = np.matrix(_mov_obs_pos).T
            _mov_obs_pos = np.vstack([_mov_obs_pos for _ in range(n)]) # make this 3*n x 1
            B_mov_fun = np.linalg.norm(_q - _mov_obs_pos) - _mov_obs_radius - self.param_delta
            # print(f"B_mov_fun = {B_mov_fun}")
            # print(f"B_mov_fun.shape = {B_mov_fun.shape}")
            grad_B_mov_fun = (_q - _mov_obs_pos).T/np.linalg.norm(_q - _mov_obs_pos)
            # print(f"grad_B_mov_fun = {grad_B_mov_fun}")
            # print(f"grad_B_mov_fun.shape = {grad_B_mov_fun.shape}")
            # vstack after all drones are iterated over
            A = np.vstack( (A, grad_B_mov_fun) )
            b = np.vstack( (b, -self.param_eta * B_mov_fun) )
            # print(f"A = {A}")
            # print(f"A.shape = {A.shape}")
            # print(f"b = {b}")
            # print(f"b.shape = {b.shape}")

        try:                 
            return ub.Utils.solve_qp(H,f,A,b), min_dist_agents, min_dist_obs
        except:
            #If it is not able to find a solution, brake for safety.
            norm_dotq = np.linalg.norm(_dotq)
            return -min(self.param_a_max,norm_dotq)*_dotq/(norm_dotq+1e-6), min_dist_agents, min_dist_obs
                
        
    
    ######################   