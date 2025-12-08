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
            
    def control_fun(self, _t, _q, _q_dot, _all_tg, _pc, _obs_props=None):
        
        # print("Control Function")
        # print("_q.shape = {}".format(_q.shape))
        # print("_q_dot.shape = {}".format(_q_dot.shape))
        # print("len(_all_tg) = {}".format(len(_all_tg)))
        # print("len(_obs_props) = {}".format(len(_obs_props)))
        # print("_t = {}".format(_t))

        n = self.param_n_robots
        #Create the objective function
        H = 2*np.identity(3*n)
        f = np.zeros((3*n,1))
        
        error = 0
        
        for i in range(n):
            qi = _q[3*i:3*(i+1),:]
            qdoti = _q_dot[3*i:3*(i+1),:]
            #f[3*i:3*i+3,:] = ( 2*self.param_k*qdoti+ (self.param_k*self.param_k)*(qi-_all_tg[i]) )
            curr_error = np.linalg.norm(qi-_all_tg[i])
            dist_max = 2.0
            new_k = self.param_k_min + (self.param_k_max - self.param_k_min)*curr_error/(dist_max - self.param_dist_interm)
            f[3*i:3*i+3,:] = ( 2*new_k*qdoti+ (new_k*new_k)*(qi-_all_tg[i]) )
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
                # print("_q_dot.shape = {}".format(_q_dot.shape))
                dist = self.dfun(_q, i, j)
                # print("dist.shape = {}".format(dist.shape))
                jac_dist = self.jac_dfun(_q, i, j)
                # print("jac_dist.shape = {}".format(jac_dist.shape))
                dist_dot = (self.dfun(_q+self.dt*_q_dot, i, j)-self.dfun(_q-self.dt*_q_dot, i, j))/(2*self.dt)
                jac_dfun = self.jac_dfun(_q+self.dt*_q_dot, i, j)
                # print("jac_dfun.shape = {}".format(jac_dfun.shape))
                dist_hess = ( (self.jac_dfun(_q+self.dt*_q_dot, i, j)-self.jac_dfun(_q-self.dt*_q_dot, i, j))/(2*self.dt))*_q_dot

                
                #Implement extra safety
                dist_hess = min(dist_hess, 0)
                
                A = np.vstack((A, jac_dist))
                b = np.vstack((b, -2*self.param_eta*dist_dot - (self.param_eta*self.param_eta)*dist-dist_hess ) )
                
                min_dist_agents = min(dist, min_dist_agents)
                
        #Create the constraints with collisions with the environment
        for i in range(n):
                dist = self.dofun(_q, i, _pc)
                jac_dist = self.jac_dofun(_q, i, _pc)
                dist_dot = (self.dofun(_q+self.dt*_q_dot, i, _pc)-self.dofun(_q-self.dt*_q_dot, i, _pc))/(2*self. dt)
                dist_hess = ( (self.jac_dofun(_q+self.dt*_q_dot, i, _pc)-self.jac_dofun(_q-self.dt*_q_dot, i, _pc))/(2*self.dt))*_q_dot
                
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
        b =   np.vstack((b, -2*self.param_eta_v*(_q_dot+self.param_v_max*onev), -2*self.param_eta_v*(self.param_v_max*onev-_q_dot) ))
        
        
        for i in range(n):
            jac_dist = np.zeros((1,3*n))
            jac_dist[0,3*i+2] = -1.0
            
            A =   np.vstack((A, jac_dist))
            b =   np.vstack((b, -2*self.param_eta*(-_q_dot[3*i+2,-1])-(self.param_eta*self.param_eta)*( self.param_zmax+0.2 - _q[3*i+2,-1] ) )) 
            
            jac_dist[0,3*i+2] = 1.0
            
            A =   np.vstack((A, jac_dist))
            b =   np.vstack((b, -2*self.param_eta*(_q_dot[3*i+2,-1])-(self.param_eta*self.param_eta)*(_q[3*i+2,-1]-self.param_radius) )) 
        
        # Add constrainsts to moving obstacle
        if _obs_props is not None:
            
            # For each drone
            for i in range(n):
                final_A = np.zeros((1,3*n))
                final_b = np.zeros((1,1))
                # For each obstacle
                for obs_props in _obs_props:
                    eta = self.param_eta
                    _obs_pos=obs_props['position'](_t)
                    _obs_vel=obs_props['velocity'](_t)
                    _obs_acc=obs_props['acceleration'](_t)
                    _obs_radius=obs_props['radius']

                    qi = _q[3*i:3*(i+1),:]
                    qdoti = _q_dot[3*i:3*(i+1),:]
                    obs_pos = _obs_pos[:,:]
                    obs_vel = _obs_vel[:,:]
                    obs_acc = _obs_acc[:,:]
                    obs_radius = _obs_radius

                    # print("qi.shape = {}".format(qi.shape))
                    # print("qi = {}".format(qi))
                    # print("qdoti.shape = {}".format(qdoti.shape))
                    # print("qdoti = {}".format(qdoti))
                    # print("obs_pos.shape = {}".format(obs_pos.shape))
                    # print("obs_pos = {}".format(obs_pos))
                    # print("obs_vel.shape = {}".format(obs_vel.shape))
                    # print("obs_acc.shape = {}".format(obs_acc.shape))
                    # print("obs_radius = {}".format(obs_radius))
                    # print("self.param_delta = {}".format(self.param_delta))
                        
                    qi_err = qi - obs_pos
                    # print("qi_err.shape = {}".format(qi_err.shape))

                    # Define B
                    B = np.linalg.norm(qi_err)**2 - obs_radius**2 - self.param_delta
                    # print("B.shape = {}".format(B.shape))
                    # print("B = {}".format(B))

                    # First partial derivative of B with respect to time
                    # B_dot = dB/dq * q_dot + dB/dt
                    dB_dq = 2*qi_err.T
                    # print("dB_dq.shape = {}".format(dB_dq.shape))
                    
                    dB_dt = 2*obs_vel.T*(-qi_err) # -qi_err because it's (q_obs - q)
                    # print("dB_dt.shape = {}".format(dB_dt.shape))
                    
                    B_dot = dB_dq*qdoti + dB_dt
                    # print("qdoti.shape = {}".format(qdoti.shape))
                    # print("B_dot.shape = {}".format(B_dot.shape))
                    
                    # Second partial derivative of B with respect to q
                    d2B_dq2 = 2*np.eye(3)
                    # print("d2B_dq2.shape = {}".format(d2B_dq2.shape))
                    
                    # Second partial derivative of B with respect to q then t
                    d2B_dqdt = -2*obs_vel.T
                    # print("d2B_dqdt.shape = {}".format(d2B_dqdt.shape))
                    
                    # Second partial derivate of B with respect to time
                    d2B_dt2 = 2*obs_acc.T*(-qi_err) + 2*obs_vel.T*obs_vel
                    # print("d2B_dt2.shape = {}".format(d2B_dt2.shape))
                    
                    # grad_B_mov_fun = (qi - obs_pos).T/np.linalg.norm(qi - obs_pos)
                    # Final full gradient 
                    # dB/dq * u >= -2*eta_B_dot
                    #grad_B = -2*eta*B_dot - eta**2*B - qdoti.T*d2B_dq2*qdoti - d2B_dqdt*qdoti - d2B_dt2
                    t1 = -2*eta*B_dot
                    # print("t1.shape = {}".format(t1.shape))
                    # print("t1 = {}".format(t1))
                    t2 = -eta**2*B
                    # print("t2.shape = {}".format(t2.shape))
                    # print("t2 = {}".format(t2))
                    t3 = -qdoti.T*d2B_dq2*qdoti
                    # print("t3.shape = {}".format(t3.shape))
                    # print("t3 = {}".format(t3))
                    t4 = -d2B_dqdt*qdoti
                    # print("t4.shape = {}".format(t4.shape))
                    # print("t4 = {}".format(t4))
                    t5 = -d2B_dt2
                    # print("t5.shape = {}".format(t5.shape))
                    # print("t5 = {}".format(t5))
                    grad_B = t1 + t2 + t3 + t4 + t5
                    # print("grad_B.shape = {}".format(grad_B.shape))
                    # print("grad_B = {}".format(grad_B))
                    # print("final_A.shape = {}".format(final_A.shape))
                    # print("dB_dq.T.shape = {}".format(dB_dq.T.shape))
                    # print("final_b.shape = {}".format(final_b.shape))
                    # print("grad_B.shape = {}".format(grad_B.shape))

                    final_A[0,3*i:3*(i+1)] = dB_dq.T.flatten()
                    final_b[0,0] += grad_B
            
                # Add all constraints at the end
                A = np.vstack( (A, final_A) )
                b = np.vstack( (b, final_b) )
            

        try:                 
            return ub.Utils.solve_qp(H,f,A,b), min_dist_agents, min_dist_obs
        except:
            #If it is not able to find a solution, brake for safety.
            norm_qdot = np.linalg.norm(_q_dot)
            return -min(self.param_a_max,norm_qdot)*_q_dot/(norm_qdot+1e-6), min_dist_agents, min_dist_obs
                
        
    
    ######################   