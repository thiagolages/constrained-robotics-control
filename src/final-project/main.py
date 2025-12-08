import os
import sys
import numpy as np
import uaibot as ub
import matplotlib.pyplot as plt
from DroneUtils import *
from DroneControl import DroneControl
from DroneFormation import DroneFormation

###################################
class DroneShow():
    def __init__(self):
        # All params

        # Control params
        self.param_k = 4.0 #1.4
        self.param_k_min = 4.0 #1.4
        self.param_k_max = 10*self.param_k_min #1.4
        self.param_eta = 0.4 #0.6
        self.param_eta_v = 0.1
        self.param_v_max = 40.0 #0.4
        self.param_a_max = 50.0 #0.5
        self.param_radius = 0.1 # drone sphere radius,for obs avoidance
        self.param_n_robots = 10
        self.param_dist_interm = 0.3
        self.param_dist_final = 0.05
        self.param_h_dist = 1e-6
        self.param_eps_dist = 1e-6
        self.param_delta = 0.5 # safety margin
        self.param_min_dist_drone_tg = 0.25 # minimum distance between drone and target point
        self.param_min_dist_tg = 0.25 # minimum distance between target points
        
        # Stage params
        self.stage_size = 5.0
        self.param_t_max = 800
        self.param_max_pc_points_per_frame = 100

        # Target points params
        self.param_xmin = -100
        self.param_xmax = 100
        self.param_ymin = -100
        self.param_ymax = 100
        self.param_zmin = -100.0
        self.param_zmax = 100.0
        self.dt = 0.02 # 0.02
        self.drone_sphere_opacity = 0.7        

        # Drone show params
        self.drone_show_z_height = 5.0

        # Dynamic Obstacle params
        self.mov_obs_radius = 0.5 # radius of moving obstacle
        self.mov_obs2_radius = 0.5 # radius of moving obstacle
        # self.mov_obs_max_obs_vel = 10.0 # m/s
        # self.mov_obs_max_obs_acc = 10.0 # m/s²

        self.obs_pos_lims = [
            [-self.stage_size, self.stage_size],
            [-self.stage_size, self.stage_size],
            [self.drone_show_z_height - 1.0, self.drone_show_z_height + 1.0],
        ]
        self.mov_obs_amplitude = [self.stage_size, self.stage_size, self.stage_size*0.2]
        ###################################################################
        # Attention ! If any 'w' is changed for position, their derivatives
        # (vel, acc) have to be changed as well !
        ###################################################################
        self.mov_obs_speed_multiplier = 0.1 # speed multiplier for oscilating movements
        self.mov_obs_w = (np.deg2rad(np.array([5, 30, 60]))*2*np.pi*self.mov_obs_speed_multiplier).tolist() # w for oscilating movements
        self.mov1_obs_props = {
            'position': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        A[0]*np.cos(w[0]*t), 
                        A[1]*np.cos(w[1]*t), 
                        A[2]*np.cos(w[2]*t) + self.drone_show_z_height
                    ]).reshape(3,1),
            'velocity': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        -w[0]*A[0]*np.sin(w[0]*t), 
                        -w[1]*A[1]*np.sin(w[1]*t), 
                        -w[2]*A[2]*np.sin(w[2]*t)
                    ]).reshape(3,1),
            
            'acceleration': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        -w[0]**2*A[0]*np.cos(w[0]*t), 
                        -w[1]**2*A[1]*np.cos(w[1]*t), 
                        -w[2]**2*A[2]*np.cos(w[2]*t)
                    ]).reshape(3,1),
            
            'radius': self.mov_obs_radius,
        }

        self.mov_obs2_props = {
            'position': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        A[0]*np.sin(w[0]*t), 
                        A[1]*np.sin(w[1]*t), 
                        A[2]*np.sin(w[2]*t) + self.drone_show_z_height
                    ]).reshape(3,1),
            'velocity': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        w[0]*A[0]*np.cos(w[0]*t), 
                        w[1]*A[1]*np.cos(w[1]*t), 
                        w[2]*A[2]*np.cos(w[2]*t)
                    ]).reshape(3,1),
            
            'acceleration': 
                lambda t, 
                    A=self.mov_obs_amplitude,
                    w=self.mov_obs_w:
                    np.matrix([
                        -w[0]**2*A[0]*np.sin(w[0]*t), 
                        -w[1]**2*A[1]*np.sin(w[1]*t), 
                        -w[2]**2*A[2]*np.sin(w[2]*t)
                    ]).reshape(3,1),
            
            'radius': self.mov_obs2_radius,
        }

        self.mov_obs_props = [self.mov1_obs_props, self.mov_obs2_props]

        self.param_dict = dict(
            param_k = self.param_k,
            param_eta = self.param_eta,
            param_eta_v = self.param_eta_v,
            param_v_max = self.param_v_max,
            param_a_max = self.param_a_max,
            param_radius = self.param_radius,
            param_n_robots = self.param_n_robots,
            param_dist_interm = self.param_dist_interm,
            param_dist_final = self.param_dist_final,
            param_h_dist = self.param_h_dist,
            param_eps_dist = self.param_eps_dist,
            param_delta = self.param_delta,
            param_min_dist_drone_tg = self.param_min_dist_drone_tg,
            param_min_dist_tg = self.param_min_dist_tg,
            param_t_max = self.param_t_max,
            param_xmin = self.param_xmin,
            param_xmax = self.param_xmax,
            param_ymin = self.param_ymin,
            param_ymax = self.param_ymax,
            param_zmin = self.param_zmin,
            param_zmax = self.param_zmax,
            param_k_min = self.param_k_min,
            param_k_max = self.param_k_max,
            dt = self.dt,
        )
        self.drones = []
        self.obstacles = []
        self.targets = []
        self.path_points = []
        self.hist_dotq = []
        self.hist_ddotq = []
        self.hist_t = []
        self.hist_dist_agent = []
        self.hist_dist_obs = []
        self.setup()

    def setup(self):
        self.control = DroneControl(**self.param_dict)
        self.formation = DroneFormation(self.param_n_robots, "circle")
        self.setup_drones()
        self.setup_obstacles()
        self.setup_points()
        self.setup_initial_poses()
        # self.setup_target_points()
        self.setup_initial_path()
        self.create_simulation()
        self.setup_initial_state()


    def setup_initial_state(self):
        # Set current targets
        self.current_tg = [self.path_points[i][0] for i in range(self.param_n_robots)]
        self.init_index = [0 for i in range(self.param_n_robots)]
        self.finished = [False for i in range(self.param_n_robots)]
        self.cont = True
        self.min_dist_agents = 1e6
        self.min_dist_obs = 1e6

    def create_simulation(self):
        # sim = ub.Simulation.create_sim_mountain(drones, light_intensity=1.5)
        # self.sim = ub.Simulation([self.drones])
        self.sim = ub.Simulation.create_sim_sky([self.drones])
        # sim.add(pc)
        self.sim.add(self.all_obstacles)
        # self.sim.add(self.all_tg_box)
        self.sim.add(self.master_path_pc)
        self.sim.add(self.mov_obs)
        self.sim.add(self.mov_obs2)
        self.sim.set_parameters(pixel_ratio=0.9)
        # self.sim.set_parameters(camera_start_pose=[ 4.2967, 2.4381, 3.5080, 3.6016, 2.0036, 2.9353, 1.0000])
        self.sim.set_parameters(camera_start_pose=[ 0.0, -10, 15.0, 0.0, 0.0, 0.0, 1.0000])
        self.sim.set_parameters(width=550,height=500)

    def setup_initial_poses(self):
        #Sample initial poses for the drones
        self.start_points = [np.matrix([0,0,0]).T for i in range(self.param_n_robots)]

        for i in range(self.param_n_robots):
            
            cont = True
            print("Finding initial pose for drone "+str(i+1)+" ... ")
            while cont:
                htm=ub.Utils.htm_rand([-self.stage_size, -self.stage_size, 0],[self.stage_size, self.stage_size,0],0)
                self.drones[i].add_ani_frame(0,htm)
                self.start_points[i] = htm[0:3,-1]
                
                #Check if this is ok
                self.ball_drone_i = self.drones[i].list_of_objects[-1]
                collided = self.ball_drone_i.compute_dist(self.pc)[2]< self.param_radius
                
                j = 0
                while (not collided) and (j<i) :
                    ball_drone_j = self.drones[j].list_of_objects[-1]
                    collided = self.ball_drone_i.compute_dist(ball_drone_j)[2] < self.param_radius
                    j+=1
                cont = collided
        
        # # Arrange drones in a circle formation around the origin on the XY plane at z=0
        # center = np.array([0.0, 0.0])
        # radius = 1.0  # or any spread-out distance you prefer
        # n = self.param_n_robots
        # for i in range(n):
        #     angle = 2 * np.pi * i / n
        #     x = center[0] + radius * np.cos(angle)
        #     y = center[1] + radius * np.sin(angle)
        #     z = 0.0  # All drones at height 0
        #     htm = np.eye(4)
        #     htm[0:3, 3] = [x, y, z]
        #     self.drones[i].add_ani_frame(0, htm)
        #     self.start_points[i] = np.matrix([[x], [y], [z]])
                        
    def setup_target_points(self):
        #Sample target points that are far apart from each other, not coliding
        #and far for the respective drone

        self.all_tg_points = []
        # self.all_tg_box = []

        for i in range(self.param_n_robots):
            
            cont = True
            
            print("Finding target points for drone "+str(i+1)+" ... ")
            
            while cont:            
                tg_point_try=ub.Utils.htm_rand([self.param_xmin,self.param_ymin,self.param_zmin],[self.param_xmax,self.param_ymax,self.param_zmax],0)[0:3,-1]
                #Check if this is ok
                self.ball_drone_i = self.drones[i].list_of_objects[-1]
                
                unfit = self.pc.projection(tg_point_try)[1]<0.3 or self.ball_drone_i.projection(tg_point_try)[1]<self.param_min_dist_drone_tg 
                
                #Check if far away from others points
                j = 0
                while (not unfit) and (j<i) :
                    unfit = np.linalg.norm(self.all_tg_points[j]-tg_point_try)<self.param_min_dist_tg
                    j+=1        
                cont = unfit

            self.all_tg_points.append(tg_point_try)
            # self.all_tg_box.append(ub.Box(htm=ub.Utils.trn(tg_point_try), color=self.drone_colors[i],width=0.03,depth=0.03,height=0.03))                            
        print("Finished!")
        
    def setup_initial_path(self):
        #Use motion planer
        self.path_points = [[] for _ in range(self.param_n_robots)]

        num_points_per_stage = 100
        theta = np.linspace(0, 2 * np.pi, num_points_per_stage) # using 1 just bc radius is 0
        radius = 5.0
        center = [0, 0, self.drone_show_z_height]
        self.master_path = []

        ########### Get out of the cage #################
        print("PHASE 0: Get out of the cage")
        self.master_path.append(np.matrix([0, 0, self.drone_show_z_height]).T) # list of 3 x 1 matrices

        for i, pos in enumerate(self.master_path):
            normal = [0,0,1]
            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Get out of cage #################

        ########### Circular around XY plane while rotating around global Z #################
        print("PHASE 1: Circular around XY plane while rotating around global Z")
        for t in theta:
            x = center[0] + radius * np.cos(t)
            y = center[1] + radius * np.sin(t)
            # print("x = {}, y = {}, self.drone_show_z_height = {} for theta = {}".format(x, y, self.drone_show_z_height, t))
            self.master_path.append(np.matrix([x, y, self.drone_show_z_height]).T) # list of 3 x 1 matrices

        # print("self.master_path = {}".format(self.master_path))
        # self.master_path = np.hstack(self.master_path).T # N x 3
        for i, pos in enumerate(self.master_path):

            # rotation in X
            # normal = [0, 0, 1]
            # Rotate around Z and use x-axis as normal
            normal = np.array((ub.Utils.rotz(i * 3*np.pi / num_points_per_stage))[0:3,0]).reshape(3,).tolist()
            # normal = np.array((ub.Utils.rotx(i * 3*np.pi / num_points_per_stage)*ub.Utils.roty(i * 3*np.pi / num_points_per_stage))[0:3,2]).reshape(3,).tolist()
            # rotation in Y
            # normal = np.array(ub.Utils.roty(i * np.pi / num_points_per_stage)[0:3,2]).reshape(3,).tolist()
            # rotation in Z
            # normal = np.array(ub.Utils.rotz(i * np.pi / num_points_per_stage)[0:3,2]).reshape(3,).tolist()

            # print("normal = {}".format(normal))
            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Circular around XY plane while rotating around global Z #################

        ########### Linear in -X while rotating around global X #################
        print("PHASE 2: Linear in -X while rotating around global X")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos + np.matrix([-2*radius, 0, 0]).T
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            # Rotate around X and use x-axis as normal
            normal = [-1,0,0]
            i_aux = i - initial_master_path_index
            phase = i_aux * np.pi / num_points_per_stage
            # print("phase = {}".format(phase))

            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Linear in -X while rotating around global X #################

        ########### Linear in +X (halfway) while rotating around global X #################
        print("PHASE 3: Linear in +X (halfway) while rotating around global X")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos + np.matrix([+1*radius, 0, 0]).T
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        prev_phase = phase
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            # Rotate around X and use x-axis as normal
            # normal = XXXX # use the same normal as before
            i_aux = i - initial_master_path_index
            phase = -i_aux * np.pi / num_points_per_stage + prev_phase # start using previous phase
            # print("phase = {}".format(phase))

            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Linear in +X (halfway) while rotating around global X #################

        ########### Rotate around +Z so normal faces +Y axis #################
        print("PHASE 4: Rotate around +Z so normal faces +Y axis")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos # same, we're just going to rotate
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        prev_phase = phase
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            i_aux = i - initial_master_path_index
            # Rotate around X and use x-axis as normal
            normal = np.array(ub.Utils.rotz(i_aux * (-np.pi/2) / num_points_per_stage)[0:3,0]).reshape(3,)
            # Get the opposite of the normal
            normal = -normal
            normal = normal.tolist()
            # phase = i * (np.pi/2) / num_points_per_stage
            # print("phase = {}".format(phase))
            phase = prev_phase

            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Rotate around +Z so normal faces +Y axis #################

        ########### Linear in +Y (halfway) while rotating around global Y #################
        print("PHASE 5: Linear in +Y (halfway) while rotating around global Y")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos + np.matrix([0, +1*radius, 0]).T
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            i_aux = i - initial_master_path_index
            # Rotate around X and use x-axis as normal
            normal = [0,1,0]
            phase = i_aux * np.pi / num_points_per_stage
            # print("phase = {}".format(phase))

            relative_positions = (
                self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Linear in +Y (halfway) while rotating around global Y #################

        ########### Linear in -Y while rotating around global Y #################
        print("PHASE 6: Linear in -Y while rotating around global Y")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos + np.matrix([0, -2*radius, 0]).T
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        prev_phase = phase
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            i_aux = i - initial_master_path_index
            # Rotate around X and use x-axis as normal
            normal = [0,1,0]
            phase = -i_aux * np.pi / num_points_per_stage + prev_phase
            # print("phase = {}".format(phase))

            relative_positions = (
                    self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Linear in -Y while rotating around global Y #################

        ########### Linear in Y while rotating around global X #################
        print("PHASE 7: Linear in Y while rotating around global -X")
        initial_pos = self.master_path[-1]
        initial_master_path_index = len(self.master_path)
        end_pos = initial_pos + np.matrix([0, 2*radius, 0]).T
        parameteric_path = np.linspace(0, 1, num_points_per_stage)
        prev_phase = phase
        
        for t in parameteric_path:
            pos = initial_pos + (end_pos - initial_pos) * t
            self.master_path.append(pos)

        for i in range(initial_master_path_index, initial_master_path_index + num_points_per_stage):
            pos = self.master_path[i]
            i_aux = i - initial_master_path_index
            # Rotate around X and use x-axis as normal
            normal = np.array((ub.Utils.rotz(i_aux * (2*np.pi) / num_points_per_stage))[0:3,1]).reshape(3,).tolist()
            phase = -i_aux * np.pi / num_points_per_stage + prev_phase
            # print("phase = {}".format(phase))

            relative_positions = (
                    self.formation.get_relative_positions(self.param_n_robots, normal, phase)
            ) # list of (3x1) matrices, of size n_robots (so n_robots x 3 x 1)
            for drone_idx in range(self.param_n_robots):
                rel_pos = relative_positions[drone_idx] # 3 x 1
                next_wp = pos + rel_pos # 3 x 1
                # print("next_wp = {} for drone_idx = {} and i = {}".format(next_wp, drone_idx, i))
                self.path_points[drone_idx].append(next_wp)
        
        ########### End of Linear in Y while rotating around global Y #################


        self.master_path_pc = ub.PointCloud(points=self.master_path,size=0.3,color=self.pc_color)



    def setup_obstacles(self):
        obs_opacity = 0.9
        # #51A2FF -light blue
        # #BABABA - light gray
        obs_color = "#BABABA"
        self.all_obstacles = []
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0,0,0.5]),width=0.1,depth=0.1,height=0.6,opacity=obs_opacity,color=obs_color)) # obs1
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.6,0.1,0.5]),width=0.2,depth=1.0,height=1.0,opacity=obs_opacity,color=obs_color)) # obs2
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([-1.0,-1.0,0.3]),width=0.4,depth=0.4,height=0.6,opacity=obs_opacity,color=obs_color)) # obs3
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([-1.0,1.0,0.2]),width=0.4,depth=0.4,height=0.4,opacity=obs_opacity,color=obs_color)) # obs4
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.6,-0.9,0.95]),width=0.2,depth=1.0,height=0.1,opacity=obs_opacity,color=obs_color)) # obs5
        # self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.6,-1.4,0.5]),width=0.2,depth=0.2,height=1.0,opacity=obs_opacity,color=obs_color)) # obs6
        
        cs = 0.75 # cube size
        cp = 2.5 # cube position offset
        cube_color = '#9379F2' # purple
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0,0,cs/2]),width=cs, height=cs, depth=cs, opacity=obs_opacity,color=cube_color)) # cube1
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([cp,cp,cs/2]),width=cs, height=cs, depth=cs, opacity=obs_opacity,color=cube_color)) # cube2
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([cp,-cp,cs/2]),width=cs, height=cs, depth=cs, opacity=obs_opacity,color=cube_color)) # cube3
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([-cp,cp,cs/2]),width=cs, height=cs, depth=cs, opacity=obs_opacity,color=cube_color)) # cube4
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([-cp,-cp,cs/2]),width=cs, height=cs, depth=cs, opacity=obs_opacity,color=cube_color)) # cube5
        
        # Walls
        wall_thickness = 0.05
        wall_height = 2.0
        walls_color = '#303030' # dark grey
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([-self.stage_size,0.0,wall_height/2]),width=wall_thickness,depth=2*self.stage_size,height=wall_height,opacity=obs_opacity,color=walls_color)) # wallxp
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([self.stage_size,0.0,wall_height/2]),width=wall_thickness,depth=2*self.stage_size,height=wall_height,opacity=obs_opacity,color=walls_color)) # wallxn
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,self.stage_size,wall_height/2]),width=2*self.stage_size,depth=wall_thickness,height=wall_height,opacity=obs_opacity,color=walls_color)) # wallyp
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,-self.stage_size,wall_height/2]),width=2*self.stage_size,depth=wall_thickness,height=wall_height,opacity=obs_opacity,color=walls_color)) # wallyn

        # Aligned with X
        wallz_color = '#F27979' # light red
        wallz_depth = 0.25
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,0.0,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz1
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,-self.stage_size/2,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz2
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,-self.stage_size,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz3
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,self.stage_size/2,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz4
        self.all_obstacles.append(ub.Box(htm=ub.Utils.trn([0.0,self.stage_size,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz5
        
        # Aligned with Y
        self.all_obstacles.append(ub.Box(htm=ub.Utils.rotz(np.pi/2)*ub.Utils.trn([0.0,0.0,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz1
        self.all_obstacles.append(ub.Box(htm=ub.Utils.rotz(np.pi/2)*ub.Utils.trn([0.0,-self.stage_size/2,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz2
        self.all_obstacles.append(ub.Box(htm=ub.Utils.rotz(np.pi/2)*ub.Utils.trn([0.0,-self.stage_size,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz3
        self.all_obstacles.append(ub.Box(htm=ub.Utils.rotz(np.pi/2)*ub.Utils.trn([0.0,self.stage_size/2,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz4
        self.all_obstacles.append(ub.Box(htm=ub.Utils.rotz(np.pi/2)*ub.Utils.trn([0.0,self.stage_size,wall_height]),width=2*self.stage_size,depth=wallz_depth,height=wall_thickness,opacity=obs_opacity,color=wallz_color)) # wallz5

        # Dynamic Obstacle
        self.mov_obs = ub.Ball(color="red", radius=self.mov_obs_radius, opacity=0.9)
        self.mov_obs_htm = lambda t: ub.Utils.trn(self.mov1_obs_props['position'](t))

        self.mov_obs2 = ub.Ball(color="purple", radius=self.mov_obs2_radius, opacity=0.9)
        self.mov_obs2_htm = lambda t: ub.Utils.trn(self.mov_obs2_props['position'](t))

        # self.all_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, wallxp, wallxn, wallyp, wallyn]

    def setup_points(self):
        self.all_points = []
        for obs in self.all_obstacles:
            self.all_points+=  [np.matrix(p).T for p in obs.to_point_cloud(disc=0.06).points.T]
        self.pc = ub.PointCloud(points=self.all_points,size=0.03,color='#005500')

    def setup_drones(self):
        self.drone_model = ub.Model3D(
            'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/CrazyFlie/crazyflie.obj',
            scale=1, mesh_material=ub.MeshMaterial.create_rough_metal())
        self.drones= []
        # #5BF527 - green
        # #2A21FF - azul claro
        # #3C93FA - azul mais claro
        # #0700CC - azul escuro
        # #6F27F5 - roxo claro
        # #00FFB7 - verde agua
        self.pc_color = '#5BF527'
        self.drone_colors = ['#00FFB7']*self.param_n_robots #['red','green','blue','yellow','magenta','cyan','gray','brown','DarkSlateGray','black']

        #Create the drones
        for i in range(self.param_n_robots):
            drone_body = ub.RigidObject(list_model_3d=[self.drone_model],htm=ub.Utils.trn([0,0,0])*ub.Utils.rotx(np.pi/2))
            drone_ball = ub.Ball(color=self.drone_colors[i], radius=self.param_radius, opacity=self.drone_sphere_opacity)
            new_drone = ub.Group(list_of_objects=[drone_body, drone_ball])
            self.drones.append(new_drone)

    def run(self):
        # Test only moving obs
        # t = 0
        # while t < self.param_t_max:
        #     self.mov_obs.add_ani_frame(time = t, htm = self.mov_obs_htm(t))
        #     t += self.dt
        # return
    
        self.q = np.matrix(np.zeros((3*self.param_n_robots,1)))
        for i in range(self.param_n_robots):
            # print("Inserting start_point[i] = {}".format(self.start_points[i]))
            # print("self.q[3*i:3*(i+1),:].shape = {}".format(self.q[3*i:3*(i+1),:].shape))
            self.q[3*i:3*(i+1),:] = self.start_points[i]
            
        #Everyone starts stopped
        self.dotq = np.matrix(0*self.q)
        # self.hist_dotq = []
        # self.hist_ddotq = []
        # self.hist_t = []
        # self.hist_dist_agent = []
        # self.hist_dist_obs = []

        self.init_index = [0 for i in range(self.param_n_robots)]
        self.current_tg = [self.path_points[i][0] for i in range(self.param_n_robots)]
        self.finished = [False for i in range(self.param_n_robots)]

        # for i in range(self.param_n_robots):
        #     self.all_tg_box[i].add_ani_frame(0,htm=ub.Utils.trn(self.current_tg[i]))

        self.mov_obs.add_ani_frame(0, htm=self.mov_obs_htm(0)) # starting position
        self.mov_obs2.add_ani_frame(0, htm=self.mov_obs2_htm(0)) # starting position
        cont = True 

        self.min_dist_agents = 1e6
        self.min_dist_obs = 1e6
        
        self.reached_waypoint = [False for i in range(self.param_n_robots)]
        need_to_break = True # indicates whether we need to brake the loop that iterates over 
        # all drones,
        # because we need to make sure all drones reach their waypoints so we can
        # start fresh on this loop, so every drone updates its own state
        iter_count = 0
        total_error = 0
        prev_error = 0
        total_error_count = 0
        while cont:
            # print("self.q.shape = {}".format(self.q.shape))
            # print("self.dotq.shape = {}".format(self.dotq.shape))
            t = i*self.dt
            pc_idx = int(self.init_index[0])
            ##int(t/self.param_t_max*len(self.master_path))
            
            self.ddotq, self.min_dist_agents_now, self.min_dist_obs_now = self.control.control_fun(t, self.q, self.dotq, self.current_tg, self.pc, self.mov_obs_props)
            
            self.min_dist_agents = min(self.min_dist_agents_now, self.min_dist_agents)
            self.min_dist_obs = min(self.min_dist_obs_now, self.min_dist_obs)
            
            # Move cursor up to overwrite previous output (skip on first iteration)
            n_lines = 1 + self.param_n_robots  # 1 for Time + n for drones
            if iter_count > 0:
                sys.stdout.write(f"\033[{n_lines}F")  # Move cursor up n_lines
            
            # Print Time line (clear line first)
            sys.stdout.write("\033[K")  # Clear line
            print("Time "+str(round(t,2))+"/"+str(self.param_t_max)+", min_dist_agent = "+str(round(self.min_dist_agents,2))+", min_dist_obs = "+str(round(self.min_dist_obs,2)))
            
            total_finished = True
            total_error = 0
            
            for j in range(self.param_n_robots):
                qj = self.q[3*j:3*(j+1),:]
                error = np.linalg.norm(qj-self.current_tg[j])
                no_targets = len(self.path_points[j])
                
                sys.stdout.write("\033[K")  # Clear line
                if self.finished[j]:
                    print("Robot "+str(j+1)+", FINISHED!, error  = "+str(round(error,2)))
                else:
                    print("Robot "+str(j+1)+", ind = "+str(self.init_index[j]+1)+"/"+str(no_targets)+", error = "+str(round(error,2)))
            
                sys.stdout.flush()
                total_error += error

                
                if error <= (self.param_dist_interm if self.init_index[j] < no_targets-1 else self.param_dist_final):
                    if not self.reached_waypoint[j]:
                        self.reached_waypoint[j] = True                    

                    # wait for everyone to reach the first waypoint before moving to the next one
                    if not all(self.reached_waypoint[i] for i in range(self.param_n_robots)):
                        # print("Not all drones reached their waypoint. Continuing.")
                        continue
                    
                    
                    # If we got here, it means all drones reached their waypoint.
                    # However, it means that only this drone will update the following lines, 
                    # and the rest of the loop for n_robots will not update it.
                    # So we need to break this loop, and start fresh on the next one
                    # so every single drone is able to update its status and go to the next wp
                    if need_to_break:
                        # print("All drones reached their waypoint. Need to break the loop.")
                        # We actually always want to break once we get here so I'm commenting this out
                        # need_to_break = False 
                        total_finished = False

                        for k in range(self.param_n_robots):
                            self.init_index[k]+=1
                            # print(f"Incrementing init_index[{k}] to {self.init_index[k]}")
                            # print("no_targets = {}".format(no_targets))
                            if self.init_index[k] == no_targets:
                                self.init_index[k] = no_targets-1
                                self.finished[k] = True
                                # print(f"Setting finished[{k}] to True")

                            self.current_tg[k] = self.path_points[k][self.init_index[k]]
                        
                        # Reset everyone
                        self.reached_waypoint = [False for i in range(self.param_n_robots)]
                            
                        if all(self.finished[i] for i in range(self.param_n_robots)):
                            total_finished = True
                        
                        # print("Breaking the loop.")
                        break

                else:
                    # print("Error not small enough")
                    pass

                total_finished = total_finished and self.finished[j]

            # # Reset everyone
            cont = t < self.param_t_max and not total_finished
            i = i + 1
            iter_count += 1
            
            self.q += self.dotq*self.dt
            self.dotq += self.ddotq*self.dt
            
            # self.hist_dotq.append(np.matrix(self.dotq))
            # self.hist_ddotq.append(self.ddotq)
            # self.hist_t.append(t)
            # self.hist_dist_agent.append(self.min_dist_agents_now)
            # self.hist_dist_obs.append(self.min_dist_obs_now)

            # Add ani frame for moving obstacle
            self.mov_obs.add_ani_frame(time = i*self.dt, htm = self.mov_obs_htm(t))
            self.mov_obs2.add_ani_frame(time = i*self.dt, htm = self.mov_obs2_htm(t))
            # self.mov_obs_htm *= ub.Utils.trn(self.mov1_obs_props['velocity'](t)*self.dt)
            # if abs(self.mov_obs_htm[0,3]) >= self.stage_size:
            #     self.mov_obs_vel *= -1 # invert

            # Add ani frame for drones and PC
            for j in range(self.param_n_robots):
                self.drones[j].add_ani_frame(time=i*self.dt, htm=ub.Utils.trn(self.q[3*j:3*(j+1),:]))
                #self.master_path_pc.add_ani_frame(time=i*self.dt, initial_ind=max(0, pc_idx-int(self.param_max_pc_points_per_frame/2)), final_ind=min(len(self.master_path), pc_idx+int(self.param_max_pc_points_per_frame/2)))
                self.master_path_pc.add_ani_frame(time=i*self.dt, initial_ind=max(0, pc_idx-int(self.param_max_pc_points_per_frame)), final_ind=min(len(self.master_path), pc_idx+1))
        

            # If error stops decreasing, kill program
            if (abs(total_error - prev_error)) <= 1e-5:
                # print(f"total_error - prev_error = {total_error - prev_error}")
                total_error_count += 1
                # print(f"total_error_count = {total_error_count}")
                if total_error_count > 100:
                    break
            else:
                total_error_count = 0
                # print(f"total_error_count = {total_error_count}")
            
            prev_error = total_error
            # print(f"prev_error = {prev_error}")
    
    def save_simulation(self):
        self.sim.set_parameters(width=1500, height=1500, pixel_ratio=0.9)
        self.sim.save(os.getcwd(),"final")
        print("Saved simulation to final.html")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("ATTENTION ! This is a complex script which takes A LOT of time to run.")
        print("There is a final.html file provided with the full final result.")
        print("You can view the simulation by opening the file in your browser.")
        input("Press Enter if you really want to proceed or ctrl+c to exit.")
    
    drone_show = DroneShow()
    drone_show.run()
    drone_show.save_simulation()