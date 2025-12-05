# import numpy as np
# import uaibot as ub
# from utils import *
# from DroneControl import DroneControl
# ###################################
# class DroneShow():
#     def __init__(self):
#         # All params
#         self.param_dict = dict(
#             param_k = 1.4,
#             param_eta = 0.6,
#             param_eta_v = 0.3,
#             param_v_max = 0.4,
#             param_a_max = 0.5,
#             param_radius = 0.25,
#             param_height = 0.35,
#             param_n_robots = 2,
#             param_dist_interm = 0.3,
#             param_dist_final = 0.03,
#             param_h_dist = 1e-6,
#             param_eps_dist = 1e-6,
#             param_delta = 0.05,
#             param_h_dist = 0.2,
#             param_eps_dist = 0.01,
#             param_delta = 0.01,
#             param_min_dist_drone_tg = 1.0,
#             param_min_dist_tg = 1.0,
#             param_t_max = 50,
#             param_xmin = -1.5,
#             param_xmax = 1.5,
#             param_ymin = -1.5,
#             param_ymax = 1.5,
#             param_zmin = 0.5,
#             param_zmax = 1.0,
#             dt = 0.02,
#         )
#         self.drones = []
#         self.obstacles = []
#         self.targets = []
#         self.path_points = []
#         self.hist_dotq = []
#         self.hist_ddotq = []
#         self.hist_t = []
#         self.hist_dist_agent = []
#         self.hist_dist_obs = []
#         self.setup()

#     def setup(self):
#         self.control = DroneControl(self.param_dict)
#         self.setup_drones()
#         self.setup_obstacles()
#         self.setup_points()
#         self.setup_initial_poses()
#         self.setup_target_points()
#         self.setup_initial_path()
#         self.create_simulation()
#         self.setup_initial_state()


#     def setup_initial_state(self):
#         # Set current targets
#         self.current_tg = [self.path_points[i][0] for i in range(self.param_n_robots)]
#         self.init_index = [0 for i in range(self.param_n_robots)]
#         self.finished = [False for i in range(self.param_n_robots)]
#         self.cont = True
#         self.min_dist_agents = 1e6
#         self.min_dist_obs = 1e6
#         self.drone_model = ub.Model3D(
#             'https://cdn.jsdelivr.net/gh/viniciusmgn/uaibot_content@master/contents/CrazyFlie/crazyflie.obj',
#             scale=1, mesh_material=ub.MeshMaterial.create_rough_metal())

#     def create_simulation(self):
#         # sim = ub.Simulation.create_sim_mountain(drones, light_intensity=1.5)
#         self.sim = ub.Simulation([self.drones])
#         # sim.add(pc)
#         self.sim.add(self.all_obstacles)
#         self.sim.add(self.all_tg_box)
#         self.sim.set_parameters(pixel_ratio=0.9)
#         self.sim.set_parameters(camera_start_pose=[ 4.2967, 2.4381, 3.5080, 3.6016, 2.0036, 2.9353, 1.0000])
#         self.sim.set_parameters(width=550,height=500)

#     def setup_initial_poses(self):
#         #Sample initial poses for the drones
#         self.start_points = [np.matrix([0,0,0]).T for i in range(self.param_n_robots)]

#         for i in range(self.param_n_robots):
            
#             cont = True
#             print("Finding initial pose for drone "+str(i+1)+" ... ")
#             while cont:
#                 htm=ub.Utils.htm_rand([self.param_xmin,self.param_ymin,self.param_zmin],[self.param_xmax,self.param_ymax,self.param_zmax],0)
#                 self.drones[i].add_ani_frame(0,htm)
#                 self.start_points[i] = htm[0:3,-1]
                
#                 #Check if this is ok
#                 self.ball_drone_i = self.drones[i].list_of_objects[-1]
#                 collided = self.ball_drone_i.compute_dist(self.pc)[2]<self.param_radius
                
#                 j = 0
#                 while (not collided) and (j<i) :
#                     ball_drone_j = self.drones[j].list_of_objects[-1]
#                     collided = self.ball_drone_i.compute_dist(ball_drone_j)[2] < self.param_radius
#                     j+=1
#                 cont = collided
                        
#     def setup_target_points(self):
#         #Sample target points that are far apart from each other, not coliding
#         #and far for the respective drone

#         self.all_tg_points = []
#         self.all_tg_box = []

#         for i in range(self.param_n_robots):
            
#             cont = True
            
#             print("Finding target points for drone "+str(i+1)+" ... ")
            
#             while cont:            
#                 tg_point_try=ub.Utils.htm_rand([self.param_xmin,self.param_ymin,self.param_zmin],[self.param_xmax,self.param_ymax,self.param_zmax],0)[0:3,-1]
#                 #Check if this is ok
#                 self.ball_drone_i = self.drones[i].list_of_objects[-1]
                
#                 unfit = self.pc.projection(tg_point_try)[1]<0.3 or self.ball_drone_i.projection(tg_point_try)[1]<self.param_min_dist_drone_tg 
                
#                 #Check if far away from others points
#                 j = 0
#                 while (not unfit) and (j<i) :
#                     unfit = np.linalg.norm(self.all_tg_points[j]-tg_point_try)<self.param_min_dist_tg
#                     j+=1        
#                 cont = unfit

#             self.all_tg_points.append(tg_point_try)
#             self.all_tg_box.append(ub.Box(htm=ub.Utils.trn(tg_point_try), color=self.drone_colors[i],width=0.03,depth=0.03,height=0.03))                            
#         print("Finished!")
        
#     def setup_initial_path(self):
#         #Use motion planer
#         self.path_points = []
#         for i in range(self.param_n_robots):
#             path = random_path_planner(self.start_points[i], self.all_tg_points[i], self.pc, self.param_radius, 
#                                     [self.param_xmin-self.param_radius/2, self.param_xmax+self.param_radius/2, 
#                                         self.param_ymin-self.param_radius/2, self.param_ymax+self.param_radius/2,
#                                         self.param_zmin-self.param_radius/2, self.param_zmax+self.param_radius/2])
#             self.path_points.append(path)

#     def setup_obstacles(self):
#         obs_opacity = 0.9
#         obs_color = "#51A2FF" # light blue
#         obs1 = ub.Box(htm=ub.Utils.trn([0,0,0.5]),width=1.0,depth=0.2,height=1.0,opacity=obs_opacity,color=obs_color)
#         obs2 = ub.Box(htm=ub.Utils.trn([0.6,0.1,0.5]),width=0.2,depth=1.0,height=1.0,opacity=obs_opacity,color=obs_color)
#         obs3 = ub.Box(htm=ub.Utils.trn([-1.0,-1.0,0.3]),width=0.4,depth=0.4,height=0.6,opacity=obs_opacity,color=obs_color)
#         obs4 = ub.Box(htm=ub.Utils.trn([-1.0,1.0,0.2]),width=0.4,depth=0.4,height=0.4,opacity=obs_opacity,color=obs_color)
#         obs5 = ub.Box(htm=ub.Utils.trn([0.6,-0.9,0.95]),width=0.2,depth=1.0,height=0.1,opacity=obs_opacity,color=obs_color)
#         obs6 = ub.Box(htm=ub.Utils.trn([0.6,-1.4,0.5]),width=0.2,depth=0.2,height=1.0,opacity=obs_opacity,color=obs_color)
#         self.all_obstacles = [obs1, obs2, obs3, obs4, obs5, obs6, wallxp, wallxn, wallyp, wallyn]

#     def setup_points(self):
#         self.all_points = []
#         for obs in self.all_obstacles:
#             self.all_points+=  [np.matrix(p).T for p in obs.to_point_cloud(disc=0.06).points.T]
#         self.pc = ub.PointCloud(points=self.all_points,size=0.03,color='#005500')

#     def setup_drones(self):
#         self.drones= []
#         # #5BF527 - green
#         self.drone_colors = ['#5BF527']*self.param_n_robots #['red','green','blue','yellow','magenta','cyan','gray','brown','DarkSlateGray','black']

#         #Create the drones
#         for i in range(self.param_n_robots):
#             drone_body = ub.RigidObject(list_model_3d=[self.drone_model],htm=ub.Utils.trn([0,0,0])*ub.Utils.rotx(np.pi/2))
#             drone_ball = ub.Ball(color=self.drone_colors[i], radius=0.1, opacity=0.7)
#             new_drone = ub.Group(list_of_objects=[drone_body, drone_ball])
#             self.drones.append(new_drone)