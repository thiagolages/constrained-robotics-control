import uaibot as ub
import numpy as np

def task_fun(_q, _t, _robot, _htm_tg):

    n = np.shape(_q)[0]

    jac, htm_e = _robot.jac_geo(_q)

    x = htm_e[0:3,0]
    y = htm_e[0:3,1]
    z = htm_e[0:3,2]
    s = htm_e[0:3,3]

    htm_tg_t = _htm_tg(_t)

    xd = htm_tg_t[0:3,0]
    yd = htm_tg_t[0:3,1]
    zd = htm_tg_t[0:3,2]
    sd = htm_tg_t[0:3,3]

    jac_v = jac[0:3,:]
    jac_w = jac[3:6,:]

    r = np.matrix(np.zeros((6,1)))

    r[0:3,0] = s-sd
    r[3,0] = 1-xd.T*x
    r[4,0] = 1-yd.T*y
    r[5,0] = 1-zd.T*z

    jac_r = np.matrix(np.zeros((6,n)))

    jac_r[0:3,:] = jac_v
    jac_r[3,:] = xd.T*ub.Utils.S(x)*jac_w
    jac_r[4,:] = yd.T*ub.Utils.S(y)*jac_w
    jac_r[5,:] = zd.T*ub.Utils.S(z)*jac_w

    return r, jac_r

def fun_phi(_r):
    K = 2.0
    _r_mod = np.matrix(np.zeros((6,1)))

    for i in range(6):
        _r_mod[i,0] = -K*np.sqrt(_r[i,0]) if _r[i,0]>=0 else K*np.sqrt(-_r[i,0])

    return _r_mod

def compute_control(_q, _t, _robot, _htm_tg):

    r, jac_r = task_fun(_q, _t, _robot, _htm_tg)

    dt = 0.01

    r_plus, _ = task_fun(_q, _t+dt, _robot, _htm_tg)
    r_minus, _ = task_fun(_q, _t-dt, _robot, _htm_tg)

    ff = (r_plus-r_minus)/(2*dt)

    u = ub.Utils.dp_inv(jac_r, 0.01)*(fun_phi(r)-1*ff)

    return u


def measure_config(_robot):
    return _robot.q

def send_joint_velocity(_robot, _u, _t):
    dt = 0.01
    qprox = _robot.q + _u*dt
    _robot.add_ani_frame(time=_t+dt, q = qprox)


######################################
robot = ub.Robot.create_kuka_lbr_iiwa()

htm_tg = lambda _t: ub.Utils.trn([0.3,0.2,0.8+0.2*np.sin(0.25*2*np.pi*_t)])

frame = ub.Frame(htm = htm_tg(0), size = 0.3)



sim = ub.Simulation([robot, frame])

# robot.add_ani_frame(time=0, q = q_inv)


#Control loop
dt = 0.01
t = 0

for i in range(2000):

    q = measure_config(robot)
    u = compute_control(q, t, robot, htm_tg)
    send_joint_velocity(robot, u, t)

    frame.add_ani_frame(time=t, htm = htm_tg(t))
    t+=dt


sim.run()
