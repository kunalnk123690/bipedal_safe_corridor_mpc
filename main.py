import mujoco as mj
import numpy as np
from LIPMPC import *
from CBFOBS import *
from DIGITQP import *
from viewer import *
from utils import *
import time


model = mj.MjModel.from_xml_path('./DigitModel/scene.xml')
data = mj.MjData(model)

# Set Initial State of Digit
data.qpos = np.array([0, 0, 0.83, 1, 0, 0, 0, 
                      2.983497082849632109e-01, -4.609038906699547411e-02, -1.028745509393533958e-02, 
                      -6.180393852816212785e-01, 7.793554143652230426e-01, -8.069510787823898357e-02, 6.419311415445301539e-02, 
                      -2.070011726210992664e-01, -2.244121380834677071e-04, 2.110699778217571820e-01, -9.877097799413092627e-04, 
                      1.236112663484994628e-01, 
                      9.951418147941888392e-01, -7.412700813079739492e-02, 8.443159191734769115e-03, -6.423914831758105459e-02, 
                      -1.744582995717965102e-01, 
                      9.882915693413950597e-01, -1.212740307946869184e-01, -1.727696856720108143e-02, 9.096092447122425262e-02, 
                      -1.432831937253021826e-01, -9.104605353826955572e-02, 
                      -0.106437, 0.89488, -1.860540486313262858e-05, 0.344684, 
                      -2.920205202074338535e-01, 4.328204378325633400e-02, 1.147273422695336415e-02, 
                      7.911839121390622509e-01, -6.027482048891177335e-01, -6.289769939569501978e-02, 8.225872650365767536e-02, 
                      2.078483653506341677e-01, 2.582094172140779226e-04, -2.120813876878341331e-01, 1.025737127413941900e-03, 
                      -1.242014477753271978e-01, 
                      9.949675536918867191e-01, 7.617350960264548942e-02, 8.627290683723530182e-03, 6.451924821831753198e-02, 
                      1.758658078088930210e-01, 
                      9.790791561883002148e-01, 1.808192819113194350e-01, -2.293005118224257163e-02, -9.045775787327837991e-02, 
                      1.443733939669925581e-01, 9.228187573855245462e-02, 
                      0.106437, -0.89488, 1.860540486313262858e-05, -0.344684])
mj.mj_forward(model, data)
map = MAP('Map')


############## Initialize MPC ##############
MPCData = {'H': 0.95,
           'Tst': 0.3,
           'N': 5,
           'alpha': 0.15,
           'Disturbance': False,
           'footboundx': np.array([-0.1, 0.6]),
           'footboundy': np.array([0.1, 0.5]),
           'distboundcom': np.array([0.00001, 0.3]),
           'gravity': 9.806,
           'rotConstr': False,
           'nPol_sides': 20,
           'CasadiOpts': dict(print_time = False, verbose = False, expand = True, error_on_fail = False),
           'solver': 'fatrop',
           'SolverOpts': dict(print_level = 0),}

p_goal = map.goal
x_goal = np.array([[p_goal[0], p_goal[1], 0, 0]]).T
c_obs, r_obs = map.getObsCenterRadi()
H_obs = 1*np.ones((map.n_obs, 1))

############## Initialize MPC ##############
MPC = LIPMPC(MPCData)


############## Initialize QP WBC ##############
QPData = {'zCL': 0.08, #walking clearance
          'KD': 10*np.diag([1,1,0.9,2,2,2,10,10,10,5,5,3]),
          'KP': (10*np.diag([1,1,0.9,2,2,2,10,10,10,5,5,3])/2)**2,
          'solver': 'osqp',
          'CasadiOpts': dict(print_time = False, verbose = False, error_on_fail = False, warm_start_primal = True, warm_start_dual = True),
          'SolverOpts': dict(verbose = False),}
DQP = PWQP(QPData)

# initialize variables
Tk_1 = 0
t_MPC_lastrun = 0
isRightStance = True
theta_init = 0.0
n_steps = 0
des_thSwFk = 0.0


q, dq, _ = output(data)
s = np.min([(data.time-Tk_1)/MPC.Tst, 1.0])

############## Select Foot ##############
if isRightStance:
    T_SwFoot = DQP.Dcf.Functions.leftfoot_pose(q).full()
    T_StFoot = DQP.Dcf.Functions.rightfoot_pose(q).full()
else:
    T_SwFoot = DQP.Dcf.Functions.rightfoot_pose(q).full()
    T_StFoot = DQP.Dcf.Functions.leftfoot_pose(q).full()
    
thSwFk_1 = np.arctan2(T_SwFoot[1,0],T_SwFoot[0,0])
PSwFk_1 = T_SwFoot[0:2,3].reshape((2,1))




############## Simulation Loop ##############
mjviewer = viewerObject(model, data)
while mjviewer.viewer.is_alive:
    
    "Traveling logic"
    x = DQP._extractLIPStatefromFull(q, dq)
    
    if map.cur_poly_idx+1 < map.n_pol:
        if map.checkinPoly(x[0:2], map.cur_poly_idx+1): #check if next polytope
            map.incrementGoalandPoly()
            p_goal = map.goal
            x_goal = np.array([[p_goal[0],p_goal[1],0,0]]).T
    
    if map.cur_poly_idx+1 == map.n_pol and np.linalg.norm(x[0:2]-x_goal[0:2])<0.5: 
        print("Reached the goal, stopping simulation.")
        break
    mjviewer.addGoal(map.goalsPxy[map.cur_goal_idx][:].reshape((1,2)),0.2*np.array([[1,1]]),np.array([[2]]))


    #############################################
    q, dq, _ = output(data)
    x0 = DQP._extractLIPStatefromFull(q, dq)
    s = np.min([(data.time-Tk_1)/MPC.Tst, 1.0])

    if isRightStance:
        T_SwFoot = DQP.Dcf.Functions.leftfoot_pose(q).full()
        T_StFoot = DQP.Dcf.Functions.rightfoot_pose(q).full()
    else:
        T_SwFoot = DQP.Dcf.Functions.rightfoot_pose(q).full()
        T_StFoot = DQP.Dcf.Functions.leftfoot_pose(q).full()


    ############## MPC ##############
    theta_goal = (np.arctan2(x_goal[1]-x[1],x_goal[0]-x[0]))[0]
    PStFoot = T_StFoot[0:3,3].reshape((3,1))
    if data.time == 0.0 or data.time - t_MPC_lastrun > 0.02:
        try:
            t_MPC_start = time.time()
            p_fst, des_thSwFk = MPC.Solve(x0, s, x_goal, isRightStance, des_thSwFk, theta_goal, map.A, map.b, PStFoot[0:2])
            t_MPC_end = time.time()
            # print("MPC time (ms): ", (t_MPC_end - t_MPC_start)*1000)
            t_MPC_lastrun = data.time
        except:
            print('MPC failed')
            mjviewer.viewer._paused = True
            break


    ############## Solve QP ##############
    try:
        t_QP_start = time.time()
        des_X_LIP = MPC._CLSolution_abs(x0, data.time - t_MPC_lastrun, PSwFk_1)
        tau = DQP.WalkingQP(q, dq, MPC.H, isRightStance, s, PSwFk_1, p_fst, MPC.Tst, des_X_LIP, thSwFk_1, des_thSwFk)
        data.ctrl = tau
        t_QP_end = time.time()
        # print("QP time (ms): ", (t_QP_end - t_QP_start)*1000)
    except:
        print('QP failed')
        break


    ############## Switching Algo ##############
    PSwFoot = T_SwFoot[0:3, 3].reshape((3, 1))
    if(s >=0.5 and (PSwFoot[2]<1e-2 or DQP.SwFspring(q,isRightStance)>0.01)):
        n_steps += 1
        isRightStance = not isRightStance
        s = 0
        Tk_1 = data.time
        t_MPC_lastrun = data.time
        PSwFk_1 = PStFoot[0:2]
        p_fst = PSwFk_1
        thSwFk_1 = getUnwrappedThetatoLast(np.arctan2(T_SwFoot[1,0], T_SwFoot[0,0]), thSwFk_1)




    ############## Simulate ##############
    mjviewer.addCuboidObs_MAP(c_obs, r_obs, H_obs)
    mj.mj_step(model, data, nstep=30)
    mjviewer.render()

    if data.time > 200:
        print("Simulation time exceeded 100 seconds, stopping.")
        break

    
while mjviewer.viewer.is_alive:
    mjviewer.render()
    time.sleep(0.03)