import numpy as np
from pyquaternion import quaternion
import casadi as cs
from angles import *
from DigitModel.CasadiFunctions.DigitCasadiFuncs import DigitCasFunc

def QuatOriError(desQuat,curQuat):
    """Compute a world-frame quaternion orientation error.

    :param desQuat: Desired scalar-first quaternion.
    :param curQuat: Current scalar-first quaternion.
    :return: Three-dimensional orientation error.
    :rtype: numpy.ndarray
    """
    q_d = quaternion.Quaternion(desQuat)
    R_d = q_d.rotation_matrix
    q_c = quaternion.Quaternion(curQuat)
    R_c = q_c.rotation_matrix
    R_e = 0.5*(R_c.T@R_d - R_d.T@R_c)
    return -R_c@np.array([R_e[2,1],R_e[0,2],R_e[1,0]])


def QuatOriError_woR(desQuat,curQuat):
    """Compute a body-frame quaternion orientation error.

    :param desQuat: Desired scalar-first quaternion.
    :param curQuat: Current scalar-first quaternion.
    :return: Three-dimensional orientation error.
    :rtype: numpy.ndarray
    """
    q_d = quaternion.Quaternion(desQuat)
    R_d = q_d.rotation_matrix
    q_c = quaternion.Quaternion(curQuat)
    R_c = q_c.rotation_matrix
    R_e = 0.5*(R_c.T@R_d - R_d.T@R_c)
    return -np.array([R_e[2,1],R_e[0,2],R_e[1,0]])


class PWQP():#pointwiseQP
    """Pointwise whole-body QP controller for Digit walking."""

    def __init__(self, QPData):
        """Build the whole-body walking QP.

        :param QPData: Gains and CasADi solver configuration.
        :type QPData: dict
        """
        # Constants
        self.n_ddq = 36
        self.n_u = 20
        self.n_lambda = 18 # Total lambda vars, will be broken down for specific constraints
        # self.n_v = self.n_ddq + self.n_u + self.n_lambda # Not directly used in Opti, but good to keep in mind
        self.Dcf = DigitCasFunc() # digit casadi functions
        self.zCL = QPData['zCL'] # walking clearence
        self.totM = 47.925414 #######################total mass check!!
        self.g = 9.806

        self.KD = QPData['KD']
        self.KP = QPData['KP']

        self.QPData = QPData

        # --- Casadi Opti Setup ---
        self.opti = cs.Opti("conic")

        # Defining Casadi Symbolic Variables
        self.vddq = self.opti.variable(self.n_ddq, 1)
        self.vu = self.opti.variable(self.n_u, 1)
        self.vlambdaCL = self.opti.variable(6, 1) #for closed loop
        self.vlambdaStF = self.opti.variable(6, 1) #for stance foot
        self.vlambdaSwF = self.opti.variable(6, 1) #for swing foot if considering on ground
        self.vlambdaSwFsp = self.opti.variable(2, 1) #making spring of Leg inf stiff (likely not needed now if constraints are perfect)
        self.vetaStF = self.opti.variable(6, 1) #relaxaction factor for stance foot contact
        self.vetaSwF = self.opti.variable(6, 1) # on swing foot
        self.vlambdaFsp = self.opti.variable(4, 1) #stiffsprings

        # Define Casadi Symbolic Parameters
        self.pH = self.opti.parameter(self.n_ddq, self.n_ddq)
        self.pC_terms = self.opti.parameter(self.n_ddq, 1)
        self.ptau = self.opti.parameter(self.n_ddq, 1)

        self.pJacCL = self.opti.parameter(6, self.n_ddq) #closed loop constraint
        self.pdJacCLdq = self.opti.parameter(6, 1)

        self.R2StF = self.opti.parameter(2, 2) #stance foot rotation matrix 2x2
        self.pJacStF = self.opti.parameter(6, self.n_ddq) #stance foot constraint
        self.pdJacStFdq = self.opti.parameter(6, 1)
        self.pJacSwF = self.opti.parameter(6, self.n_ddq) #swing foot constraint
        self.pdJacSwFdq = self.opti.parameter(6, 1)
        self.JacFsp= np.zeros((4,self.n_ddq))
        self.JacFsp[[0,1,2,3],[10,12,25,27]]=1 # This should ideally be a Casadi parameter if it varies, but if fixed, can be a numpy array

        self.pdesddq = self.opti.parameter(self.n_u, 1) #desired actuated acceleration

        self.pAG = self.opti.parameter(6, self.n_ddq) #CMM
        self.pdAGdq = self.opti.parameter(6, 1) #dCMMdq
        self.pdesdhG = self.opti.parameter(6, 1) #desired centroidal momentum
        self.pdesddh = self.opti.parameter(12, 1) #desired output space acc

        # Fixed values (can be numpy arrays or Casadi constants)
        self.JacBase = cs.DM(np.concatenate((np.zeros((3,3)),np.eye(3),np.zeros((3,30))),1)) # Jacobian of base
        self.B = cs.DM(self.Dcf.Functions.B()['o0'].full()) # input mapping matrix
        self.ctrlrange = np.array([[-1.4, -1.4, -12.5, -12.5, -0.9, -0.9, -1.4, -1.4, -1.4, -1.4, 
                                    -1.4, -1.4, -12.5, -12.5, -0.9, -0.9, -1.4, -1.4, -1.4, -1.4],
                                    [1.4, 1.4, 12.5, 12.5, 0.9, 0.9, 1.4, 1.4, 1.4, 1.4, 
                                     1.4, 1.4, 12.5, 12.5, 0.9, 0.9, 1.4, 1.4, 1.4, 1.4]]).T

        # Contact constants
        self.min_Fz = 20 #minimum up GRF
        self.mu = 0.5 #friction coeff
        self.l_foot = 0.15
        self.w_foot = 0.07 #for xaxis
        self.z_c = 2 #for zaxis

        # Formulate problems (build the symbolic graph once)
        self._setupWalkingProblem()

        self.hdesired = np.zeros((20,1))
        self.Cur_e_h= np.zeros((12,1)) #error of the custom output
        self.Cur_hang = np.zeros((3,1)) #hangular momentum
        self.isFirstRun = True # Flag to indicate if it's the first step

    ###
    "getConstraints"
    def _getEOMSSConstraint(self): # EOM of Single support base
        """Add equations of motion and single-support constraints."""
        # Constraints are added directly to the opti object
        self.opti.subject_to(self.pH@self.vddq + self.pC_terms == self.ptau 
                                                                  + self.pJacCL.T@self.vlambdaCL 
                                                                  + self.pJacStF.T@self.vlambdaStF 
                                                                  + cs.DM(self.JacFsp).T@self.vlambdaFsp 
                                                                  + self.B@self.vu)  #EOM
        self.opti.subject_to(self.pJacCL@self.vddq + self.pdJacCLdq == 0) # closed loop constraint
        self.opti.subject_to(self.pJacStF@self.vddq + self.pdJacStFdq == 0) # stance foot constraint
        self.opti.subject_to(cs.DM(self.JacFsp)@self.vddq == 0) # stiff spring


    def _getFootContactConstraintWrench(self,vlambdaF,R): #same for either of foot hence parameteri Caron2015ICRA
        """Add a rectangular-foot contact-wrench cone.

        :param vlambdaF: Six-dimensional contact wrench variable.
        :param R: Planar rotation from world to foot coordinates.
        """
        # Note: Casadi's abs function handles expressions
        self.opti.subject_to(vlambdaF[5] >= self.min_Fz)                           #vertical GRF
        self.opti.subject_to(R@vlambdaF[[3,4]] <= (self.mu)*vlambdaF[5])     #upper friction
        self.opti.subject_to(R@vlambdaF[[3,4]] >= -(self.mu)*vlambdaF[5])    #lower friction
        self.opti.subject_to(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1] <= (self.w_foot/2)*vlambdaF[5])            #mx
        self.opti.subject_to(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1] >= -(self.w_foot/2)*vlambdaF[5])
        self.opti.subject_to(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1] <= (self.l_foot/2)*vlambdaF[5])            #my
        self.opti.subject_to(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1] >= -(self.l_foot/2)*vlambdaF[5])

        # Casadi abs for the complex terms
        expr_m_mu_mx = self.w_foot/2*(R[0,0]*vlambdaF[3] + R[0,1]*vlambdaF[4])-self.mu*(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1])
        expr_m_mu_my = self.l_foot/2*(R[1,0]*vlambdaF[3] + R[1,1]*vlambdaF[4])-self.mu*(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1])
        self.opti.subject_to(vlambdaF[2] >= -self.mu*(self.l_foot+self.w_foot)/2*vlambdaF[5]
                                            + cs.fabs(expr_m_mu_mx)
                                            + cs.fabs(expr_m_mu_my))

        expr_p_mu_mx = self.w_foot/2*(R[0,0]*vlambdaF[3] + R[0,1]*vlambdaF[4])+self.mu*(R[0,0]*vlambdaF[0] + R[0,1]*vlambdaF[1])
        expr_p_mu_my = self.l_foot/2*(R[1,0]*vlambdaF[3] + R[1,1]*vlambdaF[4])+self.mu*(R[1,0]*vlambdaF[0] + R[1,1]*vlambdaF[1])
        self.opti.subject_to(vlambdaF[2] <= +self.mu*(self.l_foot+self.w_foot)/2*vlambdaF[5]
                                            - cs.fabs(expr_p_mu_mx)
                                            - cs.fabs(expr_p_mu_my))

    def _getInputConstraints(self): #actuator input torque
        """Add actuator torque limits."""
        self.opti.subject_to(self.ctrlrange[:,0].reshape(self.n_u,1) <= self.vu) #lower bound on u
        self.opti.subject_to(self.ctrlrange[:,1].reshape(self.n_u,1) >= self.vu) #upper bound on u

    "Walking Controller"
    def _getWalkingCost(self): #both feet on ground
        """Construct the whole-body tracking objective.

        :return: Symbolic scalar objective.
        :rtype: casadi.MX
        """
        objecCentroidalM = cs.sumsqr(self.pAG[0:3,:]@self.vddq+self.pdAGdq[0:3]-self.pdesdhG[0:3])
        objecConfiguration = cs.sumsqr(self.vddq[[17,18,19,20, 32,33,34,35]]-self.pdesddq[[6,7,8,9, 16,17,18,19]])
        W1 = cs.DM(np.diag([1,1,1,1,1,1,1,1,1,1,1,1])*np.sqrt(100)) # W1 as a Casadi DM
        objective = cs.sumsqr(W1@(self._getTaskOutput_ddh()-self.pdesddh))+0.1*cs.sumsqr(self.vu) + 3*objecCentroidalM + 0.5*objecConfiguration
        return objective

    def _setupWalkingProblem(self): #creates Casadi opti problem for walking QP
        """Assemble and compile the parameterized walking QP."""
        # Add constraints
        self._getEOMSSConstraint()
        self._getInputConstraints()
        self._getFootContactConstraintWrench(self.vlambdaStF, self.R2StF) #stance foot

        # Set objective
        objective = self._getWalkingCost()
        self.opti.minimize(objective)

        # Choose solver
        self.opti.solver(self.QPData['solver'], self.QPData['CasadiOpts'], self.QPData['SolverOpts'])
        
        inputs = [self.pH, self.pC_terms, self.ptau, self.pJacCL, self.pdJacCLdq, self.pAG, self.pdAGdq, self.pdesdhG,  
                  self.pdesddq, self.R2StF, self.pJacStF, self.pdJacStFdq, self.pJacSwF, self.pdJacSwFdq, self.pdesddh]
        outputs = [self.vu]
        self.F = self.opti.to_function('F', inputs, outputs)


    def _getTaskOutput_ddh(self):
        """Construct task-space accelerations from QP variables.

        :return: Base, COM, and swing-foot accelerations.
        :rtype: casadi.MX
        """
        ddcomxy = (self.pAG[3:5,:]/self.totM)@self.vddq+self.pdAGdq[3:5,:]/self.totM
        ddcomz = (self.pAG[5,:]/self.totM)@self.vddq + self.pdAGdq[5,:]/self.totM
        h = cs.vertcat(self.vddq[3:6], #fixed base orientation
                       ddcomxy, #
                       ddcomz, # Use dcomz directly, no need for reshape
                       self.pJacSwF@self.vddq + self.pdJacSwFdq)
        return h

    def _getdeshdh(self, s, PSwFk_1, PSwFk, T, H, x_LIP):
        """Generate desired task position, velocity, and acceleration.

        :param s: Normalized step phase.
        :param PSwFk_1: Swing-foot position at liftoff.
        :param PSwFk: Desired touchdown position.
        :param T: Step duration.
        :param H: Desired COM height.
        :param x_LIP: Current LIP state.
        :return: Desired task position, velocity, and feedforward acceleration.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        """
        h = np.array([0,0,0,
                      0,0,0,
                      0,0,0,
                      0.5*((1+np.cos(np.pi*s))*PSwFk_1[0] + (1-np.cos(np.pi*s))*PSwFk[0])[0],
                      0.5*((1+np.cos(np.pi*s))*PSwFk_1[1] + (1-np.cos(np.pi*s))*PSwFk[1])[0],
                      self.zCL - 4*self.zCL*(s-0.5)**2])

        dh = np.array([0,0,0,
                       0,0,0,
                       0,0,0,
                      (0.5*np.pi*np.sin(np.pi*s)/T)*(PSwFk[0]-PSwFk_1[0])[0],
                      (0.5*np.pi*np.sin(np.pi*s)/T)*(PSwFk[1]-PSwFk_1[1])[0],
                       -8*self.zCL*(s-0.5)/T])

        ddh_FF = np.array([0,0,0,
                          (self.g/H)*(x_LIP[0,0]-PSwFk_1[0,0]),
                          (self.g/H)*(x_LIP[1,0]-PSwFk_1[1,0]),
                           0,
                           0,0,0,
                          (0.5*np.pi**2*np.cos(np.pi*s)/(T**2))*(PSwFk[0]-PSwFk_1[0])[0],
                          (0.5*np.pi**2*np.cos(np.pi*s)/(T**2))*(PSwFk[1]-PSwFk_1[1])[0],
                          -8*self.zCL/(T**2)])
        return h,dh,ddh_FF


    "Utility Functions"

    def _extractLIPStatefromFull(self, q, dq):
        """Extract planar COM position and velocity from the robot state.

        :param q: Generalized positions.
        :param dq: Generalized velocities.
        :return: LIP state ``[x, y, dx, dy]``.
        :rtype: numpy.ndarray
        """
        p_com = self.Dcf.Functions.pcom(q).full()
        v_com = self.Dcf.Functions.vcom(q, dq).full()
        return np.array([p_com[0],p_com[1],v_com[0],v_com[1]])


    def SwFspring(self, q, isRightStance):
        """Return the minimum swing-leg spring deflection.

        :param q: Generalized positions.
        :param isRightStance: Whether the right foot is in stance.
        :return: Minimum shin/heel spring deflection.
        :rtype: float
        """
        if isRightStance: #left swing
            shinSpring = -q[11]
            heelSpring = -q[13]
        else: #right swing
            shinSpring = q[26]
            heelSpring = q[28]
        return np.min([shinSpring, heelSpring])

    def handDesqpos(self):
        """Return the nominal arm joint configuration.

        :rtype: numpy.ndarray
        """
        out = np.array([-0.106437-0.3,0.89488,-0.00867,0.44684-0.5, 0.106437+0.3,-0.89488,0.00867,-0.44684+0.5, ])
        return out



    def WalkingQP(self, q, dq, des_comz, leftSwFoot, s, PSwFk_1, PSwFk, T, des_x_LIP, thSwFk_1, des_thSwFk):
        """Solve the walking QP and return actuator torques.

        :param q: Robot configuration with 37 entries.
        :param dq: Robot velocity with 36 entries.
        :param des_comz: Desired COM height.
        :param leftSwFoot: Whether the left foot is swinging.
        :param s: Normalized step phase.
        :param PSwFk_1: Swing-foot position at liftoff.
        :param PSwFk: Desired swing-foot position at touchdown.
        :param T: Step duration.
        :param des_x_LIP: Desired planar LIP state.
        :param thSwFk_1: Swing-foot yaw at liftoff.
        :param des_thSwFk: Desired swing-foot yaw at touchdown.
        :return: Actuator torque vector.
        :rtype: numpy.ndarray
        """

        '''setting parameter values'''
        Dq = self.Dcf.Functions.H_matrix(q).full()
        Hqdq = self.Dcf.Functions.C_terms(q,dq).full()
        tau_stiff = self.Dcf.Functions.tau_damp(dq).full() + self.Dcf.Functions.tau_stiff(q).full()[1::] #not includng friction
        JacCL = self.Dcf.Functions.CLJacg(q).full()
        dJacCLdq = self.Dcf.Functions.CLdJacgdq(q,dq).full()
        AG = self.Dcf.Functions.CMM(q,dq).full()
        dAGdq = self.Dcf.Functions.dCMMdq(q,dq).full()
        desdhG = np.vstack((-1*self.Dcf.Functions.CM(q,dq).full()[0:3], np.zeros((3,1)))) # Ensure correct size for pdesdhG
        Pcom = self.Dcf.Functions.pcom(q).full()
        Vcom = self.Dcf.Functions.vcom(q,dq).full()
        desddq = np.zeros((self.n_u, 1)) # Initialize desired ddq
        desddq[[6,7,8,9, 16,17,18,19]] = (-100*(q[[18,19,20,21, 33,34,35,36]]-self.handDesqpos()) -10*dq[[17,18,19,20, 32,33,34,35]]).reshape((8,1))


        if leftSwFoot: #left swing foot
            JacStF = self.Dcf.Functions.rightfoot_jac(q).full()
            dJacStFdq = self.Dcf.Functions.rightfoot_djac(q, dq).full()
            JacSwF = self.Dcf.Functions.leftfoot_jac(q).full()
            dJacSwFdq = self.Dcf.Functions.leftfoot_djac(q, dq).full()

            T_SwF = self.Dcf.Functions.leftfoot_pose(q).full()
            V_SwF = self.Dcf.Functions.leftfoot_vel(q, dq).full()
            T_StF = self.Dcf.Functions.rightfoot_pose(q).full()
            V_StF = self.Dcf.Functions.rightfoot_vel(q, dq).full()
        else: #right swing foot
            JacStF = self.Dcf.Functions.leftfoot_jac(q).full()
            dJacStFdq = self.Dcf.Functions.leftfoot_djac(q, dq).full()
            JacSwF = self.Dcf.Functions.rightfoot_jac(q).full()
            dJacSwFdq = self.Dcf.Functions.rightfoot_djac(q, dq).full()

            T_SwF = self.Dcf.Functions.rightfoot_pose(q).full()
            V_SwF = self.Dcf.Functions.rightfoot_vel(q, dq).full()
            T_StF = self.Dcf.Functions.leftfoot_pose(q).full()
            V_StF = self.Dcf.Functions.leftfoot_vel(q, dq).full()
        

        # set stance foot yaw
        yawStF = np.arctan2(T_StF[1,0], T_StF[0,0])
        R2StF = np.array([[np.cos(yawStF),-np.sin(yawStF)], [np.sin(yawStF),np.cos(yawStF)]]).T

        '''Outputs FL'''
        xaxis_SwF = T_SwF[0:2, 0]
        xaxis_StF = T_StF[0:2, 0]
        xaxis_desbase = (xaxis_SwF+xaxis_StF)/2
        theta_desbase = np.arctan2(xaxis_desbase[1], xaxis_desbase[0])
        quat_desbase = quaternion.Quaternion(axis=[0,0,1], angle=theta_desbase) #average angle of swing and stance foot xaxis
        Oebase = QuatOriError_woR(quat_desbase.q, np.array(q[3:7]))

        angvel_desbase = np.array([0,0,(V_StF[2,0]+V_SwF[2,0])/2]) # only vel around z axis
        dOebase = dq[3:6] - angvel_desbase
        #Pos des com pos
        Pcomxy = des_x_LIP[[0,1]].reshape((2,))
        Vcomxy = des_x_LIP[[2,3]].reshape((2,))
        #Ori SwingFoot
        quat_SwFk_1 = quaternion.Quaternion(axis=[0,0,1], angle=thSwFk_1)
        quat_SwF_Final = quaternion.Quaternion(axis=[0,0,1], angle=des_thSwFk)
        quat_desSwF = quaternion.Quaternion.slerp(quat_SwFk_1, quat_SwF_Final, s)
        quat_SwF = quaternion.Quaternion(matrix=T_SwF[0:3,0:3])
        OeSwF = QuatOriError(quat_desSwF, quat_SwF.q)

        dOeSwF = V_SwF[0:3] - np.array([[0,0,(Angle(des_thSwFk)-Angle(thSwFk_1)).toRadian]]).T/T
        
        # Pos SwingFoot
        PSwF = T_SwF[0:3,3]
        VSwF = V_SwF[3:6]

        # Desired LIP states
        x_LIP = np.array([Pcom[0], Pcom[1], Vcom[0], Vcom[1]])
        desh, desdh, desddh = self._getdeshdh(s, PSwFk_1, PSwFk, T, des_comz, x_LIP)
        e = desh.reshape((12,1)) - np.block([[Oebase.reshape((3,1))],
                                             [-np.array([[Pcomxy[0], Pcomxy[1], des_comz]]).T+Pcom],
                                             [OeSwF.reshape((3,1))],
                                             [PSwF.reshape((3,1))]])

        de = desdh.reshape((12,1)) - np.block([[dOebase.reshape((3,1))],
                                               [-np.array([[Vcomxy[0], Vcomxy[1], 0]]).T+Vcom],
                                               [dOeSwF.reshape((3,1))],
                                               [VSwF.reshape((3,1))]])


        desddh = np.array(desddh.reshape((12,1)) + self.KP@e + self.KD@de)
        
        '''Solve QP'''
        out = self.F(Dq, Hqdq, tau_stiff, JacCL, dJacCLdq, AG, dAGdq, desdhG, desddq, R2StF, JacStF, dJacStFdq, JacSwF, dJacSwFdq, desddh).full()

        return out.squeeze()
