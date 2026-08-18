import casadi as cs
import numpy as np
import scipy.signal as ssig


class LIPMPC():
    """Sequential footstep MPC for a constant-height LIP model."""

    def __init__(self, Data):
        """Build the MPC optimization problem.

        :param Data: Model, horizon, bounds, and CasADi solver options.
        :type Data: dict
        """
        self.solution = None #initialize
        
        # const
        self.H = Data['H'] #LIP height
        self.Tst = Data['Tst'] #time of one step
        self.g = Data['gravity']
        self.beta = np.sqrt(self.g/self.H)
        self.A = self._A(self.Tst) #system matrix
        self.B = self._B(self.Tst) #input matrix
        self.N = Data['N'] #number of steps horizon
        self.footboundx = Data['footboundx'] #bounds foot placement in x direction
        self.footboundy = Data['footboundy'] #bounds foot placement of left in y direction set lf_first flag to false to mirror to right
        self.distboundcom = Data['distboundcom'] #bounds com travel
        self.alpha = Data['alpha']
        self.nPol_sides = Data['nPol_sides']

        # formulate MPC d 
        self.opti = cs.Opti()
        self.opti.solver(Data['solver'], Data['CasadiOpts'], Data['SolverOpts'])
        self.isFirstRun = True
        
        # variables
        self.x = self.opti.variable(4,self.N+1) #states of COM of lip [px,py,vx,vy]_com
        self.pfst = self.opti.variable(2,self.N) #footstep position of LIP (px,py)_footstep
        self.theta = self.opti.variable(1,self.N+1) #heading angle
        
        # paramters
        self.x0 = self.opti.parameter(4) #initial state of COM of lip
        self.bx = self.opti.parameter(2) #bound on left foot x (lbx,ubx), same for right foor
        self.by = self.opti.parameter(2) #bound on left foot y (lby, uby), mirrored for right foot
        self.dbcom = self.opti.parameter(2) #bound on com travel within step (lb,ub)
        self.xgoal = self.opti.parameter(4) #goal state of LIP
        self.EP_cbf = self.opti.parameter(2,2) #ellipse shape matrix current cbf
        self.Ec_cbf = self.opti.parameter(2) #center of ellipse cbf
        self.PA_cbf = self.opti.parameter(self.nPol_sides,2) #Polygon matrix
        self.Pb_cbf = self.opti.parameter(self.nPol_sides)    #polygon const
        self.theta0 = self.opti.parameter(1) #last heading angle
        self.theta_goal = self.opti.parameter(1) #pointing angle towards goal
        self.EP_mcbf = self.opti.parameter(2,2) #moving obstacle ellipse matrices
        self.Ec_mcbf = self.opti.parameter(2,self.N+1) #moving obstacle ellipse center
        self.Ppfst_last = self.opti.parameter(2,1) #last foot step location relative

        # Set up MPC optimization problem
        self._formulateMPC()
        
        # for pd gains
        # placing desired poles after PD control
        des_poles = np.array([0.95, 0.95, 0.15, 0.15])
        self.K = ssig.place_poles(self.A,self.B,des_poles).gain_matrix
            
    
    def predictTouchDownLIPState(self, x_intermediate,s,T,PStFabs):
        """Predict the LIP state at the end of the current step.

        :param x_intermediate: Current LIP state ``[x, y, dx, dy]``.
        :type x_intermediate: numpy.ndarray
        :param s: Normalized step phase in ``[0, 1]``.
        :type s: float
        :param T: Step duration.
        :type T: float
        :param PStFabs: Absolute stance-foot position.
        :type PStFabs: numpy.ndarray
        :return: Predicted touchdown state.
        :rtype: numpy.ndarray
        """
        T_end = (1-s)*T
        A_abs = self._A_abs(T_end)
        B_abs = self._B(T_end)
        x_end = A_abs@x_intermediate + B_abs@PStFabs[0:2]
        return x_end

    def _CLSolution_abs(self,x0,t,pfst_abs):
        """Propagate the absolute-coordinate LIP solution.

        :param x0: Initial LIP state.
        :param t: Propagation time.
        :param pfst_abs: Absolute stance-foot position.
        :return: Propagated LIP state.
        :rtype: numpy.ndarray
        """
        return self._A_abs(t)@x0+self._B(t)@pfst_abs
        
    def _A_abs(self, t): #when foot position is absolute x-p
        """Return the absolute-coordinate LIP state matrix.

        :param t: Propagation time.
        :type t: float
        :rtype: numpy.ndarray
        """
        shbt = np.sinh(self.beta*t)
        bshbt = self.beta*shbt
        shbt_b = shbt/self.beta
        chbt = np.cosh(self.beta*t)
        return np.array([[chbt,0,shbt_b,0],
                        [0,chbt,0,shbt_b],
                        [bshbt,0,chbt,0],
                        [0,bshbt,0,chbt]])
        

    def _truncategoal(self, x0, xgoal):
        """Limit goal distance while preserving its direction.

        :param x0: Current LIP state.
        :param xgoal: Requested goal state.
        :return: Distance-limited goal with zero terminal velocity.
        :rtype: numpy.ndarray
        """
        pos = xgoal.reshape(4,)[0:2] -x0.reshape(4,)[0:2]
        # vel = x0.reshape(4,)[0:2]- xgoal.reshape(4,)[2:4]
        max_dist = 4
        if np.linalg.norm(pos)>max_dist: #goal point meters close
            pos = (pos/np.linalg.norm(pos))*max_dist

        return np.concatenate((x0[0:2,:]+pos.reshape(2,1),np.zeros((2,1))))


    def _updateIntialGuess(self,x0,theta0):
        """Shift state, footstep, and heading initial guesses.

        :param x0: Initial state guess.
        :param theta0: Initial heading guess.
        """
        #states
        self.opti.set_initial(self.x[:,0],x0)
        for k in range(self.N):
            self.opti.set_initial(self.x[:,k+1],self.opti.value(self.x[:,k]))
        #input footstep
        for k in range(self.N-1):
            self.opti.set_initial(self.pfst[:,k+1],self.opti.value(self.pfst[:,k]))
        #heading angle
        self.opti.set_initial(self.theta[0,0],theta0)
        for k in range(self.N):
            self.opti.set_initial(self.theta[0,k+1],theta0)
    ##    


    def _A(self,t):
        """Return the relative-foot LIP state matrix.

        :param t: Propagation time.
        :type t: float
        :rtype: numpy.ndarray
        """
        shbt_b = np.sinh(self.beta*t)/self.beta
        chbt = np.cosh(self.beta*t)
        return np.array([[1,0,shbt_b,0],
                [0,1,0,shbt_b],
                [0,0,chbt,0],
                [0,0,0,chbt]])


    def _B(self,t):
        """Return the LIP foot-placement input matrix.

        :param t: Propagation time.
        :type t: float
        :rtype: numpy.ndarray
        """
        chbt = np.cosh(self.beta*t)
        bshbt = self.beta*np.sinh(self.beta*t)
        return np.array([[1-chbt,0],
                        [0,1-chbt],
                        [-bshbt,0],
                        [0,-bshbt]])


    def _addDynamicsConstraint(self):
        """Add discrete LIP dynamics and initial-state constraints."""
        for k in range(self.N):
            self.opti.subject_to(self.x[:,k+1]==self.A@self.x[:,k]+self.B@self.pfst[:,k])
        self.opti.subject_to(self.x[:,0]==self.x0)
        

    def _addReachabilityConstraints(self):
        """Add alternating-foot reachability and COM-travel constraints."""
        lfs = True #True left foot, false rightfoot
        for k in range(self.N):
            dpx = self.x[0,k+1]-self.x[0,k]
            dpy = self.x[1,k+1]-self.x[1,k]

            dcom = cs.sqrt(dpx**2+dpy**2)
            sth = cs.sin(self.theta[0,k+1])
            cth = cs.cos(self.theta[0,k+1])
            
            self.opti.subject_to(self.opti.bounded(self.bx[0], cth*self.pfst[0,k]+sth*self.pfst[1,k], self.bx[1]))
            self.opti.subject_to(self.opti.bounded(self.dbcom[0]**2, dcom**2, self.dbcom[1]**2))
            if lfs:
                self.opti.subject_to(self.opti.bounded(self.by[0], -sth*self.pfst[0,k]+cth*self.pfst[1,k], self.by[1]))
            else:
                self.opti.subject_to(self.opti.bounded(-self.by[1], -sth*self.pfst[0,k]+cth*self.pfst[1,k], -self.by[0])) #mirrored

            lfs = not lfs #switch foot for next step


    # turning Constraint
    def _addTurningConstraint(self):
        """Add heading continuity and per-step turning bounds."""
        self.opti.subject_to(self.theta[:,0]==self.theta0)
        theta_max = np.pi
        dtheta_max = np.deg2rad(10)
        for k in range(self.N):
            self.opti.subject_to(self.opti.bounded(-dtheta_max, self.theta[0,k+1]-self.theta[0,k], dtheta_max))
            self.opti.subject_to(self.opti.bounded(-theta_max, self.theta[0,k+1], theta_max))
            
    #cbf
    def _ellipse(self,P,xc,xk):
        """Evaluate an ellipse quadratic form.

        :param P: Ellipse shape matrix.
        :param xc: Ellipse center.
        :param xk: Query point.
        :return: Quadratic-form value.
        """
        return (xk-xc).T@P@(xk-xc)

    #cbf poly
    def _addCBFPolytopesConstraint(self):
        """Add discrete CBF constraints for convex safe corridors."""
        self.opti.set_value(self.PA_cbf,np.zeros((self.nPol_sides,2)))
        self.opti.set_value(self.Pb_cbf,np.zeros((self.nPol_sides,)))
        for k in range(self.N):
            h_xk = self.Pb_cbf-self.PA_cbf@self.x[0:2,k] 
            h_xkp1 = self.Pb_cbf-self.PA_cbf@self.x[0:2,k+1]
            self.opti.subject_to(h_xkp1>=(1-self.alpha)*h_xk)

    
    # moving obstacle
    def SetMovingCBFEllipse(self,P_mobs,c_mobs):
        """Set moving-obstacle ellipse parameters.

        :param P_mobs: ``(2, 2)`` ellipse shape matrix.
        :type P_mobs: numpy.ndarray
        :param c_mobs: Centers with shape ``(2, N + 1)``.
        :type c_mobs: numpy.ndarray
        """
        assert c_mobs.shape[1] == self.N+1, "Check horizon and number of ellipses"
        assert P_mobs.shape == (2,2), "Number of ellipse matrix must be 2x2"
        self.opti.set_value(self.EP_mcbf,P_mobs)
        for i in range(self.N+1):
            self.opti.set_value(self.Ec_mcbf[:,i],c_mobs[:,i])
    
    def _addCBFMovingObstacle(self):
        """Add discrete CBF constraints outside a moving ellipse."""
        self.opti.set_value(self.EP_mcbf,10*np.eye(2))
        self.opti.set_value(self.Ec_mcbf,100*np.ones([2,self.N+1]))#keep it out of the way
        for k in range(self.N):
            h_xk = self._ellipse(self.EP_mcbf, self.Ec_mcbf[:,k], self.x[0:2,k])-1
            h_xkp1 = self._ellipse(self.EP_mcbf,self.Ec_mcbf[:,k+1], self.x[0:2,k+1])-1
            self.opti.subject_to(h_xkp1>=(1-0.15+0*self.alpha)*h_xk)


    def _formulateMPC(self):
        """Assemble the MPC constraints and objective."""
        # constraints
        self._addDynamicsConstraint()
        self._addReachabilityConstraints()
        self._addCBFPolytopesConstraint()
        self._addTurningConstraint()
        # self._addCBFMovingObstacle()
        
        #cost
        runningcost = 0
        runningcost_pos = 0
        inputcost = 0
        headingcost = 0
        terminal_heading = 0
        for i in range(self.N):
            err = self.x[2:4,i] -self.xgoal[2:4]
            runningcost += cs.dot(err,err)
            err = self.x[0:2,i] -self.xgoal[0:2]
            runningcost_pos += cs.dot(err,err)
            fst = self.pfst[:,i]
            inputcost += cs.dot(fst,fst)
            headingcost += cs.dot(self.theta[0,i+1]-self.theta[0,i],self.theta[0,i+1]-self.theta[0,i])
        TerminalError = self.x[0:2,self.N]-self.xgoal[0:2] #terminal error
        TerminalCost = cs.dot(TerminalError,TerminalError)
        terminal_heading = cs.dot(self.theta[0,self.N]-self.theta_goal,self.theta[0,self.N]-self.theta_goal)
        err = self.Ppfst_last-self.pfst[:,0]
        InputChange_cost = cs.dot(err,err)

        self.opti.minimize(0.1*InputChange_cost + 10*runningcost + 1*runningcost_pos + 5*TerminalCost + 50*inputcost + 100*headingcost + 0*terminal_heading)


    def Solve(self, x0, s, xgoal, lf_first, theta_heading, theta_goal, A, b, pfst_last=None): # lf_first leftfoot swing first
        """Solve the sequential footstep MPC.

        :param x0: Current LIP state.
        :param s: Current normalized step phase.
        :param xgoal: Desired LIP goal state.
        :param lf_first: Whether the left foot swings first.
        :param theta_heading: Current unwrapped heading.
        :param theta_goal: Desired terminal heading.
        :param A: Active safe-corridor half-space matrix.
        :param b: Active safe-corridor half-space vector.
        :param pfst_last: Current absolute stance-foot position.
        :return: Next absolute footstep and next heading.
        :rtype: tuple[numpy.ndarray, float]
        """
        x_end = self.predictTouchDownLIPState(x0, s, self.Tst, pfst_last)
        xgoal = self._truncategoal(x0, xgoal)
        self.opti.set_value(self.x0, x_end)
        self.opti.set_value(self.xgoal, xgoal)
        self.opti.set_value(self.theta0, theta_heading)
        self.opti.set_value(self.theta_goal, theta_goal)
        
        assert A.shape[0] <= self.nPol_sides, "Increase nPol_sides"
        A_full = np.zeros((self.nPol_sides, 2))
        b_full = np.zeros((self.nPol_sides, ))
        A_full[0:A.shape[0], :] = A
        b_full[0:A.shape[0]] = b.reshape((A.shape[0],))

        if pfst_last is not None:
            self.opti.set_value(self.Ppfst_last, pfst_last)
        
        
        if lf_first:
            self.opti.set_value(self.by, self.footboundy)
        else:
            self.opti.set_value(self.by, -self.footboundy[::-1])

        self.opti.set_value(self.bx, self.footboundx)
        self.opti.set_value(self.dbcom, self.distboundcom)
        
        for k in range(self.N):
            self.opti.set_initial(self.theta[0,k+1], theta_goal)
        

        if not self.isFirstRun:
            #guess x and u
            self.opti.set_initial(self.x[:,0], x0) #initial guess is same as x0
            self.opti.set_initial(self.theta[:,0], theta_heading)
            self.opti.set_initial(self.theta[:,1:-1], self.solution.value(self.theta)[2::])

            self.opti.set_initial(self.x[:,1:-1], self.solution.value(self.x)[:,2::]) #1..N init as last 2..N+1
            self.opti.set_initial(self.pfst[:,:-1], self.solution.value(self.pfst)[:,1::]) #0..N-1 init as last 1..N
            pfst_N_guess=self.K@(xgoal-self.solution.value(self.x)[:,-1].reshape((4,1))) #
            x_Np1_guess = self.A@self.solution.value(self.x)[:,-1].reshape((4,1)) + self.B@pfst_N_guess
            self.opti.set_initial(self.x[:,-1], x_Np1_guess)
            self.opti.set_initial(self.pfst[:,-1], pfst_N_guess)
            
        else:
            xk = x0
            self.opti.set_initial(self.x[:,0], x0)
            for k in range(self.N):
                pfst_k_guess=self.K@(xgoal-xk)
                x_kp1_guess = self.A@xk + self.B@pfst_k_guess
                self.opti.set_initial(self.x[:,k+1], x_kp1_guess)
                self.opti.set_initial(self.pfst[:,k], pfst_k_guess)
                xk = x_kp1_guess
                self.opti.set_initial(self.theta[0,k+1], theta_heading) #set same as initial
            self.isFirstRun = False
    
        self.solution = self.opti.solve()
        u = self.solution.value(self.pfst)[:,0]
        nheading = self.solution.value(self.theta)[1]
        p_fst = x_end[0:2].reshape((2,1)) + u.reshape((2,1))
        return p_fst, nheading
