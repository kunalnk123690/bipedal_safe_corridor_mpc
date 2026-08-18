import scipy.io as sio
from os.path import join as pjoin
from scipy.spatial import ConvexHull
import numpy as np



class MAP():
    """Load and track obstacles, goals, and convex safe corridors."""

    def __init__(self,map_dir,obs_map_fname='obstacle_pts.mat',goal_map_fname='goal.mat',poly_map_fname='polytopes.mat',xlim=(0,50),ylim=(0,50)):
        """Initialize a map from MATLAB data files.

        :param map_dir: Directory containing the map files.
        :type map_dir: str
        :param obs_map_fname: Obstacle filename.
        :type obs_map_fname: str
        :param goal_map_fname: Goal filename.
        :type goal_map_fname: str
        :param poly_map_fname: Safe-polytope filename.
        :type poly_map_fname: str
        :param xlim: Display limits in x.
        :type xlim: tuple[float, float]
        :param ylim: Display limits in y.
        :type ylim: tuple[float, float]
        """
        self.obs_map_fname = pjoin(map_dir,obs_map_fname)
        self.goal_map_fname = pjoin(map_dir,goal_map_fname)
        self.poly_map_fname = pjoin(map_dir,poly_map_fname)
        self.obs_corner = self._getObstacles()
        self.polytopesA,self.polytopesb = self._getPolytopes()
        self.startPxy,self.goalsPxy = self._getGoals()
        self.xlim = xlim
        self.ylim = ylim
        self.n_obs = len(self.obs_corner)
        self.n_pol = len(self.polytopesA)
        self.n_goal = len(self.goalsPxy)
        assert self.n_pol==self.n_goal, "Number of goals should be equal to number of polytopes"

        #moving obs
        self.startPxy_mobs = None
        self.endPxy_mobs = None
        self.time_mobs = None #time between obstacles
        self.r_mobs = None
        self.t_mobs_start = None
        # for mpc
        self.cur_goal_idx = 0
        self.cur_poly_idx = 0
    #    
    def _getObstacles(self):
        """Load rectangular obstacle corner points.

        Corners are returned counterclockwise beginning at the lower left.

        :return: One ``(2, 4)`` corner array per obstacle.
        :rtype: list[numpy.ndarray]
        """
        mat_contents = sio.loadmat(self.obs_map_fname)
        data = mat_contents['obstacle_pts']
        n_obs = data.shape[2]
        # n_obs= len(data)
        obs_corner=[]
        for i in range(n_obs):
            ch= ConvexHull(data[:,:,i].T)
            pts_cnr = np.concatenate((ch.points[np.where(ch.vertices==0)].T,
                                    ch.points[np.where(ch.vertices==3)].T,
                                    ch.points[np.where(ch.vertices==2)].T,
                                    ch.points[np.where(ch.vertices==1)].T),1)
            pts_cnr = self._reorderObstaclePoints(pts_cnr)
            obs_corner.append(pts_cnr)
        return obs_corner
    #
    def _reorderObstaclePoints(self,pts):
        """Put rectangle corners into the map's canonical order.

        :param pts: Unordered rectangle corners with shape ``(2, 4)``.
        :type pts: numpy.ndarray
        :return: Ordered rectangle corners.
        :rtype: numpy.ndarray
        """
        pts_reordered = np.zeros_like(pts)
        #0
        pts_reordered[:,0] = pts[:,np.intersect1d(np.where(np.isclose(pts[1,:],np.min(pts,1)[1])),np.where(np.isclose(pts[0,:],np.min(pts,1)[0])))].reshape((2,))
        #1
        pts_reordered[:,1] = pts[:,np.intersect1d(np.where(np.isclose(pts[1,:],np.max(pts,1)[1])),np.where(np.isclose(pts[0,:],np.min(pts,1)[0])))].reshape((2,))
        #2
        pts_reordered[:,2] = pts[:,np.intersect1d(np.where(np.isclose(pts[1,:],np.max(pts,1)[1])),np.where(np.isclose(pts[0,:],np.max(pts,1)[0])))].reshape((2,))
        #3
        pts_reordered[:,3] = pts[:,np.intersect1d(np.where(np.isclose(pts[1,:],np.min(pts,1)[1])),np.where(np.isclose(pts[0,:],np.max(pts,1)[0])))].reshape((2,))
        return pts_reordered




    def _getPolytopes(self):
        """Load half-space representations of the safe corridors.

        :return: Lists of matrices and vectors satisfying ``A @ x <= b``.
        :rtype: tuple[list[numpy.ndarray], list[numpy.ndarray]]
        """
        mat_contents = sio.loadmat(self.poly_map_fname)
        A_appended = mat_contents['A_appended']
        b_appended = mat_contents['b_appended']
        idx = mat_contents['idx'].squeeze()
        A = []
        b = []
        for i in range(idx.shape[0]-1):
            A.append(A_appended[idx[i]:idx[i+1]])
            b.append(b_appended[idx[i]:idx[i+1]])
        return A,b
    #
    def _getGoals(self):
        """Load the initial position and sequential corridor goals.

        :return: Initial position and list of goal positions.
        :rtype: tuple[numpy.ndarray, list[numpy.ndarray]]
        """
        mat_contents = sio.loadmat(self.goal_map_fname)
        # px_appended = mat_contents['px_appended'].squeeze().reshape(1,)
        # py_appended = mat_contents['py_appended'].squeeze().reshape(1,)
        # p_initial = mat_contents['p_initial'].reshape((2,))
        px_appended = mat_contents['px_appended'].squeeze()
        py_appended = mat_contents['py_appended'].squeeze()
        p_initial = mat_contents['p_initial'].reshape((2,))
        goalsPxy = []
        for i in range(px_appended.shape[0]):
            goalsPxy.append(np.array([px_appended[i],py_appended[i]]))
        return p_initial,goalsPxy
    

    
    def getObsCenterRadi(self):
        """Return centers and deflated half extents of rectangle obstacles.

        :return: Center and radius arrays, each with shape ``(n, 2)``.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        c_obs = np.zeros((self.n_obs,2))
        r_obs = np.zeros((self.n_obs,2))
        for i in range(self.n_obs):
            w = self.obs_corner[i][0,3]-self.obs_corner[i][0,0]
            h = self.obs_corner[i][1,1]-self.obs_corner[i][1,0]
            xy = self.obs_corner[i][:,0] #corner 0xy
            c_obs[i,:] = xy+np.array([w/2,h/2])
            r_obs[i,:] = 0.9*np.array([w/2,h/2]) #deflate
        return c_obs,r_obs

    
    # def addGoalPoints(self,ax):
    #     """adds goal points and start point on ax
    #     par1 ax: axis of plot"""
    #     plt.plot(self.startPxy[0],self.startPxy[1],marker='o')
    #     n_goals = len(self.goalsPxy)
    #     for i in range(n_goals):
    #         plt.plot(self.goalsPxy[i][0],self.goalsPxy[i][1],marker='s')
    
    def checkinPoly(self,Pointxy,idx):
        """Test whether a point lies in a selected safe polytope.

        :param Pointxy: Planar point.
        :type Pointxy: numpy.ndarray
        :param idx: Polytope index.
        :type idx: int
        :return: True when all half-space inequalities hold.
        :rtype: bool
        """
        A = self.polytopesA[idx]
        b = self.polytopesb[idx]
        Pointxy = Pointxy.reshape((2,1))

        return np.all(A@Pointxy<=b)
    
    def incrementGoalandPoly(self):
        """Advance to the next safe corridor and its associated goal."""
        self.cur_poly_idx +=1
        self.cur_goal_idx +=1
    
    @property
    def A(self):
        """Return the active corridor's half-space matrix.

        :rtype: numpy.ndarray
        """
        return self.polytopesA[self.cur_poly_idx]
    @property
    def b(self):
        """Return the active corridor's half-space vector.

        :rtype: numpy.ndarray
        """
        return self.polytopesb[self.cur_poly_idx]
    @property
    def goal(self):
        """Return the active intermediate goal.

        :rtype: numpy.ndarray
        """
        return self.goalsPxy[self.cur_goal_idx]

    # Moving Obstacle

    def addMovingObstacle(self,StartPosxy,EndPosxy,Speed,r,starttime):
        """Configure a constant-speed circular moving obstacle.

        :param StartPosxy: Initial planar position.
        :type StartPosxy: numpy.ndarray
        :param EndPosxy: Final planar position.
        :type EndPosxy: numpy.ndarray
        :param Speed: Translational speed.
        :type Speed: float
        :param r: Circle radius.
        :type r: float
        :param starttime: Motion start time.
        :type starttime: float
        """

        self.startPxy_mobs = StartPosxy
        self.endPxy_mobs = EndPosxy
        self.time_mobs = np.linalg.norm(StartPosxy-EndPosxy)/Speed
        self.r_mobs = r
        self.t_mobs_start = starttime
        
    def getMovingObsEllipse(self,curtime,dTfuture=0,nfuture=0):
        """Predict the moving obstacle's ellipse over a time horizon.

        :param curtime: Current absolute time.
        :type curtime: float
        :param dTfuture: Time between predictions.
        :type dTfuture: float
        :param nfuture: Number of future intervals.
        :type nfuture: int
        :return: Ellipse matrix and predicted centers.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        P = np.array([[1/self.r_mobs**2,0],[0,1/self.r_mobs**2]])
        s = np.min([np.max([(curtime-self.t_mobs_start)/self.time_mobs,0]),1])
        c = np.zeros((2,nfuture+1))
        for i in range(nfuture+1):
            s = np.min([np.max([(curtime+dTfuture*nfuture-self.t_mobs_start)/self.time_mobs,0]),1])
            c[:,i] = self.startPxy_mobs + (self.endPxy_mobs-self.startPxy_mobs)*s
        return P,c
