"""
Written by: Abhijeet Kulkarni (amkulk@udel.edu)
"""
import mujoco
import numpy as np
from pyquaternion import quaternion

def output(data=None):
        """Extract the reduced Digit state used by the controller.

        :param data: MuJoCo simulation data.
        :type data: mujoco.MjData
        :return: Configuration, velocity, and simulation time.
        :rtype: tuple[numpy.ndarray, numpy.ndarray, float]
        """
        q_i = [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 14, 15, 16, 17, 18, 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 41, 42, 43, 44, 45, 50, 55, 56, 57, 58, 59, 60]
        dq_i = [0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 13, 14, 15, 16, 20, 24, 25, 26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 44, 48, 49, 50, 51, 52, 53]
        q = data.qpos[q_i].copy()
        v = data.qvel[dq_i].copy()

        return q, v, data.time

def _rotation_matrix_from_vectors(vec1, vec2): #https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """Return the rotation matrix that aligns one vector with another.

    :param vec1: Three-dimensional source vector.
    :type vec1: numpy.ndarray
    :param vec2: Three-dimensional destination vector.
    :type vec2: numpy.ndarray
    :return: Matrix that rotates ``vec1`` into alignment with ``vec2``.
    :rtype: numpy.ndarray
    """
    nvec1,nvec2 = np.linalg.norm(vec1),np.linalg.norm(vec2)
    if np.isclose(nvec1,0) or np.isclose(nvec2,0):
        return np.eye(3)
    a, b = (vec1 / nvec1).reshape(3), (vec2 / nvec2).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if np.isclose(s,0):
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def plot3d(mjvobj,pts3d,rgba=np.ones(4),scale_SPH=None):
    """Draw points as a polyline or a sequence of scaled spheres.

    :param mjvobj: MuJoCo viewer marker interface.
    :type mjvobj: object
    :param pts3d: Three-dimensional points with shape ``(n, 3)``.
    :type pts3d: numpy.ndarray
    :param rgba: Marker color.
    :type rgba: numpy.ndarray
    :param scale_SPH: Sphere radii, or ``None`` to draw a polyline.
    :type scale_SPH: numpy.ndarray or None
    """
    if scale_SPH is not None:
        _plot3d_SPH(mjvobj,pts3d,rgba,scale_SPH)
    else:
        _plot3d_LINE(mjvobj,pts3d,rgba)

def _plot3d_LINE(mjvobj,pts3d,rgba):
    """Draw connected 3-D line segments.

    :param mjvobj: Viewer marker interface.
    :param pts3d: Points with shape ``(n, 3)``.
    :param rgba: Marker color.
    """
    n = pts3d.shape[0]
    for i in range(n-1):
        p0 = pts3d[i,:]
        p1 = pts3d[i+1,:]
        d = np.linalg.norm(p0-p1)
        mjvobj.add_marker(type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.array([0,0,d]),
                    mat=_rotation_matrix_from_vectors(np.array([0,0,1]),p1-p0),
                    pos=p0,
                    rgba=rgba)

def _plot3d_SPH(mjvobj,pts3d,rgba,scale_SPH):
    """Draw a sphere at each 3-D point.

    :param mjvobj: Viewer marker interface.
    :param pts3d: Points with shape ``(n, 3)``.
    :param rgba: Marker color.
    :param scale_SPH: Sphere radii.
    """
    n = pts3d.shape[0]
    for i in range(n):
        p0 = pts3d[i,:]
        mjvobj.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=scale_SPH,
                    pos=p0,
                    rgba=rgba)

def _plot3d_CYL(mjvobj,pts3d,rgba,scale_SPH):
    """Draw a cylinder at each 3-D point.

    :param mjvobj: Viewer marker interface.
    :param pts3d: Points with shape ``(n, 3)``.
    :param rgba: Marker color.
    :param scale_SPH: Cylinder dimensions.
    """
    n = pts3d.shape[0]
    for i in range(n):
        p0 = pts3d[i,:]
        mjvobj.add_marker(type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=scale_SPH,
                    pos=p0,
                    rgba=rgba)

def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """
    return np.array([[0, -vector[2], vector[1]], 
                     [vector[2], 0, -vector[0]], 
                     [-vector[1], vector[0], 0]])

def QuatOriErrorbase(desQuat,curQuat):
        """Compute a sign-consistent quaternion vector error.

        :param desQuat: Desired scalar-first quaternion.
        :param curQuat: Current scalar-first quaternion.
        :return: Three-dimensional orientation error.
        :rtype: numpy.ndarray
        """
        e = desQuat[0]*curQuat[1::] - curQuat[0]*desQuat[1::] + skew(desQuat[1::])@curQuat[1::]
        # print("scalar part",desQuat[0]*curQuat[0] - np.dot(desQuat[1::],curQuat[1::]))
        e[0:2] = e[0:2]*np.sign(desQuat[0]*curQuat[0] - np.dot(desQuat[1::],curQuat[1::]))
        return e

# def QuatOriError(desQuat,curQuat):
#     q_d = quaternion.Quaternion(desQuat)
#     R_d = q_d.rotation_matrix
#     q_c = quaternion.Quaternion(curQuat)
#     R_c = q_c.rotation_matrix
#     R_ed = R_c.T@R_d
#     q_ed = quaternion.Quaternion(matrix=R_ed)
#     return -R_c@q_ed.vector
def QuatOriError(desQuat,curQuat):
    """Compute a world-frame orientation error.

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
    """Compute an unrotated orientation error.

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

def AngularVelFromQuatVel(q,dq_dt):#Quaternions And Dynamics Graf
    """Convert quaternion rate to angular velocity.

    :param q: Scalar-first unit quaternion.
    :param dq_dt: Quaternion time derivative.
    :return: Angular velocity.
    :rtype: numpy.ndarray
    """
    E = np.array([[-q[1], q[0], -q[3], q[2]],
                  [-q[2], q[3], q[0], -q[1]],
                  [-q[3], -q[2], q[1], q[0]]])
    return 2*E@dq_dt.reshape((4,1))

def AngVelAccfromQuat(q,dq_dt,ddq_ddt):
    """Convert quaternion derivatives to angular velocity and acceleration.

    :param q: Scalar-first unit quaternion.
    :param dq_dt: Quaternion velocity.
    :param ddq_ddt: Quaternion acceleration.
    :return: Angular velocity and angular acceleration.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    omega = 2*QuatMulMatrix(dq_dt,QuatConjUnit(q))
    domega = 2*QuatMulMatrix(ddq_ddt,QuatConjUnit(q)) - 0.5* QuatMulMatrix(omega,omega)
    return omega[1:4],domega[1:4]

def QuatMulMatrix(q,q1):
    """Multiply two scalar-first quaternions.

    :param q: Left quaternion.
    :param q1: Right quaternion.
    :return: Quaternion product.
    :rtype: numpy.ndarray
    """
    q = q.reshape(4,)
    q1 = q1.reshape((4,1))
    return np.array([[q[0], -q[1], -q[2], -q[3]],
                    [q[1],  q[0], -q[3],  q[2]],
                    [q[2],  q[3],  q[0], -q[1]],
                    [q[3], -q[2],  q[1],  q[0]]])@q1
def QuatConjUnit(q): #quaternion conj of unit quaternion
    """Conjugate a unit quaternion in place.

    :param q: Scalar-first quaternion.
    :return: The mutated conjugate.
    :rtype: numpy.ndarray
    """
    q[1:4] = -q[1:4]
    return q

def SlerpwithDerivative(q1,q2,u,du_dt): #from shoemake'85
    """Interpolate quaternions and compute time derivatives.

    :param q1: Initial scalar-first quaternion.
    :param q2: Final scalar-first quaternion.
    :param u: Interpolation phase.
    :param du_dt: Phase time derivative.
    :return: Quaternion, velocity, and acceleration.
    :rtype: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """
    #q1 and q2 are numpy array
    cth = np.dot(q1,q2)
    # if np.abs(cth)>0.995:
    #     qu = q1 + u*(q2-q1)
    #     dqu = (q2-q1)*du_dt
    #     print('Using Local Approximation')
    #     return qu,dqu,np.zeros_like(dqu)
    theta = np.arccos(cth)
    sth = np.sin(theta)
    qu = (np.sin(theta-u*theta)*q1 + np.sin(u*theta)*q2)/sth
    dqu = (-np.cos(theta-u*theta)*q1+np.cos(u*theta)*q2)*theta*du_dt/sth
    ddqu = (-np.sin(theta-u*theta)*q1-np.sin(u*theta)*q2)*((theta*du_dt)**2)/sth
    return qu,dqu,ddqu

def InterpolateYawInitToFinalQuaternion(theta_init, theta_final, s):
    """Interpolate between two yaw angles as a quaternion.

    :param theta_init: Initial yaw.
    :param theta_final: Final yaw.
    :param s: Interpolation phase.
    :return: Scalar-first yaw quaternion.
    :rtype: numpy.ndarray
    """
    theta_cur = theta_init + s*(theta_final - theta_init)
    q = np.array([np.cos(theta_cur/2), 0, 0, np.sin(theta_cur/2)])
    return q

def _rotMz(theta):
    """Return a rotation matrix about the z axis.

    :param theta: Yaw angle in radians.
    :rtype: numpy.ndarray
    """
    return np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])

def addFootSteps_MAP(mjvobj,c,phi,width=0.08,length=0.2,height=0.02,rgba=np.array([1,0,0,0.3])):
    """Add a footstep box marker to a MuJoCo viewer.

    :param mjvobj: MuJoCo viewer marker interface.
    :type mjvobj: object
    :param c: Planar footstep center.
    :type c: numpy.ndarray
    :param phi: Foot yaw in radians.
    :type phi: float
    :param width: Foot marker width.
    :type width: float
    :param length: Foot marker length.
    :type length: float
    :param height: Foot marker height.
    :type height: float
    :param rgba: Marker color.
    :type rgba: numpy.ndarray
    """
    # n_fst = c.shape[0]
    # for i in range(n_fst):
    mjvobj.add_marker(type=mujoco.mjtGeom.mjGEOM_BOX,
                        pos=np.array([c[0],c[1],height/2]),
                        size=np.array([length/2,width/2,height/2]),
                        mat=_rotMz(phi),
                        rgba=rgba)
    
def addCuboidObs_MAP(mjvobj,c,r,H,rgba=np.array([0.59, 0.43, 0.2,1])): #for rectangles 2d
    """Add cuboid obstacle markers to a MuJoCo viewer.

    The planar rectangles circumscribe the corresponding obstacle ellipses.

    :param mjvobj: MuJoCo viewer marker interface.
    :type mjvobj: object
    :param c: Planar obstacle centers with shape ``(n, 2)``.
    :type c: numpy.ndarray
    :param r: Planar half extents with shape ``(n, 2)``.
    :type r: numpy.ndarray
    :param H: Obstacle heights with shape ``(n, 1)``.
    :type H: numpy.ndarray
    :param rgba: Marker color.
    :type rgba: numpy.ndarray
    """
    n_obs = c.shape[0]
    for i in range(n_obs):
        mjvobj.add_marker(type=mujoco.mjtGeom.mjGEOM_BOX,
                            pos=np.array([c[i,0],c[i,1],H[i,0]/2]),
                            size=np.array([r[i,0],r[i,1],H[i,0]/2]),
                            rgba=rgba)


from angles import *
def getUnwrappedThetatoLast(theta,theta_last):
    """Unwrap an angle relative to the preceding value.

    :param theta: New wrapped angle.
    :param theta_last: Previous unwrapped angle.
    :return: Nearest equivalent unwrapped angle.
    :rtype: float
    """
    diff_theta = (Angle(theta) - Angle(theta_last))
    return theta_last + diff_theta.toRadian
