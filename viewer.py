import mujoco
import mujoco.viewer
import numpy as np


class _ViewerShim:
    """
    Compatibility shim exposing the subset of the old mujoco_viewer API that
    main.py touches directly on the `.viewer` attribute:

        mjviewer.viewer.is_alive       -> bool
        mjviewer.viewer._paused = True -> stored as a soft flag
        mjviewer.viewer.close()        -> request the viewer to shut down

    The official passive viewer runs its own GL thread. Explicitly calling
    passive.close() from the main thread on Linux/X11 can race with the GL
    thread's own context destruction and produce GLXBadContext errors on
    program exit. We therefore expose close() as a best-effort request and
    also make the main thread drop its reference to the passive handle so
    the GL thread is the only one that ends up tearing the context down.
    """

    def __init__(self, passive_handle):
        """Wrap an official MuJoCo passive-viewer handle.

        :param passive_handle: Handle returned by ``launch_passive``.
        """
        self._passive = passive_handle
        self._paused = False

    @property
    def is_alive(self):
        """Report whether the passive viewer is running.

        :rtype: bool
        """
        p = self._passive
        if p is None:
            return False
        try:
            return p.is_running()
        except Exception:
            return False

    def close(self):
        """Request viewer shutdown without propagating GL teardown errors."""
        # Best-effort: ask the viewer to close, then drop our reference so
        # we don't double-free the GL context at interpreter shutdown.
        p = self._passive
        self._passive = None
        if p is None:
            return
        try:
            p.close()
        except Exception:
            pass


class viewerObject():
    """
    Drop-in replacement for the previous mujoco_viewer-based viewer, using
    the official mujoco.viewer.launch_passive API.

    User-drawn markers (obstacles, goals, footsteps, arrows, boundaries) are
    written directly into viewer.user_scn.geoms each frame. user_scn.ngeom is
    reset at the start of every render() so the marker buffer does not grow
    unbounded -- this matches the per-frame add_marker behavior of the old
    viewer.

    Public surface (is_alive, _paused, render(), close(), add* methods) is
    preserved so main.py does not need to change its control flow.
    """

    _MAX_USER_GEOMS = 1000  # stay well under user_scn capacity

    def __init__(self, model, data):
        """Launch a passive viewer for a MuJoCo model.

        :param model: MuJoCo model.
        :type model: mujoco.MjModel
        :param data: MuJoCo simulation data.
        :type data: mujoco.MjData
        """
        self.model = model
        self.data = data

        # launch_passive is non-blocking; main.py owns physics stepping and
        # calls sync() each frame via render().
        self._passive = mujoco.viewer.launch_passive(
            model, data,
            show_left_ui=False,
            show_right_ui=False,
        )

        # Mirror the old fixed-camera selection when available.
        try:
            if model.ncam > 2:
                self._passive.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                self._passive.cam.fixedcamid = 2
        except Exception:
            pass

        # Shim exposes is_alive / _paused / close on `self.viewer`.
        self.viewer = _ViewerShim(self._passive)

        # Per-frame queue of marker specs; flushed in render().
        self._pending_markers = []

        # Legacy persistent-marker slots (render() auto-queues them if set).
        self.c_obs = None
        self.r_obs = None
        self.H_obs = None
        self.c_goal = None
        self.r_goal = None
        self.H_goal = None
        self.xlim = None
        self.ylim = None

    # ------------------------------------------------------------------
    # Marker registration
    # ------------------------------------------------------------------
    def _queue(self, spec):
        """Queue one user-scene geometry specification.

        :param spec: Geometry fields accepted by ``mjv_initGeom``.
        :type spec: dict
        """
        if len(self._pending_markers) < self._MAX_USER_GEOMS:
            self._pending_markers.append(spec)

    def addFootSteps_MAP(self, c, phi, width=0.08, length=0.2, height=0.02,
                         rgba=np.array([1, 0, 0, 0.3])):
        """Queue a footstep box marker.

        :param c: Planar center.
        :param phi: Foot yaw.
        :param width: Box width.
        :param length: Box length.
        :param height: Box height.
        :param rgba: Marker color.
        """
        self._queue({
            'type': mujoco.mjtGeom.mjGEOM_BOX,
            'pos': np.array([c[0], c[1], height / 2], dtype=np.float64),
            'size': np.array([length / 2, width / 2, height / 2], dtype=np.float64),
            'mat': self._rotMz(phi),
            'rgba': np.asarray(rgba, dtype=np.float32),
        })

    def addCuboidObs_MAP(self, c, r, H, rgba=np.array([0.59, 0.43, 0.2, 1])):
        """Queue axis-aligned cuboid obstacles.

        :param c: Centers with shape ``(n, 2)``.
        :param r: Half extents with shape ``(n, 2)``.
        :param H: Heights with shape ``(n, 1)`` or a scalar.
        :param rgba: Marker color.
        """
        n_obs = c.shape[0]
        H_is_array = not np.isscalar(H)
        if H_is_array:
            H_arr = np.asarray(H).reshape(-1, 1)
        for i in range(n_obs):
            h_i = float(H_arr[i, 0]) if H_is_array else float(H)
            self._queue({
                'type': mujoco.mjtGeom.mjGEOM_BOX,
                'pos': np.array([c[i, 0], c[i, 1], h_i / 2], dtype=np.float64),
                'size': np.array([r[i, 0], r[i, 1], h_i / 2], dtype=np.float64),
                'mat': np.eye(3),
                'rgba': np.asarray(rgba, dtype=np.float32),
            })

    def addCylindergoal_MAP(self, c, r, H, rgba=np.array([1, 0, 1, 0.1])):
        """Queue cylindrical goal markers.

        :param c: Centers with shape ``(n, 2)``.
        :param r: Radii with shape ``(n, 2)``.
        :param H: Heights with shape ``(n, 1)`` or a scalar.
        :param rgba: Marker color.
        """
        n_obs = c.shape[0]
        H_is_array = not np.isscalar(H)
        if H_is_array:
            H_arr = np.asarray(H).reshape(-1, 1)
        for i in range(n_obs):
            h_i = float(H_arr[i, 0]) if H_is_array else float(H)
            self._queue({
                'type': mujoco.mjtGeom.mjGEOM_CYLINDER,
                'pos': np.array([c[i, 0], c[i, 1], h_i / 2], dtype=np.float64),
                'size': np.array([r[i, 0], r[i, 0], h_i / 2], dtype=np.float64),
                'mat': np.eye(3),
                'rgba': np.asarray(rgba, dtype=np.float32),
            })

    def addArrow_MAP(self, p0, p1, r, rgba=np.array([1, 0, 0, 1])):
        """Queue an arrow marker.

        :param p0: Arrow start point.
        :param p1: Arrow end point.
        :param r: Shaft radius.
        :param rgba: Marker color.
        """
        p0 = np.asarray(p0, dtype=np.float64).reshape(3,)
        p1 = np.asarray(p1, dtype=np.float64).reshape(3,)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-9:
            return
        self._queue({
            'type': mujoco.mjtGeom.mjGEOM_ARROW,
            'pos': p0,
            'size': np.array([r, r, length], dtype=np.float64),
            'mat': self._rotation_matrix_from_vectors(np.array([0, 0, 1]), p1 - p0),
            'rgba': np.asarray(rgba, dtype=np.float32),
        })

    def _drawBoundary(self, xlim, ylim, thickness=0.2, height=0.5):
        """Queue the legacy rectangular boundary wall.

        :param xlim: Boundary x limits.
        :param ylim: Boundary y limits.
        :param thickness: Wall thickness.
        :param height: Wall height.
        """
        c01 = np.array([[xlim[0], (xlim[0] + ylim[1]) / 2]])
        r01 = np.array([[thickness / 2, ylim[1] / 2]])
        self.addCuboidObs_MAP(c01, r01, height, rgba=np.array([0, 0, 0, 1]))

    def addGoal(self, c, r, H):
        """Queue goal cylinders.

        :param c: Goal centers.
        :param r: Goal radii.
        :param H: Goal heights.
        """
        self.addCylindergoal_MAP(c, r, H)

    # ------------------------------------------------------------------
    # Rotation helpers
    # (The old _rotMz was written as an instance method but missing `self`;
    #  that was a latent bug. Both are staticmethods here.)
    # ------------------------------------------------------------------
    @staticmethod
    def _rotMz(theta):
        """Return a rotation matrix about the z axis.

        :param theta: Yaw angle.
        :rtype: numpy.ndarray
        """
        c, s = np.cos(theta), np.sin(theta)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    @staticmethod
    def _rotation_matrix_from_vectors(vec1, vec2):
        """Return a rotation aligning one 3-D vector with another.

        :param vec1: Source vector.
        :param vec2: Destination vector.
        :rtype: numpy.ndarray
        """
        nvec1, nvec2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if np.isclose(nvec1, 0) or np.isclose(nvec2, 0):
            return np.eye(3)
        a = (vec1 / nvec1).reshape(3)
        b = (vec2 / nvec2).reshape(3)
        v = np.cross(a, b)
        c = float(np.dot(a, b))
        s = float(np.linalg.norm(v))
        if np.isclose(s, 0):
            return np.eye(3) if c > 0 else -np.eye(3)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]], dtype=np.float64)
        return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------
    def _flush_markers(self):
        """Write queued markers into the MuJoCo user scene."""
        scn = self._passive.user_scn
        cap = scn.geoms.shape[0] if hasattr(scn.geoms, 'shape') else len(scn.geoms)

        n = 0
        for spec in self._pending_markers:
            if n >= cap:
                break
            g = scn.geoms[n]
            mujoco.mjv_initGeom(
                g,
                type=int(spec['type']),
                size=spec['size'],
                pos=spec['pos'],
                mat=np.asarray(spec['mat'], dtype=np.float64).flatten(),
                rgba=spec['rgba'],
            )
            n += 1

        scn.ngeom = n
        self._pending_markers.clear()

    def render(self):
        """Flush persistent and queued markers, then synchronize the viewer."""
        if self.c_obs is not None:
            self.addCuboidObs_MAP(self.c_obs, self.r_obs, self.H_obs)
        if self.c_goal is not None:
            self.addCylindergoal_MAP(self.c_goal, self.r_goal, self.H_goal)
        if self.xlim is not None:
            self._drawBoundary(self.xlim, self.ylim)

        if self._passive is None or not self._passive.is_running():
            return

        with self._passive.lock():
            self._flush_markers()
        self._passive.sync()

    def close(self):
        """Close the passive viewer, suppressing GL teardown errors."""
        # See _ViewerShim.close for rationale: do not force-destroy the
        # GL context from the main thread. Drop our reference and let the
        # viewer's own thread tear things down. This avoids GLXBadContext
        # on Linux/X11.
        p = self._passive
        self._passive = None
        if p is None:
            return
        try:
            p.close()
        except Exception:
            pass
