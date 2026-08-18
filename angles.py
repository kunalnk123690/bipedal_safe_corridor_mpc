# @author: Abhijeet Kulkarni

import numpy as np

class Angle:
    """Represent a wrapped planar angle using a unit complex number."""

    def __init__(self, theta=0.0):
        """Initialize an angle.

        :param theta: Angle in radians.
        :type theta: float
        """
        self.a = 1.0 # real part
        self.b = 0.0 # imaginary part
        self._from_theta_to_complex(theta)


    def _from_theta_to_complex(self, theta):
        """Set the complex representation from radians.

        :param theta: Angle in radians.
        :type theta: float
        """
        self.a = np.cos(theta)
        self.b = np.sin(theta)
    
    def _normalize(self):
        """Normalize the internal complex number to unit magnitude."""
        norm = np.sqrt(self.a**2 + self.b**2)
        self.a = self.a/norm
        self.b = self.b/norm

    @property
    def toRadian(self):
        """Return the wrapped angle in radians.

        :rtype: float
        """
        return np.arctan2(self.b, self.a)
    
    @property
    def toDegree(self):
        """Return the wrapped angle in degrees.

        :rtype: float
        """
        return np.rad2deg(self.toRadian)
    
    def __add__(self, other):
        """Add another angle or a scalar in radians.

        :param other: Value to add.
        :type other: Angle or float
        :return: Wrapped sum.
        :rtype: Angle
        """
        if isinstance(other, Angle):
            return Angle(self.toRadian + other.toRadian)
        elif isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian + other)
        else:
            raise TypeError("unsupported operand type(s) for +: 'Angle' and '{}'".format(type(other)))
    
    def __sub__(self, other):
        """Subtract another angle or a scalar in radians.

        :param other: Value to subtract.
        :type other: Angle or float
        :return: Wrapped difference.
        :rtype: Angle
        """
        if isinstance(other, Angle):
            return Angle(self.toRadian - other.toRadian)
        elif isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian - other)
        else:
            raise TypeError("unsupported operand type(s) for -: 'Angle' and '{}'".format(type(other)))
    
    def __mul__(self, other):
        """Scale this angle.

        :param other: Scalar multiplier.
        :type other: float
        :return: Scaled wrapped angle.
        :rtype: Angle
        """
        if isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian * other)
        else:
            raise TypeError("unsupported operand type(s) for *: 'Angle' and '{}'".format(type(other)))
    
    def __truediv__(self, other):
        """Divide this angle by a scalar.

        :param other: Scalar divisor.
        :type other: float
        :return: Scaled wrapped angle.
        :rtype: Angle
        """
        if isinstance(other, float) or isinstance(other, int):
            return Angle(self.toRadian / other)
        else:
            raise TypeError("unsupported operand type(s) for /: 'Angle' and '{}'".format(type(other)))
    
    def __repr__(self):
        """Return an unambiguous representation.

        :rtype: str
        """
        return "Angle({})".format(self.toRadian)

    def __str__(self):
        """Return a readable representation.

        :rtype: str
        """
        return "Angle({})".format(self.toRadian)
    
