# Copyright (c) 2016, Nezar Abdennur
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import re
from matplotlib.path import Path
import matplotlib.transforms as transforms
from math import sin, cos, sqrt, degrees, radians, acos
import numpy as np


COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
UPPERCASE = set('MZLHVCSQTA')

COMMAND_RE = re.compile(r"([MmZzLlHhVvCcSsQqTtAa])")
FLOAT_RE = re.compile(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?")

COMMAND_CODES = {
    'M': (Path.MOVETO,),    # moveto
    'L': (Path.LINETO,),    # line
    'H': (Path.LINETO,),    # shorthand for horizontal line
    'V': (Path.LINETO,),    # shorthand for vertical line
    'Q': (Path.CURVE3,)*2,  # quadratic bezier
    'T': (Path.CURVE3,)*2,  # shorthand for smooth quadratic bezier
    'C': (Path.CURVE4,)*3,  # cubic bezier
    'S': (Path.CURVE4,)*3,  # shorthand for smooth cubic bezier
    'Z': (Path.CLOSEPOLY,), # closepath
    'A': None               # arc
}

PARAMS = {
    'M': 2, # moveto
    'L': 2, # line
    'H': 1, # shorthand for horizontal line
    'V': 1, # shorthand for vertical line
    'Q': 4, # quadratic bezier
    'T': 4, # shorthand for smooth quadratic bezier
    'C': 6, # cubic bezier
    'S': 6, # shorthand for smooth cubic bezier
    'Z': 0, # closepath
    'A': 7  # arc
}


def endpoint_to_center(start, radius, rotation, large, sweep, end):
    """
    Translates the "endpoint" parameterization of an elliptical arc used by
    the SVG spec to the "center" parameterization.
    Parameters
    ----------
    start : complex
        Starting point (x1, y1).
    radius : complex
        Two elliptical radii (rx, ry).
    rotation : float
        Angle from the x-axis of the current coordinate system to the x-axis of
        the ellipse.
    large : bool
        False if an arc spanning < 180 degrees is to be drawn, True if an arc
        spanning >= 180 degrees is to be drawn.
    sweep : bool
        If sweep-flag is True, then the arc will be drawn in a "positive-angle"
        direction from start to end.
    end : complex
        End point (x2, y2).
    Returns
    -------
    radius : complex
        Radii of the ellipse (rx, ry), potentially corrected if out-of-range.
    center : complex
        Center of the ellipse (xc, yc).
    theta1, theta2 : float
        Start and end angles of an arc on the unit circle prior to being
        stretched and rotated into an elliptical arc.
    Notes
    -----
    See [1]_, `svg.path <https://github.com/regebro/svg.path>`_, and this
    `SO question <https://stackoverflow.com/q/197649>`_.
    References
    ----------
    .. [1] http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
    """
    # Step 1: compute x1prim, y1prim
    cosr = cos(radians(rotation))
    sinr = sin(radians(rotation))
    dx = (start.real - end.real) / 2
    dy = (start.imag - end.imag) / 2
    x1prim = cosr * dx + sinr * dy
    y1prim = -sinr * dx + cosr * dy
    x1prim_sq = x1prim * x1prim
    y1prim_sq = y1prim * y1prim

    rx = abs(radius.real)
    ry = abs(radius.imag)
    rx_sq = rx * rx
    ry_sq = ry * ry

    # Correct out of range radii
    radius_scale = (x1prim_sq / rx_sq) + (y1prim_sq / ry_sq)
    if radius_scale > 1:
        radius_scale = sqrt(radius_scale)
        rx *= radius_scale
        ry *= radius_scale
        rx_sq = rx * rx
        ry_sq = ry * ry
    else:
        # SVG spec only scales UP
        radius_scale = 1
    radius = rx + ry * 1j

    # Step 2: compute cxprim, cyprim
    t1 = rx_sq * y1prim_sq
    t2 = ry_sq * x1prim_sq
    c = sqrt(abs((rx_sq * ry_sq - t1 - t2) / (t1 + t2)))
    if large == sweep:
        c = -c
    cxprim = c * rx * y1prim / ry
    cyprim = -c * ry * x1prim / rx

    # Step 3: compute the center from cxprim, cyprim
    center = complex(
        (cosr * cxprim - sinr * cyprim) + ((start.real + end.real) / 2),
        (sinr * cxprim + cosr * cyprim) + ((start.imag + end.imag) / 2),
    )

    # Step 4: compute theta and delta_theta
    ux = (x1prim - cxprim) / rx
    uy = (y1prim - cyprim) / ry
    vx = (-x1prim - cxprim) / rx
    vy = (-y1prim - cyprim) / ry
    n = sqrt(ux * ux + uy * uy)
    p = ux
    theta = degrees(acos(p / n))
    if uy < 0:
        theta = -theta
    theta = theta % 360

    n = sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
    p = ux * vx + uy * vy
    d = p / n
    # In certain cases the above calculation can through inaccuracies
    # become just slightly out of range, f ex -1.0000000000000002.
    d = np.clip(d, -1.0, 1.0)

    delta = degrees(acos(d))
    if (ux * vy - uy * vx) < 0:
        delta = -delta
    delta = delta % 360
    if not sweep:
        if delta > 0:
            delta -= 360

    return radius, center, theta, theta + delta


def arc_path(start, radius, rotation, large, sweep, end):
    """
    Generate an elliptical arc path given an endpoint parameterization.
    Uses matplotlib to draw the arc using quadratic Bezier curves.
    Parameters
    ----------
    start : complex
        Starting point (x1, y1).
    radius : complex
        Two elliptical radii (rx, ry).
    rotation : float
        Angle from the x-axis of the current coordinate system to the x-axis of
        the ellipse.
    large : bool
        False if an arc spanning < 180 degrees is to be drawn, True if an arc
        spanning >= 180 degrees is to be drawn.
    sweep : bool
        If sweep-flag is True, then the arc will be drawn in a "positive-angle"
        direction from start to end.
    end : complex
        End point (x2, y2).
    Returns
    -------
    codes : array (n,)
        Command codes
    verts : array (n,2)
        Vertices
    Notes
    -----
    We first perform a conversion to center parameterization and generate
    a circular arc at the origin. Then we apply scaling, translation and
    rotation.
    One can think of an ellipse as a circle that has been stretched and then
    rotated. Start by making an arc along the unit circle from `theta1` to
    `theta2`, centered at `center`. Then scale the circle along the x and y
    axes according to the given radii. Finally, rotate the arc around the
    center through the given angle `rotation`.
    """
    radius, center, theta1, theta2 = endpoint_to_center(
        start,
        radius,
        rotation,
        large,
        sweep,
        end
    )

    # Create an arc on the unit circle
    # Matplotlib does this using CURVE4 operations
    # https://matplotlib.org/stable/_modules/matplotlib/path.html#Path.arc
    if theta2 > theta1:
        arc = Path.arc(theta1=theta1, theta2=theta2)
        reverse_path = False
    else:
        arc = Path.arc(theta1=theta2, theta2=theta1)
        reverse_path = True

    # Transform it into an elliptical arc:
    # * scale the minor and major axes
    # * translate it to the center
    # * rotate the x-axis of the ellipse from the x-axis of the current
    #   coordinate system
    trans = (
        transforms.Affine2D()
        .scale(radius.real, radius.imag)
        .translate(center.real, center.imag)
        .rotate_deg_around(center.real, center.imag, rotation)
    )
    arc = trans.transform_path(arc)
    codes = np.array(arc.codes)
    verts = np.array(arc.vertices)

    # Make sure we are drawing from start to end
    if reverse_path:
        verts = verts[::-1, :]

    # Change the initial MOVETO operation into a LINETO to connect to the
    # previous path
    codes[0] = Path.LINETO

    return codes, verts


def _tokenize_path(pathdef):
    for x in COMMAND_RE.split(pathdef):
        if x in COMMANDS:
            yield x
        for token in FLOAT_RE.findall(x):
            yield token


def _next_pos(elements):
    return float(elements.pop()) + float(elements.pop()) * 1j


def _parse_path(pathdef, current_pos):
    # In the SVG specs, initial movetos are absolute, even if specified as 'm'.
    # This is the default behavior here as well. But if you pass in a
    # current_pos variable, the initial moveto will be relative to that
    # current_pos. This is useful.
    elements = list(_tokenize_path(pathdef))
    # Reverse for easy use of .pop()
    elements.reverse()

    start_pos = None
    command = None

    while elements:

        # 1. Determine the current command

        if elements[-1] in COMMANDS:
            # New command.
            last_command = command  # Used by S and T
            command = elements.pop()
            absolute = command in UPPERCASE
            command = command.upper()
        else:
            # Implicit command.
            # If this element starts with numbers, it is an implicit command
            # and we don't change the command. Check that it's allowed:
            if command is None:
                raise ValueError(
                    "Unallowed implicit command in {}, position {}".format(
                        pathdef, len(pathdef.split()) - len(elements))
                    )
            last_command = command  # Used by S and T

        # 2. Parse the current command

        # MOVETO
        if command == 'M':
            pos = _next_pos(elements)
            if absolute:
                current_pos = pos
            else:
                current_pos += pos

            # when M is called, reset start_pos
            # This behavior of Z is defined in svg spec:
            # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
            start_pos = current_pos

            yield COMMAND_CODES['M'], [(current_pos.real, current_pos.imag)]

            # Implicit moveto commands are treated as lineto commands.
            # So we set command to lineto here, in case there are
            # further implicit commands after this moveto.
            command = 'L'

        # CLOSEPATH
        elif command == 'Z':
            # path closure
            if current_pos != start_pos:
                verts = [(start_pos.real, start_pos.imag)]
                yield COMMAND_CODES['L'], verts

            # mpl.Path: a point is required but ignored
            verts = [(start_pos.real, start_pos.imag)]
            yield COMMAND_CODES['Z'], verts

            current_pos = start_pos
            start_pos = None
            command = None  # You can't have implicit commands after closing.

        # LINETO
        elif command == 'L':
            pos = _next_pos(elements)
            if not absolute:
                pos += current_pos
            verts = [(pos.real, pos.imag)]
            yield COMMAND_CODES['L'], verts
            current_pos = pos

        # HORIZONTAL_PATHTO
        elif command == 'H':
            x = elements.pop()
            pos = float(x) + current_pos.imag * 1j
            if not absolute:
                pos += current_pos.real
            verts = [(pos.real, pos.imag)]
            yield COMMAND_CODES['H'], verts
            current_pos = pos

        # VERTICAL_PATHTO
        elif command == 'V':
            y = elements.pop()
            pos = current_pos.real + float(y) * 1j
            if not absolute:
                pos += current_pos.imag * 1j
            verts = [(pos.real, pos.imag)]
            yield COMMAND_CODES['V'], verts
            current_pos = pos

        # CUBIC_BEZIER
        elif command == 'C':
            control1 = _next_pos(elements)
            control2 = _next_pos(elements)
            end = _next_pos(elements)
            if not absolute:
                control1 += current_pos
                control2 += current_pos
                end += current_pos
            verts = [
                 (control1.real, control1.imag),
                 (control2.real, control2.imag),
                 (end.real, end.imag)
            ]
            yield COMMAND_CODES['C'], verts
            current_pos = end

        # SMOOTH_CUBIC_BEZIER
        elif command == 'S':
            # Smooth curve. First control point is the "reflection" of
            # the second control point in the previous path.
            if last_command not in 'CS':
                # If there is no previous command or if the previous command
                # was not an C, c, S or s, assume the first control point is
                # coincident with the current point.
                control1 = current_pos
            else:
                # The first control point is assumed to be the reflection of
                # the second control point on the previous command relative
                # to the current point.
                last_control = control2
                control1 = current_pos + current_pos - last_control
            control2 = _next_pos(elements)
            end = _next_pos(elements)
            if not absolute:
                control2 += current_pos
                end += current_pos
            verts = [
                (control1.real, control1.imag),
                (control2.real, control2.imag),
                (end.real, end.imag)
            ]
            yield COMMAND_CODES['S'], verts
            current_pos = end

        # QUADRATIC_BEZIER
        elif command == 'Q':
            control = _next_pos(elements)
            end = _next_pos(elements)
            if not absolute:
                control += current_pos
                end += current_pos
            verts = [
                (control.real, control.imag),
                (end.real, end.imag)
            ]
            yield COMMAND_CODES['Q'], verts
            current_pos = end

        # SMOOTH_QUADRATIC_BEZIER
        elif command == 'T':
            # Smooth curve. Control point is the "reflection" of
            # the second control point in the previous path.
            if last_command not in 'QT':
                # If there is no previous command or if the previous command
                # was not an Q, q, T or t, assume the first control point is
                # coincident with the current point.
                control = current_pos
            else:
                # The control point is assumed to be the reflection of
                # the control point on the previous command relative
                # to the current point.
                last_control = control
                control = current_pos + current_pos - last_control
            end = _next_pos(elements)
            if not absolute:
                end += current_pos
            verts = [
                (control.real, control.imag),
                (end.real, end.imag)
            ]
            yield COMMAND_CODES['T'], verts
            current_pos = end

        # ELLIPTICAL_ARC
        elif command == 'A':
            radius = _next_pos(elements)
            rotation = float(elements.pop())
            large = float(elements.pop())
            sweep = float(elements.pop())
            end = _next_pos(elements)

            if not absolute:
                end += current_pos

            if current_pos == end:
                # This is equivalent of omitting the segment, so do nothing
                continue
            elif radius.real == 0 or radius.imag == 0:
                # This should be treated as a straight line
                verts = [(end.real, end.imag)]
                yield COMMAND_CODES['L'], verts
            else:
                codes, verts = arc_path(
                    current_pos, radius, rotation, large, sweep, end
                )
                yield codes, verts

            current_pos = end


def parse_path(pathdef, current_pos=0 + 0j):
    """
    Parse an SVG path definition string into a matplotlib Path object.
    Parameters
    ----------
    pathdef : str
        SVG path 'd' attribute, e.g. 'M 100 100 L 300 100 L 200 300 z'.
    current_pos : complex, optional
        Coordinates of the starting position of the path, given as a complex
        number. When provided, an initial moveto operation will be intepreted
        as relative to this position even if given as M.
    Returns
    -------
    :class:`matplotlib.path.Path` instance
    See also
    --------
    matplotlib.path.Path
    matplotlib.patches.PathPatch
    matplotlib.collections.PathCollection
    matplotlib.transforms
    """
    codes = []
    verts = []
    for c, v in _parse_path(pathdef, current_pos):
        codes.extend(c)
        verts.extend(v)
    return Path(verts, codes)
