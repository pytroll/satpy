#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2010.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>

# This file is part of mpop.

# mpop is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.

# mpop is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# mpop.  If not, see <http://www.gnu.org/licenses/>.

"""Spherical geometry module.
"""

import math
import numpy as np

EPSILON = 0.0000001
            
class Coordinate(object):
    """Point on earth in terms of lat and lon.
    """
    lat = None
    lon = None
    x__ = None
    y__ = None
    z__ = None
    
    def __init__(self, lat=None, lon=None,
                 x__=None, y__=None, z__=None, R=1):

        self.R = R

        if lat is not None and lon is not None:
            self.lat = math.radians(lat)
            self.lon = math.radians(lon)
            self._update_cart()
        else:
            self.x__ = x__
            self.y__ = y__
            self.z__ = z__
            self._update_lonlat()

    def _update_cart(self):
        """Convert lon/lat to cartesian coordinates.
        """

        self.x__ = math.cos(self.lat) * math.cos(self.lon)
        self.y__ = math.cos(self.lat) * math.sin(self.lon)
        self.z__ = math.sin(self.lat)
        

    def _update_lonlat(self):
        """Convert cartesian to lon/lat.
        """
        
        self.lat = math.degrees(math.asin(self.z__ / self.R))
        self.lon = math.degrees(math.atan2(self.y__, self.x__))
        
    def __ne__(self, other):
        if(abs(self.lat - other.lat) < EPSILON and
           abs(self.lon - other.lon) < EPSILON):
            return 0
        else:
            return 1

    def __eq__(self, other):
        return not self.__ne__(other)

    def __str__(self):
        return str((math.degrees(self.lat), math.degrees(self.lon)))
    
    def __repr__(self):
        return str((math.degrees(self.lat), math.degrees(self.lon)))

    def cross2cart(self, point):
        """Compute the cross product, and convert to cartesian coordinates
        (assuming radius 1).
        """

        lat1 = self.lat
        lon1 = self.lon
        lat2 = point.lat
        lon2 = point.lon

        res = Coordinate(
            x__=(math.sin(lat1 - lat2) * math.sin((lon1 + lon2) / 2) *
                 math.cos((lon1 - lon2) / 2) - math.sin(lat1 + lat2) *
                 math.cos((lon1 + lon2) / 2) * math.sin((lon1 - lon2) / 2)),
            y__=(math.sin(lat1 - lat2) * math.cos((lon1 + lon2) / 2) *
                 math.cos((lon1 - lon2) / 2) + math.sin(lat1 + lat2) *
                 math.sin((lon1 + lon2) / 2) * math.sin((lon1 - lon2) / 2)),
            z__=(math.cos(lat1) * math.cos(lat2) * math.sin(lon1 - lon2)))

        return res

    def distance(self, point):
        """Vincenty formula.
        """
        dlambda = self.lon - point.lon
        num = ((math.cos(point.lat) * math.sin(dlambda)) ** 2 +
               (math.cos(self.lat) * math.sin(point.lat) -
                math.sin(self.lat) * math.cos(point.lat) *
                math.cos(dlambda)) ** 2)
        den = (math.sin(self.lat) * math.sin(point.lat) +
               math.cos(self.lat) * math.cos(point.lat) * math.cos(dlambda))

        return math.atan2(math.sqrt(num), den)

    def norm(self):
        """Return the norm of the vector.
        """
        return math.sqrt(self.x__ ** 2 + self.y__ ** 2 + self.z__ ** 2)

    def normalize(self):
        """normalize the vector.
        """

        norm = self.norm()
        self.x__ /= norm
        self.y__ /= norm
        self.z__ /= norm

        return self

    def cross(self, point):
        """cross product with another vector.
        """
        x__ = self.y__ * point.z__ - self.z__ * point.y__
        y__ = self.z__ * point.x__ - self.x__ * point.z__
        z__ = self.x__ * point.y__ - self.y__ * point.x__
        
        return Coordinate(x__=x__, y__=y__, z__=z__)

    def dot(self, point):
        """dot product with another vector.
        """
        return (self.x__ * point.x__ +
                self.y__ * point.y__ +
                self.z__ * point.z__)

class Arc(object):
    """An arc of the great circle between two points.
    """
    start = None
    end = None

    def __init__(self, start, end):
        self.start, self.end = start, end

    def center_angle(self):
        """Angle of an arc at the center of the sphere.
        """
        val = (math.cos(self.start.lat - self.end.lat) +
               math.cos(self.start.lon - self.end.lon) - 1)

        if val > 1:
            val = 1
        elif val < -1:
            val = -1
        
        return math.acos(val)
                           
    def __eq__(self, other):
        if(self.start == other.start and self.end == other.end):
            return 1
        return 0

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return str((str(self.start), str(self.end)))

    def angle(self, other_arc):
        """Oriented angle between two arcs.
        """
        if self.start == other_arc.start:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.end
        elif self.start == other_arc.end:
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
        elif self.end == other_arc.end:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.start
        elif self.end == other_arc.start:
            a__ = self.end
            b__ = self.start
            c__ = other_arc.end
        else:
            raise ValueError("No common point in angle computation.")

        ua_ = a__.cross(b__)
        ub_ = a__.cross(c__)

        val =  ua_.dot(ub_) / (ua_.norm() * ub_.norm())
        if abs(val - 1) < EPSILON:
            angle = 0
        elif abs(val + 1) < EPSILON:
            angle = math.pi
        else:
            angle = math.acos(val)    

        n__ = ua_.normalize()
        if n__.dot(c__) > 0:
            return -angle
        else:
            return angle
        
    def intersections(self, other_arc):
        """Gives the two intersections of the greats circles defined by the 
       current arc and *other_arc*.
        """
        
        
        if self.end.lon - self.start.lon > math.pi:
            self.end.lon -= 2 * math.pi
        if other_arc.end.lon - other_arc.start.lon > math.pi:
            other_arc.end.lon -= 2 * math.pi
        if self.end.lon - self.start.lon < -math.pi:
            self.end.lon += 2 * math.pi
        if other_arc.end.lon - other_arc.start.lon < -math.pi:
            other_arc.end.lon += 2 * math.pi
            
        ea_ = self.start.cross2cart(self.end).normalize()
        eb_ = other_arc.start.cross2cart(other_arc.end).normalize()

        cross = ea_.cross(eb_)
        lat = math.atan2(cross.z__, math.sqrt(cross.x__ ** 2 + cross.y__ ** 2))
        lon = math.atan2(-cross.y__, cross.x__)

        return (Coordinate(math.degrees(lat), math.degrees(lon)),
                Coordinate(math.degrees(-lat),
                           math.degrees(modpi(lon + math.pi))))

    def intersects(self, other_arc):
        """Says if two arcs defined by the current arc and the *other_arc*
        intersect. An arc is defined as the shortest tracks between two points.
        """

        
        for i in self.intersections(other_arc):
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
            d__ = other_arc.end


            ab_ = a__.distance(b__)
            cd_ = c__.distance(d__)

            if(abs(a__.distance(i) + b__.distance(i) - ab_) < EPSILON and
               abs(c__.distance(i) + d__.distance(i) - cd_) < EPSILON):
                return True
        return False

    def intersection(self, other_arc):
        """Says where, if two arcs defined by the current arc and the
        *other_arc* intersect. An arc is defined as the shortest tracks between
        two points.
        """

        
        for i in self.intersections(other_arc):
            a__ = self.start
            b__ = self.end
            c__ = other_arc.start
            d__ = other_arc.end


            ab_ = a__.distance(b__)
            cd_ = c__.distance(d__)

            if(abs(a__.distance(i) + b__.distance(i) - ab_) < EPSILON and
               abs(c__.distance(i) + d__.distance(i) - cd_) < EPSILON):
                return i
        return None

def modpi(val):
    """Puts *val* between -pi and pi.
    """
    return (val + math.pi) % (2 * math.pi) - math.pi

def modpi2(val):
    """Puts *val* between 0 and 2pi.
    """
    return val % (2 * math.pi)


def point_inside(point, corners):
    """Is a point inside the 4 corners ? This uses great circle arcs as area
    boundaries.
    """
    arc1 = Arc(corners[0], corners[1])
    arc2 = Arc(corners[1], corners[2])
    arc3 = Arc(corners[2], corners[3])
    arc4 = Arc(corners[3], corners[0])
    
    arc5 = Arc(corners[1], point)
    arc6 = Arc(corners[3], point)

    angle1 = modpi(arc1.angle(arc2))
    angle1bis = modpi(arc1.angle(arc5))

    angle2 = modpi(arc3.angle(arc4))
    angle2bis = modpi(arc3.angle(arc6))

    return (np.sign(angle1) == np.sign(angle1bis) and
            abs(angle1) > abs(angle1bis) and 
            np.sign(angle2) == np.sign(angle2bis) and
            abs(angle2) > abs(angle2bis))
    
def overlaps(area_corners, segment_corners):
    """Are two areas overlapping ? This uses great circle arcs as area
    boundaries.
    """
    for i in area_corners:
        if point_inside(i, segment_corners):
            return True
    for i in segment_corners:
        if point_inside(i, area_corners):
            return True
    
    area_arc1 = Arc(area_corners[0], area_corners[1])
    area_arc2 = Arc(area_corners[1], area_corners[2])
    area_arc3 = Arc(area_corners[2], area_corners[3])
    area_arc4 = Arc(area_corners[3], area_corners[0])

    segment_arc1 = Arc(segment_corners[0], segment_corners[1])
    segment_arc2 = Arc(segment_corners[1], segment_corners[2])
    segment_arc3 = Arc(segment_corners[2], segment_corners[3])
    segment_arc4 = Arc(segment_corners[3], segment_corners[0])

    for i in (area_arc1, area_arc2, area_arc3, area_arc4):
        for j in (segment_arc1, segment_arc2, segment_arc3, segment_arc4):
            if i.intersects(j):
                return True
    return False

def get_intersections(b__, boundaries):
    """Get the intersections of *b__* with *boundaries*.
    Returns both the intersection coordinates and the concerned boundaries.
    """
    
    intersections = []
    bounds = []
    for other_b in boundaries:
        inter = b__.intersection(other_b)
        if inter is not None:
            intersections.append(inter)
            bounds.append(other_b)
    return intersections, bounds
    
def get_first_intersection(b__, boundaries):
    """Get the first intersection on *b__* with *boundaries*.
    """
    intersections, bounds = get_intersections(b__, boundaries)
    del bounds
    dists = np.array([b__.start.distance(p__) for p__ in intersections])
    indices = dists.argsort()
    if len(intersections) > 0:
        return intersections[indices[0]]
    return None

def get_next_intersection(p__, b__, boundaries):
    """Get the next intersection from the intersection of arcs *p__* and *b__*
    along segment *b__* with *boundaries*.
    """
    new_b = Arc(p__, b__.end)
    intersections, bounds = get_intersections(new_b, boundaries)
    dists = np.array([b__.start.distance(p2) for p2 in intersections])
    indices = dists.argsort()
    if len(intersections) > 0 and intersections[indices[0]] != p__:
        return intersections[indices[0]], bounds[indices[0]]
    elif len(intersections) > 1:
        return intersections[indices[1]], bounds[indices[1]]
    return None, None

def polygon(area_corners, segment_corners):
    """Get the intersection polygon between two areas.
    """
    area_boundaries = [Arc(area_corners[0], area_corners[1]),
                       Arc(area_corners[1], area_corners[2]),
                       Arc(area_corners[2], area_corners[3]),
                       Arc(area_corners[3], area_corners[0])]
    segment_boundaries = [Arc(segment_corners[0], segment_corners[1]),
                          Arc(segment_corners[1], segment_corners[2]),
                          Arc(segment_corners[2], segment_corners[3]),
                          Arc(segment_corners[3], segment_corners[0])]

    angle1 = area_boundaries[0].angle(area_boundaries[1])
    angle2 = segment_boundaries[0].angle(segment_boundaries[1])
    if np.sign(angle1) != np.sign(angle2):
        segment_corners.reverse()
        segment_boundaries = [Arc(segment_corners[0], segment_corners[1]),
                              Arc(segment_corners[1], segment_corners[2]),
                              Arc(segment_corners[2], segment_corners[3]),
                              Arc(segment_corners[3], segment_corners[0])]
    poly = []

    boundaries = area_boundaries
    other_boundaries = segment_boundaries

    b__ = None

    for b__ in boundaries:
        if point_inside(b__.start, segment_corners):
            poly.append(b__.start)
            break
        else:
            inter = get_first_intersection(b__, other_boundaries)
            if inter is not None:
                poly.append(inter)
                break
    if len(poly) == 0:
        return None
    while len(poly) < 2 or poly[0] != poly[-1]:
        inter, b2_ = get_next_intersection(poly[-1], b__, other_boundaries)
        if inter is None:
            poly.append(b__.end)
            idx = (boundaries.index(b__) + 1) % len(boundaries)
            b__ = boundaries[idx]
        else:
            poly.append(inter)
            b__ = b2_
            boundaries, other_boundaries = other_boundaries, boundaries
    return poly[:-1]

R = 1

def get_area(corners):
    """Get the area of the convex area defined by *corners*.
    """
    c1_ = corners[0]
    area = 0

    for idx in range(1, len(corners) - 1):
        b1_ = Arc(c1_, corners[idx])
        b2_ = Arc(c1_, corners[idx + 1])
        b3_ = Arc(corners[idx], corners[idx + 1])
        e__ = (abs(b1_.angle(b2_)) +
               abs(b2_.angle(b3_)) + 
               abs(b3_.angle(b1_)))
        area += R ** 2 * e__ - math.pi
    return area

def overlap_rate(swath_corners, area_corners):
    """Get how much a swath overlaps an area.
    """
    area_area = get_area(area_corners)
    inter_area = get_area(polygon(area_corners, swath_corners))
    return inter_area / area_area

def min_distances(area_corners, segment_corners):
    """Min distances between each corner of *area_corners* and
    *segment_corners*.
    """
    dists = np.ones(4) * np.infty
    for i, ic_ in enumerate(area_corners):
        for jc_ in segment_corners:
            dist = ic_.distance(jc_)
            if dists[i] > dist:
                dists[i] = dist
    return dists

def should_wait(area_corners, segment_corners, previous_segment_corners):
    """Are the newest cornest still inside the area ? is the last segment
    boundary overlapping any boundary of the area ? In this case we should wait
    for the next segment to arrive.
    """
    dists = min_distances(segment_corners, previous_segment_corners)
    indices = np.argsort(dists)
    new_corners = np.array(segment_corners)[indices[2:]]
    if len(new_corners) != 2:
        raise ValueError("More than 2 corners differ from previous segment...")
    new_arc = Arc(new_corners[0], new_corners[1])
    
    for i in new_corners:
        if point_inside(i, area_corners):
            return True

    area_arc1 = Arc(area_corners[0], area_corners[1])
    area_arc2 = Arc(area_corners[1], area_corners[2])
    area_arc3 = Arc(area_corners[2], area_corners[3])
    area_arc4 = Arc(area_corners[3], area_corners[0])

    for i in (area_arc1, area_arc2, area_arc3, area_arc4):
        if i.intersects(new_arc):
            return True

    return False
    
import unittest
class TestSphereGeometry(unittest.TestCase):
    """Testing sphere geometry from this module.
    """

    def test_angle(self):
        """Testing the angle value between two arcs.
        """

        base = 0

        p0_ = Coordinate(base, base)
        p1_ = Coordinate(base + 1, base)
        p2_ = Coordinate(base, base + 1)
        p3_ = Coordinate(base - 1, base)
        p4_ = Coordinate(base, base - 1)

        arc1 = Arc(p0_, p1_)
        arc2 = Arc(p0_, p2_)
        arc3 = Arc(p0_, p3_)
        arc4 = Arc(p0_, p4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        self.assertAlmostEqual(arc1.angle(arc3), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc3.angle(arc1), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc2.angle(arc4), math.pi,
                               msg="this should be pi")
        self.assertAlmostEqual(arc4.angle(arc2), math.pi,
                               msg="this should be pi")


        p5_ = Coordinate(base + 1, base + 1)
        p6_ = Coordinate(base - 1, base + 1)
        p7_ = Coordinate(base - 1, base - 1)
        p8_ = Coordinate(base + 1, base - 1)

        arc5 = Arc(p0_, p5_)
        arc6 = Arc(p0_, p6_)
        arc7 = Arc(p0_, p7_)
        arc8 = Arc(p0_, p8_)

        self.assertAlmostEqual(arc1.angle(arc5), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc5.angle(arc2), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc2.angle(arc6), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc6.angle(arc3), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc3.angle(arc7), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc7.angle(arc4), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc4.angle(arc8), math.pi / 4, 3,
                               msg="this should be pi/4")
        self.assertAlmostEqual(arc8.angle(arc1), math.pi / 4, 3,
                               msg="this should be pi/4")

        self.assertAlmostEqual(arc1.angle(arc6), 3 * math.pi / 4, 3,
                               msg="this should be 3pi/4")


        c0_ = Coordinate(0, 180)
        c1_ = Coordinate(1, 180)
        c2_ = Coordinate(0, -179)
        c3_ = Coordinate(-1, -180)
        c4_ = Coordinate(0, 179)


        arc1 = Arc(c0_, c1_)
        arc2 = Arc(c0_, c2_)
        arc3 = Arc(c0_, c3_)
        arc4 = Arc(c0_, c4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        # case of the north pole

        c0_ = Coordinate(90, 0)
        c1_ = Coordinate(89, 0)
        c2_ = Coordinate(89, -90)
        c3_ = Coordinate(89, 180)
        c4_ = Coordinate(89, 90)

        arc1 = Arc(c0_, c1_)
        arc2 = Arc(c0_, c2_)
        arc3 = Arc(c0_, c3_)
        arc4 = Arc(c0_, c4_)

        self.assertAlmostEqual(arc1.angle(arc2), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc2.angle(arc3), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc3.angle(arc4), math.pi / 2,
                               msg="this should be pi/2")
        self.assertAlmostEqual(arc4.angle(arc1), math.pi / 2,
                               msg="this should be pi/2")

        self.assertAlmostEqual(arc1.angle(arc4), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc4.angle(arc3), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc3.angle(arc2), -math.pi / 2,
                               msg="this should be -pi/2")
        self.assertAlmostEqual(arc2.angle(arc1), -math.pi / 2,
                               msg="this should be -pi/2")

        self.assertAlmostEqual(Arc(c1_, c2_).angle(arc1), math.pi/4, 3,
                               msg="this should be pi/4")
                               
        self.assertAlmostEqual(Arc(c4_, c3_).angle(arc4), -math.pi/4, 3,
                               msg="this should be -pi/4")

        self.assertAlmostEqual(Arc(c1_, c4_).angle(arc1), -math.pi/4, 3,
                               msg="this should be -pi/4")


    def test_inside(self):
        """Testing if a point is inside for other points.
        """
        c1_ = Coordinate(-11, -11)
        c2_ = Coordinate(11, -11)
        c3_ = Coordinate(11, 11)
        c4_ = Coordinate(-11, 11)    

        corners = [c1_, c2_, c3_, c4_]

        point = Coordinate(0, 0)
        self.assertTrue(point_inside(point, corners))

        point = Coordinate(0, 12)
        self.assertFalse(point_inside(point, corners))



        c1_ = Coordinate(-1, 180)
        c2_ = Coordinate(1, 179)
        c3_ = Coordinate(1, -179)
        c4_ = Coordinate(-1, -179)

        corners = [c1_, c2_, c3_, c4_]
        point = Coordinate(0, 180)
        self.assertTrue(point_inside(point, corners))

        point = Coordinate(12, 180)
        self.assertFalse(point_inside(point, corners))

        point = Coordinate(-12, 180)
        self.assertFalse(point_inside(point, corners))

        point = Coordinate(0, 192)
        self.assertFalse(point_inside(point, corners))

        point = Coordinate(0, -192)
        self.assertFalse(point_inside(point, corners))


        # case of the north pole
        c1_ = Coordinate(89, 0)
        c2_ = Coordinate(89, 90)
        c3_ = Coordinate(89, 180)
        c4_ = Coordinate(89, -90)    

        corners = [c1_, c2_, c3_, c4_]

        point = Coordinate(90, 90)

        self.assertTrue(point_inside(point, corners))

    def test_intersects(self):
        """Test if two arcs intersect.
        """
        p0_ = Coordinate(0, 0)
        p1_ = Coordinate(1, 0)
        p2_ = Coordinate(0, 1)
        p3_ = Coordinate(-1, 0)
        p4_ = Coordinate(0, -1)
        p5_ = Coordinate(1, 1)
        p6_ = Coordinate(-1, 1)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))



        p0_ = Coordinate(0, 180)
        p1_ = Coordinate(1, 180)
        p2_ = Coordinate(0, -179)
        p3_ = Coordinate(-1, -180)
        p4_ = Coordinate(0, 179)
        p5_ = Coordinate(1, -179)
        p6_ = Coordinate(-1, -179)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))

        # case of the north pole

        p0_ = Coordinate(90, 0)
        p1_ = Coordinate(89, 0)
        p2_ = Coordinate(89, 90)
        p3_ = Coordinate(89, 180)
        p4_ = Coordinate(89, -90)    
        p5_ = Coordinate(89, 45)
        p6_ = Coordinate(89, 135)

        arc13 = Arc(p1_, p3_)
        arc24 = Arc(p2_, p4_)

        arc32 = Arc(p3_, p2_)
        arc41 = Arc(p4_, p1_)

        arc40 = Arc(p4_, p0_)
        arc56 = Arc(p5_, p6_)

        arc45 = Arc(p4_, p5_)
        arc02 = Arc(p0_, p2_)

        arc35 = Arc(p3_, p5_)

        self.assertTrue(arc13.intersects(arc24))

        self.assertFalse(arc32.intersects(arc41))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc56.intersects(arc40))

        self.assertFalse(arc45.intersects(arc02))

        self.assertTrue(arc35.intersects(arc24))

    def test_overlaps(self):
        """Test if two areas overlap.
        """
        p1_ = Coordinate(89, 0)
        p2_ = Coordinate(89, 90)
        p3_ = Coordinate(89, 180)
        p4_ = Coordinate(89, -90)    
        p5_ = Coordinate(89, 45)
        p6_ = Coordinate(89, 135)
        p7_ = Coordinate(89, -135)
        p8_ = Coordinate(89, -45)

        self.assertTrue(overlaps([p1_, p2_, p3_, p4_],
                                 [p5_, p6_, p7_, p8_]))
        self.assertFalse(overlaps([p1_, p5_, p2_, p6_],
                                  [p3_, p7_, p4_, p8_]))

        p1_ = Coordinate(1, -1)
        p2_ = Coordinate(1, 1)
        p3_ = Coordinate(-1, 1)
        p4_ = Coordinate(-1, -1)

        p5_ = Coordinate(0, 0)
        p6_ = Coordinate(0, 2)
        p7_ = Coordinate(2, 2)
        p8_ = Coordinate(2, 0)

        self.assertTrue(overlaps([p1_, p2_, p3_, p4_], [p5_, p6_, p7_, p8_]))
        self.assertFalse(overlaps([p1_, p8_, p5_, p4_], [p2_, p3_, p6_, p7_]))


    def test_overlap_rate(self):
        """Test how much two areas overlap.
        """
        p1_ = Coordinate(1, -1)
        p2_ = Coordinate(1, 1)
        p3_ = Coordinate(-1, 1)
        p4_ = Coordinate(-1, -1)

        p5_ = Coordinate(0, 0)
        p6_ = Coordinate(0, 2)
        p7_ = Coordinate(2, 2)
        p8_ = Coordinate(2, 0)
        self.assertAlmostEqual(overlap_rate([p1_, p2_, p3_, p4_],
                                            [p5_, p6_, p7_, p8_]), 0.25, 3)
        
        c1_ = [(60.5944, 82.829699999999974),
               (52.859999999999999, 36.888300000000001),
               (66.7547, 2.8773),
               (80.395899999999997, 98.145499999999984)]
        c2_ = [(62.953206630716465, 7.8098183315148422),
               (62.953206630716465, 26.189349044600252),
               (53.301561187195546, 26.189349044600252),
               (53.301561187195546, 7.8098183315148422)]
        cor1 = [Coordinate(t[0], t[1]) for t in c1_]
        cor2 = [Coordinate(t[0], t[1]) for t in c2_]
        self.assertAlmostEqual(overlap_rate(cor1, cor2), 0.07, 2)

        
        c1_ = [(60.5944, 82.829699999999974),
               (52.859999999999999, 36.888300000000001),
               (66.7547, 2.8773),
               (80.395899999999997, 98.145499999999984)]
        c2_ = [(65.98228561983025, 12.108984194981202),
               (65.98228561983025, 30.490647126520301),
               (57.304862819933433, 30.490647126520301),
               (57.304862819933433, 12.108984194981202)]
        cor1 = [Coordinate(t[0], t[1]) for t in c1_]
        cor2 = [Coordinate(t[0], t[1]) for t in c2_]
        self.assertAlmostEqual(overlap_rate(cor1, cor2), 0.5, 2)

        
if __name__ == '__main__':
    unittest.main()



