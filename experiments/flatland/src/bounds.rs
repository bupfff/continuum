//! Conservative geometry for charging ledgers: convex hulls, overlap tests and
//! bounds on the angle a region can subtend.

use crate::geom::{Vec2, v2};

#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec2,
    pub max: Vec2,
}

impl Aabb {
    pub fn of(points: &[Vec2]) -> Aabb {
        let mut b = Aabb {
            min: points[0],
            max: points[0],
        };
        for p in &points[1..] {
            b.min = v2(b.min.x.min(p.x), b.min.y.min(p.y));
            b.max = v2(b.max.x.max(p.x), b.max.y.max(p.y));
        }
        b
    }

    pub fn overlaps(&self, o: &Aabb) -> bool {
        self.min.x <= o.max.x
            && o.min.x <= self.max.x
            && self.min.y <= o.max.y
            && o.min.y <= self.max.y
    }

    pub fn corners(&self) -> [Vec2; 4] {
        [
            self.min,
            v2(self.max.x, self.min.y),
            self.max,
            v2(self.min.x, self.max.y),
        ]
    }

    /// Distance from `p` to the nearest point of the box (0 inside).
    pub fn distance_to(&self, p: Vec2) -> f32 {
        let q = v2(
            p.x.clamp(self.min.x, self.max.x),
            p.y.clamp(self.min.y, self.max.y),
        );
        (p - q).length()
    }
}

/// Counter-clockwise convex hull (Andrew's monotone chain). Collinear input
/// yields its two extreme points; a single repeated point yields one point.
pub fn convex_hull(points: &[Vec2]) -> Vec<Vec2> {
    let mut pts = points.to_vec();
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let turn = |o: Vec2, a: Vec2, b: Vec2| (a - o).cross(b - o);
    let mut hull: Vec<Vec2> = Vec::with_capacity(pts.len() * 2);
    for pass in [&pts[..], &pts.iter().rev().copied().collect::<Vec<_>>()[..]] {
        let start = hull.len();
        for &p in pass {
            while hull.len() >= start + 2
                && turn(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Separating-axis test for two convex polygons (points or segments allowed).
/// Touching counts as overlapping, which keeps callers conservative.
pub fn convex_overlap(a: &[Vec2], b: &[Vec2]) -> bool {
    let separated_along = |axis: Vec2| {
        let range = |poly: &[Vec2]| {
            poly.iter()
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                    let d = p.dot(axis);
                    (lo.min(d), hi.max(d))
                })
        };
        let (a_lo, a_hi) = range(a);
        let (b_lo, b_hi) = range(b);
        a_hi < b_lo || b_hi < a_lo
    };
    for poly in [a, b] {
        if poly.len() < 2 {
            continue;
        }
        for i in 0..poly.len() {
            let edge = poly[(i + 1) % poly.len()] - poly[i];
            if separated_along(edge.perp()) {
                return false;
            }
        }
    }
    true
}

pub fn point_segment_distance(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
}

/// Distance between two convex polygons (points or segments allowed); 0 if they overlap.
pub fn convex_distance(a: &[Vec2], b: &[Vec2]) -> f32 {
    if convex_overlap(a, b) {
        return 0.0;
    }
    let mut d = f32::INFINITY;
    for (from, to) in [(a, b), (b, a)] {
        for &v in from {
            for i in 0..to.len() {
                d = d.min(point_segment_distance(v, to[i], to[(i + 1) % to.len()]));
            }
        }
    }
    d
}

/// Largest angle a segment of length `len` can subtend from a point at least
/// `distance` away from it (attained on its perpendicular bisector).
pub fn max_segment_angle(len: f32, distance: f32) -> f32 {
    2.0 * (len / (2.0 * distance)).atan()
}

/// Smallest circle centered on the points' bounding-box center that contains them.
#[derive(Clone, Copy, Debug)]
pub struct Circle {
    pub center: Vec2,
    pub radius: f32,
}

impl Circle {
    pub fn enclosing(points: &[Vec2]) -> Circle {
        let b = Aabb::of(points);
        let center = (b.min + b.max) * 0.5;
        let radius = points
            .iter()
            .map(|&p| (p - center).length())
            .fold(0.0, f32::max);
        Circle { center, radius }
    }

    /// Largest angle this circle can subtend from any point in `region`.
    pub fn max_subtended_angle(&self, region: &Aabb) -> f32 {
        let d = region.distance_to(self.center);
        if d <= self.radius {
            std::f32::consts::PI
        } else {
            2.0 * (self.radius / d).asin()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hull_of_square_with_interior_point() {
        let pts = [
            v2(0.0, 0.0),
            v2(1.0, 0.0),
            v2(1.0, 1.0),
            v2(0.0, 1.0),
            v2(0.5, 0.5),
        ];
        let hull = convex_hull(&pts);
        assert_eq!(hull.len(), 4);
        assert!(!hull.contains(&v2(0.5, 0.5)));
    }

    #[test]
    fn overlap_cases() {
        let square = convex_hull(&[v2(0.0, 0.0), v2(1.0, 0.0), v2(1.0, 1.0), v2(0.0, 1.0)]);
        let inside_segment = [v2(0.2, 0.2), v2(0.8, 0.3)];
        let crossing_segment = [v2(-1.0, 0.5), v2(2.0, 0.5)];
        let outside_segment = [v2(1.5, 0.0), v2(2.5, 1.0)];
        let diagonal_miss = [v2(1.2, 0.0), v2(2.0, -0.8)];
        assert!(convex_overlap(&square, &inside_segment));
        assert!(convex_overlap(&square, &crossing_segment));
        assert!(!convex_overlap(&square, &outside_segment));
        assert!(!convex_overlap(&square, &diagonal_miss));
        assert!(convex_overlap(&square, &[v2(0.5, 0.5)]));
    }

    #[test]
    fn distance_between_polygons() {
        let square = convex_hull(&[v2(0.0, 0.0), v2(1.0, 0.0), v2(1.0, 1.0), v2(0.0, 1.0)]);
        assert_eq!(convex_distance(&square, &[v2(0.5, 0.5)]), 0.0);
        assert!((convex_distance(&square, &[v2(2.0, 0.5), v2(3.0, 0.5)]) - 1.0).abs() < 1e-6);
        // Nearest feature is a vertex of the other shape against an edge of the square.
        assert!((convex_distance(&square, &[v2(0.5, 1.5), v2(2.0, 3.0)]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn segment_angle_bound_holds_off_axis() {
        let (len, d) = (0.2f32, 1.0f32);
        let bound = max_segment_angle(len, d);
        // A point offset along the segment's direction sees a smaller angle.
        let (a, b, p) = (v2(-0.1, 0.0), v2(0.1, 0.0), v2(0.3, -1.0));
        let real = (a - p).cross(b - p).atan2((a - p).dot(b - p)).abs();
        assert!(real <= bound);
    }

    #[test]
    fn subtended_angle_bounds_the_real_angle() {
        let c = Circle {
            center: v2(0.0, 2.0),
            radius: 0.5,
        };
        let region = Aabb {
            min: v2(-0.1, -0.1),
            max: v2(0.1, 0.1),
        };
        // From the box's nearest point (0, 0.1) the true angle is 2·asin(0.5 / 1.9).
        let expected = 2.0 * (0.5f32 / 1.9).asin();
        assert!((c.max_subtended_angle(&region) - expected).abs() < 1e-5);
    }
}
