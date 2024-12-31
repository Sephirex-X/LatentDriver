import numpy as np
from shapely.geometry import LineString

# Helper function to calculate tangents and normalize them
def calculate_tangents(points):
    tangents = np.diff(points, axis=0)
    tangents_norm = tangents / np.linalg.norm(tangents, axis=1)[:, np.newaxis]
    return tangents_norm

# Helper function to calculate angles between consecutive tangents
def calculate_angles(tangents_norm):
    angles = np.arccos(np.clip(np.sum(tangents_norm[:-1] * tangents_norm[1:], axis=1), -1.0, 1.0))
    return angles

# Helper function to calculate segment lengths
def calculate_segment_lengths(points):
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return segment_lengths

# Main function to calculate curvature
def calculate_curvature(line):
    points = np.array(line.coords)
    tangents_norm = calculate_tangents(points)
    angles = calculate_angles(tangents_norm)
    segment_lengths = calculate_segment_lengths(points)
    
    curvatures = angles / segment_lengths[:-1]
    cross_product_z = np.cross(tangents_norm[:-1], tangents_norm[1:])
    curvature_sign = np.sign(cross_product_z)
    
    return curvatures, curvature_sign

# Helper function to calculate cumulative distances
def calculate_cumulative_distances(points):
    dists = np.sqrt(np.sum(np.diff(points, axis=1)**2, axis=2))
    cum_dists = np.cumsum(dists, axis=1)
    cum_dists = np.insert(cum_dists, 0, 0, axis=1)
    cum_dists /= cum_dists[:, -1, np.newaxis]
    return cum_dists

# Helper function to interpolate new points
def interpolate_points(cum_dists, points, num_points):
    new_dists = np.linspace(0, 1, num_points)
    new_points = np.zeros((points.shape[0], num_points, points.shape[2]))
    
    for i in range(points.shape[0]):
        for j in range(points.shape[2]):
            new_points[i, :, j] = np.interp(new_dists, cum_dists[i], points[i, :, j])
    
    return new_points

# Main function to resample line
def resample_line(points, num_points):
    cum_dists = calculate_cumulative_distances(points)
    new_points = interpolate_points(cum_dists, points, num_points)
    return new_points

# Main function to get length and curvature
def get_length_curvature(sdc_traj, yaw):
    line = LineString(sdc_traj)
    length = line.length

    if length < 3:
        max_curvature = -1
        curvature_sign = 0
        diff = 0
        lane = LineString([])
    else:
        num_samples = int(length)
        resampled_points = resample_line(sdc_traj[np.newaxis, ...], num_samples)[0]
        lane = LineString(resampled_points)
        curvatures, curvature_sign = calculate_curvature(lane)
        diff = np.abs(yaw[0] - yaw[-1])
        max_curvature_idx = np.argmax(curvatures)
        max_curvature = curvatures[max_curvature_idx]
        curvature_sign = curvature_sign[max_curvature_idx]
    
    return max_curvature, curvature_sign, length, diff, lane

if __name__ == '__main__':
    '''
        max_curvature : 
            0.0 ~ 0.10 near straight
            0.10 ~ 0.3 turning
            > 0.3 U-turn
        sign:
            1 left
            -1 right
            0 stationary
        length:
            unit is m
    '''
    # Example usage
    # max_curvature, sign, length = get_length_curvature(sdc_xy[i], yaw)
    pass