#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: some utilities

import os
import math
import numpy as np
import utils.depth_map_utils as depth_map_utils


try:
  from c_gen_normal_map import gen_normal_map
except:
  print("You are currently using python library, which could be slow.")
  print("If you want to use fast C library, please Export PYTHONPATH=<path-to-range-image-library>")


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  # current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  # depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  from_proj_x = np.copy(proj_x)  # store a copy in orig order

  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  from_proj_y = np.copy(proj_y)  # stope a copy in original order

  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_range, proj_vertex, proj_intensity, proj_idx, from_proj_x, from_proj_y


# def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):  # 高64，宽900
#   """ Generate a normal image given the range projection of a point cloud.
#       Args:
#         current_range:  range projection of a point cloud, each pixel contains the corresponding depth
#         current_vertex: range projection of a point cloud,
#                         each pixel contains the corresponding point (x, y, z, 1)
#       Returns: 
#         normal_data: each pixel contains the corresponding normal
#   """
#   normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
  
#   # iterate over all pixels in the range image
#   for x in range(proj_W):
#     for y in range(proj_H - 1):
#       p = current_vertex[y, x][:3]
#       depth = current_range[y, x]
      
#       if depth > 0:
#         wrap_x = wrap(x + 1, proj_W)
#         u = current_vertex[y, wrap_x][:3]
#         u_depth = current_range[y, wrap_x]
#         if u_depth <= 0:
#           continue
        
#         v = current_vertex[y + 1, x][:3]
#         v_depth = current_range[y + 1, x]
#         if v_depth <= 0:
#           continue
        
#         u_norm = (u - p) / np.linalg.norm(u - p)
#         v_norm = (v - p) / np.linalg.norm(v - p)
        
#         w = np.cross(v_norm, u_norm)
#         norm = np.linalg.norm(w)
#         if norm > 0:
#           normal = w / norm
#           normal_data[y, x] = normal
  
#   return normal_data

# def wrap(x, dim):
#   """ Wrap the boarder of the range image.
#   """
#   value = x
#   if value >= dim:
#     value = (value - dim)
#   if value < 0:
#     value = (value + dim)
#   return value

def compute_normals_range(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50, extrapolate = True, blur_type = 'gaussian'):
  """ compute normals for each point using range image-based method.
  """
  proj_range, proj_vertex, _, _, from_proj_x, from_proj_y = range_projection(current_vertex)
  proj_range = depth_map_utils.fill_in_fast(proj_range, extrapolate=extrapolate, blur_type=blur_type)

  # generate normal image
  normal_data = gen_normal_map(proj_range, proj_vertex, proj_H, proj_W)

  unproj_normal_data = normal_data[from_proj_y,from_proj_x]

  return unproj_normal_data

if __name__ == '__main__':
  proj_H=64
  proj_W=900
  extrapolate = True
  blur_type = 'gaussian'
  
  scan_path = '/data/sequences/00/velodyne/000000.bin'
  current_vertex = np.fromfile(scan_path, dtype=np.float32)
  current_vertex = current_vertex.reshape((-1, 4))
  
#   # generate range image from point cloud
#   proj_range, proj_vertex, _, _, from_proj_x, from_proj_y = range_projection(current_vertex)
#   proj_range = depth_map_utils.fill_in_fast(proj_range, extrapolate=extrapolate, blur_type=blur_type)

#   # generate normal image
#   normal_data = gen_normal_map(proj_range, proj_vertex, proj_H, proj_W)

#   unproj_normal_data = normal_data[from_proj_y,from_proj_x]
  unproj_normal_data = compute_normals_range(current_vertex)
  print(unproj_normal_data.shape)

