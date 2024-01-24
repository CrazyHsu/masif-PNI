#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: readDataFromSurface.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:50:24
Last modified: 2022-09-11 20:50:24
'''

import time, os, shutil, itertools
import pymesh
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from commonFuncs import makeLink
from geometry import compute_polar_coordinates
from inputOutputProcess import save_ply
from sklearn.neighbors import KDTree
from triangulation import (computeMSMS, fixMesh, computeHydrophobicity, computeCharges, assignChargesToNewMesh,
                           computeAPBS, compute_normal, assignNamesToNewMesh)


def read_data_from_surface(ply_fn, params):
    """
    Read data from a ply file -- decompose into patches.
    Returns:
    list_desc: List of features per patch
    list_coords: list of angular and polar coordinates.
    list_indices: list of indices of neighbors in the patch.
    list_sc_labels: list of shape complementarity labels (computed here).
    """
    mesh = pymesh.load_mesh(ply_fn)

    # Normals:
    n1 = mesh.get_attribute("vertex_nx")
    n2 = mesh.get_attribute("vertex_ny")
    n3 = mesh.get_attribute("vertex_nz")
    normals = np.stack([n1, n2, n3], axis=1)

    # Compute the angular and radial coordinates.
    rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'],
                                                                max_vertices=params['max_shape_size'])

    # Compute the principal curvature components for the shape index.
    mesh.add_attribute("vertex_mean_curvature")
    H = mesh.get_attribute("vertex_mean_curvature")
    mesh.add_attribute("vertex_gaussian_curvature")
    K = mesh.get_attribute("vertex_gaussian_curvature")
    elem = np.square(H) - K
    # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
    # set to an epsilon.
    elem[elem < 0] = 1e-8
    k1 = H + np.sqrt(elem)
    k2 = H - np.sqrt(elem)
    # Compute the shape index
    si = (k1 + k2) / (k1 - k2)
    si = np.arctan(si) * (2 / np.pi)

    # Normalize the charge.
    charge = mesh.get_attribute("vertex_charge")
    charge = normalize_electrostatics(charge)

    # Hbond features
    hbond = mesh.get_attribute("vertex_hbond")

    # Hydropathy features
    # Normalize hydropathy by dividing by 4.5
    hphob = mesh.get_attribute("vertex_hphob") / 4.5

    # Iface labels (for ground truth only)
    if "vertex_iface" in mesh.get_attribute_names():
        iface_labels = mesh.get_attribute("vertex_iface")
    else:
        iface_labels = np.zeros_like(hphob)

    # n: number of patches, equal to the number of vertices.
    n = len(mesh.vertices)

    input_feat = np.zeros((n, params['max_shape_size'], 5))

    # Compute the input features for each patch.
    for vix in range(n):
        # Patch members.
        neigh_vix = np.array(neigh_indices[vix])

        # Compute the distance-dependent curvature for all neighbors of the patch.
        patch_v = mesh.vertices[neigh_vix]
        patch_n = normals[neigh_vix]
        patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
        mask_pos = np.where(mask[vix] == 1.0)[0]  # nonzero elements
        patch_rho = rho[vix][mask_pos]  # nonzero elements of rho
        ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)

        input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]
        input_feat[vix, :len(neigh_vix), 1] = ddc
        input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]
        input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix]
        input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]

    return input_feat, rho, theta, mask, neigh_indices, iface_labels, np.copy(mesh.vertices)


def filterLabelByCount(new_names, labelIndex):
    raw_name_list = [i.split("_") for i in new_names]
    raw_name_list = pd.DataFrame(raw_name_list, columns=["chain", "pos", "x", "res", "atom", "color"])
    name_list = [i.split("_") for i in new_names[labelIndex]]
    name_list = pd.DataFrame(name_list, columns=["chain", "pos", "x", "res", "atom", "color"])
    name_list_gp = name_list.groupby(["chain", "pos", "res"]).size().reset_index(name='counts')
    if sum(name_list_gp.counts) < 200:
        filterCount = 2
    elif 200 <= sum(name_list_gp.counts) and sum(name_list_gp.counts) < 400:
        filterCount = 3
    elif 400 <= sum(name_list_gp.counts) and sum(name_list_gp.counts) < 600:
        filterCount = 4
    else:
        filterCount = 5
    name_list_gp = name_list_gp[name_list_gp.counts >= filterCount]

    commonKeys = ["chain", "pos", "res"]
    i1 = raw_name_list.set_index(commonKeys).index
    i2 = name_list_gp.set_index(commonKeys).index
    tmp_index = raw_name_list[i1.isin(i2)].index
    filterIndex = list(set(tmp_index) & set(labelIndex))
    return filterIndex


# def read_data_from_surface1(pPlyFile, params):
#     """
#     Read data from a ply file -- decompose into patches.
#     Returns:
#     list_desc: List of features per patch
#     list_coords: list of angular and polar coordinates.
#     list_indices: list of indices of neighbors in the patch.
#     list_sc_labels: list of shape complementarity labels (computed here).
#     """
#     mesh = pymesh.load_mesh(pPlyFile)
#
#     # # Normals:
#     # n1 = mesh.get_attribute("vertex_nx")
#     # n2 = mesh.get_attribute("vertex_ny")
#     # n3 = mesh.get_attribute("vertex_nz")
#     # normals = np.stack([n1, n2, n3], axis=1)
#
#     # Compute the angular and radial coordinates.
#     rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'],
#                                                                 max_vertices=params['max_shape_size'])
#
#     # # Compute the principal curvature components for the shape index.
#     # mesh.add_attribute("vertex_mean_curvature")
#     # H = mesh.get_attribute("vertex_mean_curvature")
#     # mesh.add_attribute("vertex_gaussian_curvature")
#     # K = mesh.get_attribute("vertex_gaussian_curvature")
#     # elem = np.square(H) - K
#     # # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
#     # # set to an epsilon.
#     # elem[elem < 0] = 1e-8
#     # k1 = H + np.sqrt(elem)
#     # k2 = H - np.sqrt(elem)
#     # # Compute the shape index
#     # si = (k1 + k2) / (k1 - k2)
#     # si = np.arctan(si) * (2 / np.pi)
#     #
#     # # Normalize the charge.
#     # charge = mesh.get_attribute("vertex_charge")
#     # charge = normalize_electrostatics(charge)
#     #
#     # # Hbond features
#     # hbond = mesh.get_attribute("vertex_hbond")
#     #
#     # # Hydropathy features
#     # # Normalize hydropathy by dividing by 4.5
#     # hphob = mesh.get_attribute("vertex_hphob") / 4.5
#     #
#     # # Iface labels (for ground truth only)
#     # if "vertex_iface" in mesh.get_attribute_names():
#     #     iface_labels = mesh.get_attribute("vertex_iface")
#     # else:
#     #     iface_labels = np.zeros_like(hphob)
#     #
#     # # n: number of patches, equal to the number of vertices.
#     # n = len(mesh.vertices)
#     #
#     # input_feat = np.zeros((n, params['max_shape_size'], 5))
#     #
#     # # Compute the input features for each patch.
#     # for vix in range(n):
#     #     # Patch members.
#     #     neigh_vix = np.array(neigh_indices[vix])
#     #
#     #     # Compute the distance-dependent curvature for all neighbors of the patch.
#     #     patch_v = mesh.vertices[neigh_vix]
#     #     patch_n = normals[neigh_vix]
#     #     patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
#     #     mask_pos = np.where(mask[vix] == 1.0)[0]  # nonzero elements
#     #     patch_rho = rho[vix][mask_pos]  # nonzero elements of rho
#     #     ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)
#     #
#     #     input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]
#     #     input_feat[vix, :len(neigh_vix), 1] = ddc
#     #     input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]
#     #     input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix]
#     #     input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]
#
#     return rho, theta, mask, neigh_indices, np.copy(mesh.vertices)


# def rebuildInputFeat(pPlyFile, params, naPlyFile=None):
#     mesh = pymesh.load_mesh(pPlyFile)
#
#     # Normals:
#     n1 = mesh.get_attribute("vertex_nx")
#     n2 = mesh.get_attribute("vertex_ny")
#     n3 = mesh.get_attribute("vertex_nz")
#     normals = np.stack([n1, n2, n3], axis=1)
#
#     # Compute the angular and radial coordinates.
#     rho, theta, neigh_indices, mask = compute_polar_coordinates(mesh, radius=params['max_distance'],
#                                                                 max_vertices=params['max_shape_size'])
#
#     # Compute the principal curvature components for the shape index.
#     mesh.add_attribute("vertex_mean_curvature")
#     H = mesh.get_attribute("vertex_mean_curvature")
#     mesh.add_attribute("vertex_gaussian_curvature")
#     K = mesh.get_attribute("vertex_gaussian_curvature")
#     elem = np.square(H) - K
#     # In some cases this equation is less than zero, likely due to the method that computes the mean and gaussian curvature.
#     # set to an epsilon.
#     elem[elem < 0] = 1e-8
#     k1 = H + np.sqrt(elem)
#     k2 = H - np.sqrt(elem)
#     # Compute the shape index
#     si = (k1 + k2) / (k1 - k2)
#     si = np.arctan(si) * (2 / np.pi)
#
#     # Normalize the charge.
#     charge = mesh.get_attribute("vertex_charge")
#     charge = normalize_electrostatics(charge)
#
#     # Hbond features
#     hbond = mesh.get_attribute("vertex_hbond")
#
#     # Hydropathy features
#     # Normalize hydropathy by dividing by 4.5
#     hphob = mesh.get_attribute("vertex_hphob") / 4.5
#
#     # Iface labels (for ground truth only)
#     if "vertex_iface" in mesh.get_attribute_names():
#         iface_labels = mesh.get_attribute("vertex_iface")
#     else:
#         iface_labels = np.zeros_like(hphob)
#
#     # n: number of patches, equal to the number of vertices.
#     n = len(mesh.vertices)
#
#     input_feat = np.zeros((n, params['max_shape_size'], 5))
#
#     # Compute the input features for each patch.
#     for vix in range(n):
#         # Patch members.
#         neigh_vix = np.array(neigh_indices[vix])
#
#         # Compute the distance-dependent curvature for all neighbors of the patch.
#         patch_v = mesh.vertices[neigh_vix]
#         patch_n = normals[neigh_vix]
#         patch_cp = np.where(neigh_vix == vix)[0][0]  # central point
#         mask_pos = np.where(mask[vix] == 1.0)[0]  # nonzero elements
#         patch_rho = rho[vix][mask_pos]  # nonzero elements of rho
#         ddc = compute_ddc(patch_v, patch_n, patch_cp, patch_rho)
#
#         input_feat[vix, :len(neigh_vix), 0] = si[neigh_vix]
#         input_feat[vix, :len(neigh_vix), 1] = ddc
#         input_feat[vix, :len(neigh_vix), 2] = hbond[neigh_vix]
#         input_feat[vix, :len(neigh_vix), 3] = charge[neigh_vix]
#         input_feat[vix, :len(neigh_vix), 4] = hphob[neigh_vix]
#
#     return input_feat, iface_labels


# From a full shape in a full protein, extract a patch around a vertex.
# If patch_indices = True, then store the indices of all neighbors.
# def extract_patch_and_coord(vix, shape, coord, max_distance, max_vertices, patch_indices=False):
#     # Member vertices are nonzero elements
#     i, j = coord[np.int(vix), : coord.shape[1] // 2].nonzero()
#
#     # D = np.squeeze(np.asarray(coord[np.int(vix),j].todense()))
#     D = np.squeeze(np.asarray(coord[np.int(vix), : coord.shape[1] // 2].todense()))
#     j = np.where((D < max_distance) & (D > 0))[0]
#     max_dist_tmp = max_distance
#     old_j = len(j)
#     while len(j) > max_vertices:
#         max_dist_tmp = max_dist_tmp * 0.95
#         j = np.where((D < max_dist_tmp) & (D > 0))[0]
#     #    print('j = {} {}'.format(len(j), old_j))
#     D = D[j]
#     patch = {}
#     patch["X"] = shape["X"][0][j]
#     patch["Y"] = shape["Y"][0][j]
#     patch["Z"] = shape["Z"][0][j]
#     patch["charge"] = shape["charge"][0][j]
#     patch["hbond"] = shape["hbond"][0][j]
#     patch["normal"] = shape["normal"][:, j]
#     patch["shape_index"] = shape["shape_index"][0][j]
#     if "hphob" in shape:
#         patch["hphob"] = shape["hphob"][0][j]
#
#     patch["center"] = np.argmin(D)
#
#     j_theta = j + coord.shape[1] // 2
#     theta = np.squeeze(np.asarray(coord[np.int(vix), j_theta].todense()))
#     coord = np.concatenate([D, theta], axis=0)
#
#     if patch_indices:
#         return patch, coord, j
#     else:
#         return patch, coord


# neigh1 and neigh2 are the precomputed indices; rho1 and rho2 their distances.
# def compute_shape_complementarity(ply_fn1, ply_fn2, neigh1, neigh2, rho1, rho2, mask1, mask2, params):
#     """
#         compute_shape_complementarity: compute the shape complementarity between all pairs of patches.
#         ply_fnX: path to the ply file of the surface of protein X=1 and X=2
#         neighX, rhoX, maskX: (N,max_vertices_per_patch) matrices with the indices of the neighbors, the distances to the center
#                 and the mask
#
#         Returns: vX_sc (2,N,10) matrix with the shape complementarity (shape complementarity 25 and 50)
#         of each vertex to its nearest neighbor in the other protein, in 10 rings.
#     """
#     # Mesh 1
#     mesh1 = pymesh.load_mesh(ply_fn1)
#     # Normals:
#     nx = mesh1.get_attribute("vertex_nx")
#     ny = mesh1.get_attribute("vertex_ny")
#     nz = mesh1.get_attribute("vertex_nz")
#     n1 = np.stack([nx, ny, nz], axis=1)
#
#     # Mesh 2
#     mesh2 = pymesh.load_mesh(ply_fn2)
#     # Normals:
#     nx = mesh2.get_attribute("vertex_nx")
#     ny = mesh2.get_attribute("vertex_ny")
#     nz = mesh2.get_attribute("vertex_nz")
#     n2 = np.stack([nx, ny, nz], axis=1)
#
#     w = params['sc_w']
#     int_cutoff = params['sc_interaction_cutoff']
#     radius = params['sc_radius']
#     num_rings = 10
#     scales = np.arange(0, radius, radius / 10)
#     scales = np.append(scales, radius)
#
#     v1 = mesh1.vertices
#     v2 = mesh2.vertices
#
#     v1_sc = np.zeros((2, len(v1), 10))
#     v2_sc = np.zeros((2, len(v2), 10))
#
#     # Find all interface vertices
#     kdt = cKDTree(v2)
#     d, nearest_neighbors_v1_to_v2 = kdt.query(v1)
#     # Interface vertices in v1
#     interface_vertices_v1 = np.where(d < int_cutoff)[0]
#
#     # Go through every interface vertex.
#     for cv1_iiix in range(len(interface_vertices_v1)):
#         cv1_ix = interface_vertices_v1[cv1_iiix]
#         assert (d[cv1_ix] < int_cutoff)
#         # First shape complementarity s1->s2 for the entire patch
#         patch_idxs1 = np.where(mask1[cv1_ix] == 1)[0]
#         neigh_cv1 = np.array(neigh1[cv1_ix])[patch_idxs1]
#         # Find the point cv2_ix in s2 that is closest to cv1_ix
#         cv2_ix = nearest_neighbors_v1_to_v2[cv1_ix]
#         patch_idxs2 = np.where(mask2[cv2_ix] == 1)[0]
#         neigh_cv2 = np.array(neigh2[cv2_ix])[patch_idxs2]
#
#         patch_v1 = v1[neigh_cv1]
#         patch_v2 = v2[neigh_cv2]
#         patch_n1 = n1[neigh_cv1]
#         patch_n2 = n2[neigh_cv2]
#
#         patch_kdt = cKDTree(patch_v1)
#         p_dists_v2_to_v1, p_nearest_neighbor_v2_to_v1 = patch_kdt.query(patch_v2)
#         patch_kdt = cKDTree(patch_v2)
#         p_dists_v1_to_v2, p_nearest_neighbor_v1_to_v2 = patch_kdt.query(patch_v1)
#
#         # First v1->v2
#         neigh_cv1_p = p_nearest_neighbor_v1_to_v2
#         comp1 = [np.dot(patch_n1[x], -patch_n2[neigh_cv1_p][x]) for x in range(len(patch_n1))]
#         comp1 = np.multiply(comp1, np.exp(-w * np.square(p_dists_v1_to_v2)))
#         # Use 10 rings such that each ring has equal weight in shape complementarity
#         comp_rings1_25 = np.zeros(num_rings)
#         comp_rings1_50 = np.zeros(num_rings)
#
#         patch_rho1 = np.array(rho1[cv1_ix])[patch_idxs1]
#         for ring in range(num_rings):
#             scale = scales[ring]
#             members = np.where((patch_rho1 >= scales[ring]) & (patch_rho1 < scales[ring + 1]))
#             if len(members[0]) == 0:
#                 comp_rings1_25[ring] = 0.0
#                 comp_rings1_50[ring] = 0.0
#             else:
#                 comp_rings1_25[ring] = np.percentile(comp1[members], 25)
#                 comp_rings1_50[ring] = np.percentile(comp1[members], 50)
#
#         # Now v2->v1
#         neigh_cv2_p = p_nearest_neighbor_v2_to_v1
#         comp2 = [np.dot(patch_n2[x], -patch_n1[neigh_cv2_p][x]) for x in range(len(patch_n2))]
#         comp2 = np.multiply(comp2, np.exp(-w * np.square(p_dists_v2_to_v1)))
#         # Use 10 rings such that each ring has equal weight in shape complementarity
#         comp_rings2_25 = np.zeros(num_rings)
#         comp_rings2_50 = np.zeros(num_rings)
#
#         # Apply mask to patch rho coordinates.
#         patch_rho2 = np.array(rho2[cv2_ix])[patch_idxs2]
#         for ring in range(num_rings):
#             scale = scales[ring]
#             members = np.where((patch_rho2 >= scales[ring]) & (patch_rho2 < scales[ring + 1]))
#             if len(members[0]) == 0:
#                 comp_rings2_25[ring] = 0.0
#                 comp_rings2_50[ring] = 0.0
#             else:
#                 comp_rings2_25[ring] = np.percentile(comp2[members], 25)
#                 comp_rings2_50[ring] = np.percentile(comp2[members], 50)
#
#         v1_sc[0, cv1_ix, :] = comp_rings1_25
#         v2_sc[0, cv2_ix, :] = comp_rings2_25
#         v1_sc[1, cv1_ix, :] = comp_rings1_50
#         v2_sc[1, cv2_ix, :] = comp_rings2_50
#
#     return v1_sc, v2_sc


def normalize_electrostatics(in_elec):
    """
        Normalize electrostatics to a value between -1 and 1
    """
    elec = np.copy(in_elec)
    upper_threshold = 3
    lower_threshold = -3
    elec[elec > upper_threshold] = upper_threshold
    elec[elec < lower_threshold] = lower_threshold
    elec = elec - lower_threshold
    elec = elec / (upper_threshold - lower_threshold)
    elec = 2 * elec - 1
    return elec


def mean_normal_center_patch(D, n, r):
    """
        Function to compute the mean normal of vertices within r radius of the center of the patch.
    """
    c_normal = [n[i] for i in range(len(D)) if D[i] <= r]
    mean_normal = np.mean(c_normal, axis=0, keepdims=True).T
    mean_normal = mean_normal / np.linalg.norm(mean_normal)
    return np.squeeze(mean_normal)


def compute_ddc(patch_v, patch_n, patch_cp, patch_rho):
    """
        Compute the distance dependent curvature, Yin et al PNAS 2009
            patch_v: the patch vertices
            patch_n: the patch normals
            patch_cp: the index of the central point of the patch
            patch_rho: the geodesic distance to all members.
        Returns a vector with the ddc for each point in the patch.
    """
    n = patch_n
    r = patch_v
    i = patch_cp
    # Compute the mean normal 2.5A around the center point
    ni = mean_normal_center_patch(patch_rho, n, 2.5)
    dij = np.linalg.norm(r - r[i], axis=1)
    # Compute the step function sf:
    sf = r + n
    sf = sf - (ni + r[i])
    sf = np.linalg.norm(sf, axis=1)
    sf = sf - dij
    sf[sf > 0] = 1
    sf[sf < 0] = -1
    sf[sf == 0] = 0
    # Compute the curvature between i and j
    dij[dij == 0] = 1e-8
    kij = np.divide(np.linalg.norm(n - ni, axis=1), dij)
    kij = np.multiply(sf, kij)
    # Ignore any values greater than 0.7 and any values smaller than 0.7
    kij[kij > 0.7] = 0
    kij[kij < -0.7] = 0

    return kij


def extractProteinTriangulate1(masifpniOpts, pPdbFile, naPdbFile, outPlyFile, outVerticeFile, compute_iface=True):
    pdbFileBase = os.path.splitext(pPdbFile)[0]

    vertices1, faces1, normals1, names1, areas1 = computeMSMS(pPdbFile, masifpniOpts=masifpniOpts, protonate=True,
                                                              protPDB=True, naPDB=False)
    if not vertices1.any() or not faces1.any() or not normals1.any() or not names1 or not areas1:
        return

    # Fix the mesh.
    mesh = pymesh.form_mesh(vertices1, faces1)
    regular_mesh = fixMesh(mesh, masifpniOpts['mesh_res'])

    # Compute the normals
    vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
    # Assign charges on new vertices based on charges of old vertices (nearest neighbor)
    new_names = assignNamesToNewMesh(regular_mesh.vertices, vertices1, names1)
    new_names = np.array(new_names)

    vertex_hbond = np.array([])
    if masifpniOpts['use_hbond']:
        vertex_hbond = computeCharges(pdbFileBase, vertices1, names1)
        vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masifpniOpts)

    vertex_hphobicity = np.array([])
    if masifpniOpts['use_hphob']:
        vertex_hphobicity = computeHydrophobicity(names1)
        vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masifpniOpts)

    vertex_charges = np.array([])
    if masifpniOpts['use_apbs']:
        vertex_charges = computeAPBS(regular_mesh.vertices, pdbFileBase + ".pdb", pdbFileBase)

    if not vertex_charges.any() or not vertex_hbond.any() or not vertex_hphobicity.any() or not vertex_normal.any(): return

    iface = np.zeros(len(regular_mesh.vertices))
    if 'compute_iface' in masifpniOpts and masifpniOpts['compute_iface'] and compute_iface:
        # Compute the surface of the entire complex and from that compute the interface.
        v2, f2, _, _, _ = computeMSMS(naPdbFile, masifpniOpts=masifpniOpts, protonate=True,
                                      protPDB=False, naPDB=True)
        if not v2.any() or not f2.any():
            return

        kdt = cKDTree(regular_mesh.vertices)
        labelIndex = kdt.query_ball_point(v2, 3, p=2., eps=0)
        labelIndex = sorted(set(itertools.chain(*labelIndex)))
        labelIndex = filterLabelByCount(new_names, labelIndex)
        iface[labelIndex] = 1.0

        # Convert to ply and save.
        save_ply(outPlyFile, regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
                 charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
                 iface=iface)
        np.savez(file=outVerticeFile, vertices=regular_mesh.vertices, names=new_names)
    else:
        # Convert to ply and save.
        save_ply(outPlyFile, regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
                 charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
        np.savez(file=outVerticeFile, vertices=regular_mesh.vertices, names=new_names)

    # print(outPlyFile, outVerticeFile)


# def extractProteinTriangulate(masifpniOpts, pdbFile, rawPdbFile):
#     pdbFileBase = os.path.splitext(pdbFile)[0]
#     # tmpOutFileBase = os.path.join(masifpniOpts["tmp_dir"], pdbFileBase)
#
#     vertices1, faces1, normals1, names1, areas1 = computeMSMS(pdbFile, masifpniOpts=masifpniOpts, protonate=True,
#                                                               protPDB=True, naPDB=False)
#     if not vertices1.any() or not faces1.any() or not normals1.any() or not names1 or not areas1:
#         return
#
#     # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
#     vertices2 = vertices1
#     faces2 = faces1
#
#     # Fix the mesh.
#     mesh = pymesh.form_mesh(vertices2, faces2)
#     regular_mesh = fixMesh(mesh, masifpniOpts['mesh_res'])
#
#     # Compute the normals
#     vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
#     # Assign charges on new vertices based on charges of old vertices (nearest
#     # neighbor)
#     new_names = assignNamesToNewMesh(regular_mesh.vertices, vertices1, names1)
#     new_names = np.array(new_names)
#
#     vertex_hbond = np.array([])
#     if masifpniOpts['use_hbond']:
#         vertex_hbond = computeCharges(pdbFileBase, vertices1, names1)
#         vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masifpniOpts)
#
#     vertex_hphobicity = np.array([])
#     if masifpniOpts['use_hphob']:
#         vertex_hphobicity = computeHydrophobicity(names1)
#         vertex_hphobicity = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hphobicity, masifpniOpts)
#
#     vertex_charges = np.array([])
#     if masifpniOpts['use_apbs']:
#         vertex_charges = computeAPBS(regular_mesh.vertices, pdbFileBase + ".pdb", pdbFileBase)
#
#     if not vertex_charges.any() or not vertex_hbond.any() or not vertex_hphobicity.any() or not vertex_normal.any(): return
#     iface = np.zeros(len(regular_mesh.vertices))
#     if 'compute_iface' in masifpniOpts and masifpniOpts['compute_iface']:
#         # Compute the surface of the entire complex and from that compute the interface.
#         v3, f3, _, _, _ = computeMSMS(rawPdbFile, masifpniOpts=masifpniOpts, protonate=True, protPDB=False, naPDB=False)
#         if not v3.any() or not f3.any():
#             return
#         # Regularize the mesh
#         mesh = pymesh.form_mesh(v3, f3)
#         # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
#         full_regular_mesh = mesh
#         # Find the vertices that are in the iface.
#         v3 = full_regular_mesh.vertices
#         # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
#         kdt = KDTree(v3)
#         d, r = kdt.query(regular_mesh.vertices)
#         d = np.square(d)  # Square d, because this is how it was in the pyflann version.
#         assert (len(d) == len(regular_mesh.vertices))
#         iface_v = np.where(d >= 2.0)[0]
#         iface[iface_v] = 1.0
#         # Convert to ply and save.
#         save_ply(pdbFileBase + ".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
#                  charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity,
#                  iface=iface)
#         np.savez(file=pdbFileBase + ".vertices_name.npz", vertices=regular_mesh.vertices, names=new_names)
#
#     else:
#         # Convert to ply and save.
#         save_ply(pdbFileBase + ".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
#                  charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, hphob=vertex_hphobicity)
#         np.savez(file=pdbFileBase + ".vertices_name.npz", vertices=regular_mesh.vertices, names=new_names)
#
#     if not os.path.exists(masifpniOpts['ply_chain_dir']):
#         os.makedirs(masifpniOpts['ply_chain_dir'])
#     if not os.path.exists(masifpniOpts['pdb_chain_dir']):
#         os.makedirs(masifpniOpts['pdb_chain_dir'])
#
#     makeLink(os.path.realpath(pdbFileBase + '.ply'), os.path.join(masifpniOpts['ply_chain_dir'], os.path.basename(pdbFileBase) + '.ply'))
#     makeLink(os.path.realpath(pdbFileBase + '.pdb'), os.path.join(masifpniOpts['pdb_chain_dir'], os.path.basename(pdbFileBase) + '.pdb'))


# def extractNaTriangulate(masifpniOpts, naPdbFile, pnaPdbFile):
#     pdbFileBase = os.path.splitext(naPdbFile)[0]
#
#     vertices1, faces1, normals1, names1, areas1 = computeMSMS(naPdbFile, masifpniOpts=masifpniOpts, protonate=True,
#                                                               protPDB=False, naPDB=True)
#     if not vertices1.any() or not faces1.any() or not normals1.any() or not names1 or not areas1:
#         return
#     # vertex_hbond = np.array([])
#     # if masifpniOpts['use_hbond']:
#     #     vertex_hbond = computeCharges(tmpOutFileBase, vertices1, names1)
#
#     # For each surface residue, assign the hydrophobicity of its amino acid.
#     # vertex_hphobicity = np.array([])
#     # if masifpniOpts['use_hphob']:
#     #     vertex_hphobicity = computeHydrophobicity(names1)
#
#     # If protonate = false, recompute MSMS of surface, but without hydrogens (set radius of hydrogens to 0).
#     vertices2 = vertices1
#     faces2 = faces1
#
#     # Fix the mesh.
#     mesh = pymesh.form_mesh(vertices2, faces2)
#     regular_mesh = fixMesh(mesh, masifpniOpts['mesh_res'])
#
#     # Compute the normals
#     print(naPdbFile)
#     vertex_normal = compute_normal(regular_mesh.vertices, regular_mesh.faces)
#     # Assign charges on new vertices based on charges of old vertices (nearest
#     # neighbor)
#
#     vertex_hbond = np.array([])
#     # if masifpniOpts['use_hbond']:
#     #     vertex_hbond = computeCharges(pdbFileBase, vertices1, names1)
#     #     vertex_hbond = assignChargesToNewMesh(regular_mesh.vertices, vertices1, vertex_hbond, masifpniOpts)
#
#     vertex_charges = np.array([])
#     # if masifpniOpts['use_apbs']:
#     #     vertex_charges = computeAPBS(regular_mesh.vertices, pdbFileBase + ".pdb", pdbFileBase)
#
#     # if not vertex_charges.any() or not vertex_normal.any(): return
#     iface = np.zeros(len(regular_mesh.vertices))
#     if 'compute_iface' in masifpniOpts and masifpniOpts['compute_iface']:
#         # Compute the surface of the entire complex and from that compute the interface.
#         v3, f3, _, _, _ = computeMSMS(pnaPdbFile, masifpniOpts=masifpniOpts, protonate=True, protPDB=False, naPDB=False)
#         if not v3.any() or not f3.any():
#             return
#         # Regularize the mesh
#         mesh = pymesh.form_mesh(v3, f3)
#         # I believe It is not necessary to regularize the full mesh. This can speed up things by a lot.
#         full_regular_mesh = mesh
#         # Find the vertices that are in the iface.
#         v3 = full_regular_mesh.vertices
#         # Find the distance between every vertex in regular_mesh.vertices and those in the full complex.
#         kdt = KDTree(v3)
#         d, r = kdt.query(regular_mesh.vertices)
#         d = np.square(d)  # Square d, because this is how it was in the pyflann version.
#         assert (len(d) == len(regular_mesh.vertices))
#         iface_v = np.where(d >= 2.0)[0]
#         iface[iface_v] = 1.0
#         # Convert to ply and save.
#         save_ply(pdbFileBase + ".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
#                  charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond, iface=iface)
#
#     else:
#         # Convert to ply and save.
#         save_ply(pdbFileBase + ".ply", regular_mesh.vertices, regular_mesh.faces, normals=vertex_normal,
#                  charges=vertex_charges, normalize_charges=True, hbond=vertex_hbond)
#
#     if not os.path.exists(masifpniOpts['ply_chain_dir']):
#         os.makedirs(masifpniOpts['ply_chain_dir'])
#     if not os.path.exists(masifpniOpts['pdb_chain_dir']):
#         os.makedirs(masifpniOpts['pdb_chain_dir'])
#
#     makeLink(os.path.realpath(pdbFileBase + '.ply'),
#              os.path.join(masifpniOpts['ply_chain_dir'], os.path.basename(pdbFileBase) + '.ply'))
#     makeLink(os.path.realpath(pdbFileBase + '.pdb'),
#              os.path.join(masifpniOpts['pdb_chain_dir'], os.path.basename(pdbFileBase) + '.pdb'))
