#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: parseConfig.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:44:43
Last modified: 2022-09-11 20:44:43
'''

import tempfile, sys, os
import numpy as np
from configparser import ConfigParser, ExtendedInterpolation
# from IPython.core.debugger import set_trace

DIR_SECTION = "default_dirs"
FILE_SECTION = "default_files"
SYS_SECTION = "default_params"
SURFEAT_SECTION = "surface_features"
DOWNLOAD_SECTION = "download_pdb"
MASIFPNISITE_SECTION = "masifpni_site"

SECTION_TYPE_LIST_ORIGIN = [DIR_SECTION, FILE_SECTION, SYS_SECTION, SURFEAT_SECTION, DOWNLOAD_SECTION, MASIFPNISITE_SECTION]
SECTION_TYPE_LIST_LOWER = [i.lower() for i in SECTION_TYPE_LIST_ORIGIN]

DIR_TAGS = [OUT_BASE_DIR, RAW_PDB_DIR, VERTEX_CHAIN_DIR, PLY_CHAIN_DIR, TMP_DIR, LOG_DIR, EXTRACT_PDB, FILTERED_PLY] = \
    ["out_base_dir", "raw_pdb_dir", "vertex_chain_dir", "ply_chain_dir", "tmp_dir", "log_dir", "extract_pdb", "filtered_ply"]

FILE_TAGS = [PDB_INFO, SETTING_LOG, PLY_FILE_TEMPLATE, VERTEX_FILE_TAMPLATE] = \
    ["pdb_info", "setting_log", "ply_file_template", "vertex_file_template"]

SYS_TAGS = [N_THREADS, NO_BATCH_RUN, NA_TYPE, USE_GPU, USE_CPU, GPU_DEV, CPU_DEV, MERGE_NA_CHAIN, CUSTOM_PDB] = \
    ["n_threads", "no_batch_run", "na_type", "use_gpu", "use_cpu", "gpu_dev", "cpu_dev", "merge_na_chain", "custom_pdb"]

DOWNLOAD_TAGS = [PDB_ID_LIST, PDB_ID_FILE, ALL_PDBS, OVERWRITE_PDB] = \
    ["pdb_id_list", "pdb_id_file", "all_pdbs", "overwrite_pdb"]

SURFEAT_TAGS = [USE_HBOND, USE_HPHOB, USE_APBS, COMPUTE_IFACE, MESH_RES, FEATURE_INTERPOLATION, RADIUS] = \
    ["use_hbond", "use_hphob", "use_apbs", "compute_iface", "mesh_res", "feature_interpolation", "radius"]

COMMON_TAGS = [MAX_SHAPE_SIZE, MAX_DISTANCE, MASIF_PRECOMPUTATION_DIR, MODEL_DIR, N_FEAT] = \
    ["max_shape_size", "max_distance", "masif_precomputation_dir", "model_dir", "n_feat"]

MODEL_SHAPE_TAG = [N_THETA, N_RHO, N_ROTATIONS, N_CONV_LAYERS] = \
    ["n_theta", "n_rho", "n_rotations", "n_conv_layers"]

MASIFPNISITE_TAGS = [TRAINING_LIST, TESTING_LIST, RANGE_VAL_SAMPLES, OUT_PRED_DIR, OUT_SURF_DIR, OUT_EVAL_DIR] = \
    ["training_list", "testing_list", "range_val_samples", "out_pred_dir", "out_surf_dir", "out_eval_dir"]

MASIFPNISITE_TAGS = COMMON_TAGS + MODEL_SHAPE_TAG + MASIFPNISITE_TAGS

INTEGER_TAGS = [N_THREADS, MAX_SHAPE_SIZE, N_CONV_LAYERS, N_FEAT, N_THETA, N_RHO, N_ROTATIONS]
FLOAT_TAGS = [MESH_RES, RADIUS, MAX_DISTANCE, RANGE_VAL_SAMPLES]
BOOLEAN_TAGS = [USE_CPU, USE_GPU, USE_APBS, USE_HBOND, USE_HPHOB, COMPUTE_IFACE, FEATURE_INTERPOLATION, NO_BATCH_RUN,
                ALL_PDBS, OVERWRITE_PDB, MERGE_NA_CHAIN]

VALID_TAGS = list(set(DIR_TAGS + FILE_TAGS + SYS_TAGS + SURFEAT_TAGS + DOWNLOAD_TAGS + MASIFPNISITE_TAGS))


DEFAULT_PARAMS_DICT = {
    "config": None, "out_base_dir": None, "pdb_id_list": None, "pdb_id_file": None, "all_pdbs": False,
    "overwrite_pdb": False, "n_threads": 1, "no_batch_run": False, "radius": 9, "max_vertices_n": 100,
    "merge_na_chain": False, "na_type": "RNA", "training_list": None, "testing_list": None, "filterChainByLen": False,
    "draw_roc": False, "model_dir": None, "custom_pdb": None, "n_theta": 4, "n_rho": 3, "n_rotations": 4
}


def check_params_changed_from_command_line(argv):
    param_changed_dict = dict.fromkeys(DEFAULT_PARAMS_DICT.keys(), False)
    for param in vars(argv).keys():
        if param in DEFAULT_PARAMS_DICT:
            changed_flag = False if vars(argv)[param] == DEFAULT_PARAMS_DICT[param] else True
            param_changed_dict.update({param: changed_flag})
    return param_changed_dict


class ParseConfig(object):
    def __init__(self, cfgFile=None, **args):
        self.config = ConfigParser(interpolation=ExtendedInterpolation())
        if cfgFile:
            self.config.read(cfgFile)
            self.validate()
            self.params = self.instantiate()

    def validate(self):
        validSet = set(VALID_TAGS)
        for secId in self.getIds():
            configSet = set(self.config.options(secId))
            badOptions = configSet - validSet
            if badOptions:
                raise ValueError('Unrecognized options found in %s section: %s\nValid options are: %s' % (
                    secId, ', '.join(badOptions), ', '.join(validSet)))

    def instantiate(self):
        params = {}
        tmpMapping = dict(zip([MASIFPNISITE_SECTION.lower()],
                              [MASIFPNISITE_SECTION]))
        for sec in self.getIds():
            if sec.lower() not in SECTION_TYPE_LIST_LOWER:
                raise ValueError("The section name of %s isn't in %s" % (sec, ",".join(SECTION_TYPE_LIST_ORIGIN)))

            if sec.lower() in [DIR_SECTION.lower(), FILE_SECTION.lower(), SYS_SECTION.lower(), SURFEAT_SECTION.lower(),
                               DOWNLOAD_SECTION.lower()]:
                for opt in self.config.options(sec):
                    if self.getValue(sec, opt) == "True":
                        params[opt] = True
                    elif self.getValue(sec, opt) == "False":
                        params[opt] = False
                    elif self.getValue(sec, opt) == "None":
                        params[opt] = None
                    else:
                        params[opt] = self.getValue(sec, opt)
            else:
                params[tmpMapping[sec.lower()]] = {}
                for opt in self.config.options(sec):
                    if self.getValue(sec, opt) == "True":
                        params[tmpMapping[sec.lower()]][opt] = True
                    elif self.getValue(sec, opt) == "False":
                        params[tmpMapping[sec.lower()]][opt] = False
                    elif self.getValue(sec, opt) == "None":
                        params[tmpMapping[sec.lower()]][opt] = None
                    else:
                        params[tmpMapping[sec.lower()]][opt] = self.getValue(sec, opt)
        return params

    def getIds(self):
        """Returns a list of all plot sections found in the config file."""
        return [s for s in self.config.sections()]

    def getValue(self, section, name, default=None):
        """Returns a value from the configuration."""
        tag = name.lower()

        try:
            if tag in FLOAT_TAGS:
                value = self.config.getfloat(section, name)
            elif tag in INTEGER_TAGS:
                value = self.config.getint(section, name)
            elif tag in BOOLEAN_TAGS:
                value = self.config.getboolean(section, name)
            else:
                value = self.config.get(section, name)
        except Exception:
            return default

        return value


class DefaultConfig(object):
    def __init__(self):
        self.masifpniOpts = self.getDefaultConfig()
        self.radii = self.getRadii()
        self.polarHydrogens = self.getPolarHydrogens()
        self.hbond_std_dev = np.pi / 3
        self.acceptorAngleAtom = self.getAcceptorAngleAtom()
        self.acceptorPlaneAtom = self.getAcceptorPlaneAtom()
        self.donorAtom = self.getDonorAtom()

    def getDefaultConfig(self):
        masifpniOpts = {}
        # Default directories
        masifpniOpts["out_base_dir"] = "test_data"
        prepara_dir = os.path.join(masifpniOpts["out_base_dir"], "data_preparation")
        masifpniOpts["raw_pdb_dir"] = os.path.join(prepara_dir, "00-raw_pdbs")
        masifpniOpts["extract_pdb"] = os.path.join(prepara_dir, "01-extract_pdb")
        masifpniOpts["ply_chain_dir"] = os.path.join(prepara_dir, "02a-benchmark_surfaces")
        masifpniOpts["vertex_chain_dir"] = os.path.join(prepara_dir, "02b-benchmark_vertices")
        masifpniOpts["tmp_dir"] = os.path.join(masifpniOpts["out_base_dir"], "tmp")
        masifpniOpts["log_dir"] = os.path.join(masifpniOpts["out_base_dir"], "log")

        # Default files
        masifpniOpts["pdb_info"] = os.path.join(masifpniOpts["out_base_dir"], "pdb_info.npz")
        masifpniOpts["setting_log"] = os.path.join(masifpniOpts["log_dir"], "setting_log.txt")
        masifpniOpts["filtered_ply"] = os.path.join(prepara_dir, "filtered_ply")
        masifpniOpts["ply_file_template"] = os.path.join(masifpniOpts["ply_chain_dir"], "{}_{}_{}.ply")
        masifpniOpts["vertex_file_template"] = os.path.join(masifpniOpts["vertex_chain_dir"], "{}_{}_{}.vertices_name.npz")

        # Default system params
        masifpniOpts["n_threads"] = 1
        masifpniOpts["no_batch_run"] = False
        masifpniOpts["na_type"] = "RNA"
        masifpniOpts["merge_na_chain"] = False
        masifpniOpts["use_gpu"] = False
        masifpniOpts["use_cpu"] = True
        masifpniOpts["gpu_dev"] = "/gpu:0"
        masifpniOpts["cpu_dev"] = "/cpu:0"
        masifpniOpts["custom_pdb"] = None

        # Default download params
        masifpniOpts["pdb_id_list"] = None
        masifpniOpts["pdb_id_file"] = None
        masifpniOpts["all_pdbs"] = False
        masifpniOpts["overwrite_pdb"] = False

        # Surface features
        masifpniOpts["use_hbond"] = True
        masifpniOpts["use_hphob"] = True
        masifpniOpts["use_apbs"] = True
        masifpniOpts["compute_iface"] = True
        # Mesh resolution. Everything gets very slow if it is lower than 1.0
        masifpniOpts["mesh_res"] = 1.0
        masifpniOpts["feature_interpolation"] = True
        # Coords params
        masifpniOpts["radius"] = 9.0

        # Neural network patch application specific parameters.
        masifpniOpts["masifpni_site"] = {}
        masifpniOpts["masifpni_site"]["training_list"] = "defaultFiles/masifpni_site_train.txt"
        masifpniOpts["masifpni_site"]["testing_list"] = "defaultFiles/masifpni_site_test.txt"
        masifpniOpts["masifpni_site"]["max_shape_size"] = 100
        masifpniOpts["masifpni_site"]["n_conv_layers"] = 3
        masifpniOpts["masifpni_site"]["n_theta"] = 4
        masifpniOpts["masifpni_site"]["n_rho"] = 3
        masifpniOpts["masifpni_site"]["n_rotations"] = 4
        masifpniOpts["masifpni_site"]["max_distance"] = 9.0  # Radius for the neural network.
        run_name = "{}_{}".format(masifpniOpts["masifpni_site"]["max_distance"], masifpniOpts["masifpni_site"]["max_shape_size"])
        masifpniOpts["masifpni_site"]["masif_precomputation_dir"] = os.path.join(prepara_dir, "03-precomputation", "site", run_name)
        masifpniOpts["masifpni_site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifpniOpts["masifpni_site"]["model_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "training", run_name, "nn_models")
        masifpniOpts["masifpni_site"]["out_pred_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_data")
        masifpniOpts["masifpni_site"]["out_surf_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_surfaces")
        masifpniOpts["masifpni_site"]["out_eval_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "eval_results")
        masifpniOpts["masifpni_site"]["n_feat"] = 5

        return masifpniOpts

    def updateDefaultFiles(self, out_dir):
        masifpniOpts = {}
        # Default directories
        masifpniOpts["out_base_dir"] = out_dir
        prepara_dir = os.path.join(masifpniOpts["out_base_dir"], "data_preparation")
        masifpniOpts["raw_pdb_dir"] = os.path.join(prepara_dir, "00-raw_pdbs")
        masifpniOpts["extract_pdb"] = os.path.join(prepara_dir, "01-extract_pdb")
        masifpniOpts["ply_chain_dir"] = os.path.join(prepara_dir, "02a-benchmark_surfaces")
        masifpniOpts["vertex_chain_dir"] = os.path.join(prepara_dir, "02b-benchmark_vertices")
        masifpniOpts["tmp_dir"] = os.path.join(masifpniOpts["out_base_dir"], "tmp")
        masifpniOpts["log_dir"] = os.path.join(masifpniOpts["out_base_dir"], "log")

        # Default files
        masifpniOpts["pdb_info"] = os.path.join(masifpniOpts["out_base_dir"], "pdb_info.npz")
        masifpniOpts["setting_log"] = os.path.join(masifpniOpts["log_dir"], "setting_log.txt")
        masifpniOpts["filtered_ply"] = os.path.join(prepara_dir, "filtered_ply")
        masifpniOpts["ply_file_template"] = os.path.join(masifpniOpts["ply_chain_dir"], "{}_{}_{}.ply")
        masifpniOpts["vertex_file_template"] = os.path.join(masifpniOpts["vertex_chain_dir"], "{}_{}_{}.vertices_name.npz")
        return masifpniOpts

    def getListFromFile(self, myFile):
        myList = []
        with open(myFile) as f:
            for i in f.readlines():
                if i.startswith("#"): continue
                myList.append(i.strip())
        return myList

    def getRadii(self):
        radii = {}
        radii["N"] = "1.540000"
        radii["O"] = "1.400000"
        radii["C"] = "1.740000"
        radii["H"] = "1.200000"
        radii["S"] = "1.800000"
        radii["P"] = "1.800000"
        radii["Z"] = "1.39"
        radii["X"] = "0.770000"  ## Radii of CB or CA in disembodied case.

        return radii

    def getPolarHydrogens(self):
        # This  polar hydrogen's names correspond to that of the program Reduce.
        polarHydrogens = {}
        polarHydrogens["ALA"] = ["H"]
        polarHydrogens["GLY"] = ["H"]
        polarHydrogens["SER"] = ["H", "HG"]
        polarHydrogens["THR"] = ["H", "HG1"]
        polarHydrogens["LEU"] = ["H"]
        polarHydrogens["ILE"] = ["H"]
        polarHydrogens["VAL"] = ["H"]
        polarHydrogens["ASN"] = ["H", "HD21", "HD22"]
        polarHydrogens["GLN"] = ["H", "HE21", "HE22"]
        polarHydrogens["ARG"] = ["H", "HH11", "HH12", "HH21", "HH22", "HE"]
        polarHydrogens["HIS"] = ["H", "HD1", "HE2"]
        polarHydrogens["TRP"] = ["H", "HE1"]
        polarHydrogens["PHE"] = ["H"]
        polarHydrogens["TYR"] = ["H", "HH"]
        polarHydrogens["GLU"] = ["H"]
        polarHydrogens["ASP"] = ["H"]
        polarHydrogens["LYS"] = ["H", "HZ1", "HZ2", "HZ3"]
        polarHydrogens["PRO"] = []
        polarHydrogens["CYS"] = ["H"]
        polarHydrogens["MET"] = ["H"]

        return polarHydrogens

    def getAcceptorAngleAtom(self):
        # Dictionary from an acceptor atom to its directly bonded atom on which to
        # compute the angle.
        acceptorAngleAtom = {}
        acceptorAngleAtom["O"] = "C"
        acceptorAngleAtom["O1"] = "C"
        acceptorAngleAtom["O2"] = "C"
        acceptorAngleAtom["OXT"] = "C"
        acceptorAngleAtom["OT1"] = "C"
        acceptorAngleAtom["OT2"] = "C"

        # ASN Acceptor
        acceptorAngleAtom["OD1"] = "CG"

        # ASP
        # Plane: CB-CG-OD1
        # Angle CG-ODX-point: 120
        acceptorAngleAtom["OD2"] = "CG"

        acceptorAngleAtom["OE1"] = "CD"
        acceptorAngleAtom["OE2"] = "CD"

        # HIS Acceptors: ND1, NE2
        # Plane ND1-CE1-NE2
        # Angle: ND1-CE1 : 125.5
        # Angle: NE2-CE1 : 125.5
        acceptorAngleAtom["ND1"] = "CE1"
        acceptorAngleAtom["NE2"] = "CE1"

        # TYR acceptor OH
        # Plane: CE1-CZ-OH
        # Angle: CZ-OH 120
        acceptorAngleAtom["OH"] = "CZ"

        # SER acceptor:
        # Angle CB-OG-X: 120
        acceptorAngleAtom["OG"] = "CB"

        # THR acceptor:
        # Angle: CB-OG1-X: 120
        acceptorAngleAtom["OG1"] = "CB"

        return acceptorAngleAtom

    def getAcceptorPlaneAtom(self):
        # Dictionary from acceptor atom to a third atom on which to compute the plane.
        acceptorPlaneAtom = {}
        acceptorPlaneAtom["O"] = "CA"

        acceptorPlaneAtom["OD1"] = "CB"
        acceptorPlaneAtom["OD2"] = "CB"

        acceptorPlaneAtom["OE1"] = "CG"
        acceptorPlaneAtom["OE2"] = "CG"

        acceptorPlaneAtom["ND1"] = "NE2"
        acceptorPlaneAtom["NE2"] = "ND1"

        acceptorPlaneAtom["OH"] = "CE1"

        return acceptorPlaneAtom

    def getDonorAtom(self):
        # Dictionary from an H atom to its donor atom.
        donorAtom = {}
        donorAtom["H"] = "N"
        # Hydrogen bond information.
        # ARG
        # ARG NHX
        # Angle: NH1, HH1X, point and NH2, HH2X, point 180 degrees.
        # radii from HH: radii[H]
        # ARG NE
        # Angle: ~ 120 NE, HE, point, 180 degrees
        donorAtom["HH11"] = "NH1"
        donorAtom["HH12"] = "NH1"
        donorAtom["HH21"] = "NH2"
        donorAtom["HH22"] = "NH2"
        donorAtom["HE"] = "NE"

        # ASN
        # Angle ND2,HD2X: 180
        # Plane: CG,ND2,OD1
        # Angle CG-OD1-X: 120
        donorAtom["HD21"] = "ND2"
        donorAtom["HD22"] = "ND2"

        # GLU
        # PLANE: CD-OE1-OE2
        # ANGLE: CD-OEX: 120
        # GLN
        # PLANE: CD-OE1-NE2
        # Angle NE2,HE2X: 180
        # ANGLE: CD-OE1: 120
        donorAtom["HE21"] = "NE2"
        donorAtom["HE22"] = "NE2"

        # HIS Donors: ND1, NE2
        # Angle ND1-HD1 : 180
        # Angle NE2-HE2 : 180
        donorAtom["HD1"] = "ND1"
        donorAtom["HE2"] = "NE2"

        # TRP Donor: NE1-HE1
        # Angle NE1-HE1 : 180
        donorAtom["HE1"] = "NE1"

        # LYS Donor NZ-HZX
        # Angle NZ-HZX : 180
        donorAtom["HZ1"] = "NZ"
        donorAtom["HZ2"] = "NZ"
        donorAtom["HZ3"] = "NZ"

        # TYR donor: OH-HH
        # Angle: OH-HH 180
        donorAtom["HH"] = "OH"

        # SER donor:
        # Angle: OG-HG-X: 180
        donorAtom["HG"] = "OG"

        # THR donor:
        # Angle: OG1-HG1-X: 180
        donorAtom["HG1"] = "OG1"

        return donorAtom

    def update(self, masifpniOpts):
        updatedFiles = self.updateDefaultFiles(masifpniOpts["out_base_dir"])
        for key in updatedFiles:
            masifpniOpts[key] = updatedFiles[key]
        prepara_dir = os.path.join(masifpniOpts["out_base_dir"], "data_preparation")
        run_name = "{}_{}".format(masifpniOpts["masifpni_site"]["max_distance"], masifpniOpts["masifpni_site"]["max_shape_size"])
        masifpniOpts["masifpni_site"]["masif_precomputation_dir"] = os.path.join(prepara_dir, "03-precomputation", "site", run_name)
        # masifpniOpts["masifpni_site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifpniOpts["masifpni_site"]["model_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "training", run_name, "nn_models")
        masifpniOpts["masifpni_site"]["out_pred_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_data")
        masifpniOpts["masifpni_site"]["out_surf_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_surfaces")
        masifpniOpts["masifpni_site"]["out_eval_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "eval_results")
        return masifpniOpts

    def updateRunName(self, masifpniOpts):
        prepara_dir = os.path.join(masifpniOpts["out_base_dir"], "data_preparation")
        run_name = "{}_{}".format(masifpniOpts["masifpni_site"]["max_distance"], masifpniOpts["masifpni_site"]["max_shape_size"])
        masifpniOpts["masifpni_site"]["masif_precomputation_dir"] = os.path.join(prepara_dir, "03-precomputation", "site", run_name)
        # masifpniOpts["masifpni_site"]["range_val_samples"] = 0.9  # 0.9 to 1.0
        masifpniOpts["masifpni_site"]["out_pred_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_data")
        masifpniOpts["masifpni_site"]["out_surf_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "pred_surfaces")
        masifpniOpts["masifpni_site"]["out_eval_dir"] = os.path.join(masifpniOpts["out_base_dir"], "site", "pred", run_name, "eval_results")
        return masifpniOpts


class GlobalVars(object):
    def __init__(self):
        self.reduce_bin = ""
        self.msms_bin = ""
        self.pdb2pqr_bin = ""
        self.apbs_bin = ""
        self.multivalue_bin = ""
        self.epsilon = 1.0e-6

    def initation(self):
        utilPath = os.path.abspath("utils")
        reduce_bin = os.path.join(utilPath, "reduce")
        msms_bin = os.path.join(utilPath, "msms")
        pdb2pqr_bin = os.path.join(utilPath, "pdb2pqr_pack", "pdb2pqr")
        apbs_bin = os.path.join(utilPath, "apbs")
        multivalue_bin = os.path.join(utilPath, "multivalue")

        if os.path.exists(reduce_bin):
            self.reduce_bin = reduce_bin
        else:
            print("ERROR: reduce_bin not set. Variable should point to MSMS program.")
            sys.exit(1)

        if os.path.exists(msms_bin):
            self.msms_bin = msms_bin
        else:
            print("ERROR: MSMS_BIN not set. Variable should point to MSMS program.")
            sys.exit(1)

        if os.path.exists(pdb2pqr_bin):
            self.pdb2pqr_bin = pdb2pqr_bin
        else:
            print("ERROR: PDB2PQR_BIN not set. Variable should point to PDB2PQR_BIN program.")
            sys.exit(1)

        if os.path.exists(apbs_bin):
            self.apbs_bin = apbs_bin
        else:
            print("ERROR: APBS_BIN not set. Variable should point to APBS program.")
            sys.exit(1)

        if os.path.exists(multivalue_bin):
            self.multivalue_bin = multivalue_bin
        else:
            print("ERROR: MULTIVALUE_BIN not set. Variable should point to MULTIVALUE program.")
            sys.exit(1)

    def setEnviron(self):
        utilPath = os.path.abspath("utils")
        if "LD_LIBRARY_PATH" not in os.environ:
            os.environ["LD_LIBRARY_PATH"] = ""
        os.environ["LD_LIBRARY_PATH"] += os.pathsep + os.path.join(utilPath, "lib")
