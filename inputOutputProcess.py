#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
File name: inputOutputProcess.py
Author: CrazyHsu @ crazyhsu9627@gmail.com
Created on: 2022-09-11 20:46:54
Last modified: 2022-09-11 20:46:54
'''

import numpy as np
import pandas as pd
import pymesh
import itertools

from commonFuncs import *
from parseConfig1 import GlobalVars

from scipy import spatial
from biopandas.pdb import PandasPdb
from subprocess import Popen, PIPE
from collections import namedtuple
from Bio import SeqIO
from Bio.Seq import reverse_complement
from Bio.PDB import StructureBuilder, PDBParser, Selection, PDBIO, parse_pdb_header
# from Bio.PDB.Polypeptide import three_to_one
from Bio.PDB.PDBIO import Select
from Bio.Data.SCOPData import protein_letters_3to1
from Bio.SeqUtils import IUPACData
PROTEIN_LETTERS = [x.upper() for x in IUPACData.protein_letters_3to1.keys()]


# Exclude disordered atoms.
class NotDisordered(Select):
    def accept_atom(self, atom):
        return not atom.is_disordered() or atom.get_altloc() == "A" or atom.get_altloc() == "1"


# def checkProteinChain(pdbFile, chainId, pdbHandle=None):
#     from Bio.PDB import is_aa
#     if not pdbHandle:
#         parser = PDBParser(QUIET=True)
#         pdbHandle = parser.get_structure(pdbFile, pdbFile)
#
#     chains = Selection.unfold_entities(pdbHandle, "C")
#     for c in chains:
#         if c.get_id() == chainId:
#             return is_aa(c.get_list()[0])
#     return False


# def read_msms_old(file_root):
#     """
#     read_msms: Read an msms output file that was output by MSMS (MSMS is the program we use to build a surface)
#     Pablo Gainza - LPDI STI EPFL 2019
#     Released under an Apache License 2.0
#     """
#     # read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face
#
#     vertfile = open(file_root + ".vert")
#     meshdata = (vertfile.read().rstrip()).split("\n")
#     vertfile.close()
#
#     # Read number of vertices.
#     count = {}
#     header = meshdata[2].split()
#     count["vertices"] = int(header[0])
#     ## Data Structures
#     vertices = np.zeros((count["vertices"], 3))
#     normalv = np.zeros((count["vertices"], 3))
#     atom_id = [""] * count["vertices"]
#     res_id = [""] * count["vertices"]
#     for i in range(3, len(meshdata)):
#         fields = meshdata[i].split()
#         vi = i - 3
#         vertices[vi][0] = float(fields[0])
#         vertices[vi][1] = float(fields[1])
#         vertices[vi][2] = float(fields[2])
#         normalv[vi][0] = float(fields[3])
#         normalv[vi][1] = float(fields[4])
#         normalv[vi][2] = float(fields[5])
#         atom_id[vi] = fields[7]
#         res_id[vi] = fields[9]
#         count["vertices"] -= 1
#
#     # Read faces.
#     facefile = open(file_root + ".face")
#     meshdata = (facefile.read().rstrip()).split("\n")
#     facefile.close()
#
#     # Read number of vertices.
#     header = meshdata[2].split()
#     count["faces"] = int(header[0])
#     faces = np.zeros((count["faces"], 3), dtype=int)
#     normalf = np.zeros((count["faces"], 3))
#
#     for i in range(3, len(meshdata)):
#         fi = i - 3
#         fields = meshdata[i].split()
#         faces[fi][0] = int(fields[0]) - 1
#         faces[fi][1] = int(fields[1]) - 1
#         faces[fi][2] = int(fields[2]) - 1
#         count["faces"] -= 1
#
#     assert count["vertices"] == 0
#     assert count["faces"] == 0
#
#     return vertices, faces, normalv, res_id


def read_msms(file_root):
    """
    read_msms: Read an msms output file that was output by MSMS (MSMS is the program we use to build a surface)
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
    """
    # read the surface from the msms output. MSMS outputs two files: {file_root}.vert and {file_root}.face

    import glob
    count = {}

    vertfiles = glob.glob(file_root + "*.vert")
    meshdata = []
    count["vertices"] = 0
    for i in range(len(vertfiles)):
        vertfile = open(vertfiles[i])
        tmpData = (vertfile.read().rstrip()).split("\n")
        count["vertices"] += int(tmpData[2].split()[0])
        meshdata.extend(tmpData[3:])
        vertfile.close()

    ## Data Structures
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(len(meshdata)):
        fields = meshdata[i].split()
        vi = i
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    # Read faces.
    facefiles = glob.glob(file_root + "*.face")
    meshdata = []
    count["faces"] = 0
    for i in range(len(facefiles)):
        facefile = open(facefiles[i])
        tmpData = (facefile.read().rstrip()).split("\n")
        count["faces"] += int(tmpData[2].split()[0])
        meshdata.extend(tmpData[3:])
        facefile.close()
    faces = np.zeros((count["faces"], 3), dtype=int)
    normalf = np.zeros((count["faces"], 3))

    for i in range(len(meshdata)):
        fi = i
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id


def find_modified_amino_acids(path):
    """
    Contributed by github user jomimc - find modified amino acids in the PDB (e.g. MSE)
    """
    res_set = set()
    for line in open(path, 'r'):
        if line[:6] == 'SEQRES':
            for res in line.split()[4:]:
                res_set.add(res)
    for res in list(res_set):
        if res in PROTEIN_LETTERS:
            res_set.remove(res)
    return res_set


# def getPdbChainSeq(pdbFile):
#     pdb = open(pdbFile, "r").readlines()
#
#     seq = [l for l in pdb if l[0:6] == "SEQRES"]
#     if len(seq) != 0:
#         chain_dict = dict([(l[11], []) for l in seq])
#         for c in list(chain_dict.keys()):
#             chain_seq = [l[19:70].split() for l in seq if l[11] == c]
#             for x in chain_seq:
#                 chain_dict[c].extend(x)
#
#         return chain_dict
#     else:
#         return {}


def getPdbChainSeq1(pdbFile):
    ppdb = PandasPdb()
    pdbStruc = ppdb.read_pdb(pdbFile)
    seqInfo = pdbStruc.df["OTHERS"].loc[pdbStruc.df["OTHERS"].record_name == "SEQRES"]
    if not seqInfo.empty:
        chain_dict = dict([(l[5], []) for l in seqInfo.entry])
        for c in list(chain_dict.keys()):
            chain_seq = [l[13:].split() for l in seqInfo.entry if l[5] == c]
            for x in chain_seq:
                chain_dict[c].extend(x)
    else:
        chain_dict = {}

    return chain_dict


def longLetterToNormalLetter(letter):
    NTP = ["ATP", "GTP", "CTP", "TTP", "UTP"]
    if "A" in letter:
        return "A"
    elif "C" in letter:
        return "C"
    elif "T" in letter and "TM" not in letter and "TP" not in letter:
        return "T"
    elif "G" in letter:
        return "G"
    elif "U" in letter and "UNK" not in letter:
        return "U"
    elif "UNK" in letter:
        return "N"
    elif "I" in letter:
        return "I"
    elif "N" in letter:
        return "N"
    elif "P" in letter:
        return "P"
    elif "TM" in letter:
        return "N"
    elif "ATP" in letter:
        return "A"
    elif "GTP" in letter:
        return "G"
    elif "CTP" in letter:
        return "C"
    elif "TTP" in letter:
        return "T"
    elif "UTP" in letter:
        return "U"
    elif "R" in letter:
        return "R"
    else:
        return "N"


def getPdbChainFullInfo(pdb_id, pdbFile):
    pid2chainSeq = getPdbChainSeq1(pdbFile)
    chains = pid2chainSeq.keys()
    pChains = []
    naChains = []
    naType = {}
    for c in chains:
        sequence = pid2chainSeq[c]
        if len(set(sequence) & set(PROTEIN_LETTERS)) > 0:
            pChains.append(c)
        else:
            sequence = [longLetterToNormalLetter(i) for i in sequence]
            if len(set(sequence)) == 1 and list(set(sequence))[0] == "N": continue
            if "T" in "".join(sequence) or "D" in "".join(sequence):
                naType[c] = "DNA"
                naChains.append(c)
            else:
                naType[c] = "RNA"
                naChains.append(c)

    dnaChain2Seq = {}
    fullInfoList = []
    for i, j in itertools.product(pChains, naChains):
        pSeq = pid2chainSeq[i]
        tmpSeq = []
        for x in pSeq:
            tmpSeq.append(protein_letters_3to1[x] if x in protein_letters_3to1 else "X")
        # pSeq = [protein_letters_3to1[x] for x in pSeq if x in protein_letters_3to1]
        pSeq = "".join(tmpSeq)
        naSeq = pid2chainSeq[j]
        naSeq = [longLetterToNormalLetter(x) for x in naSeq]
        naSeq = "".join(naSeq)
        if naType[j] == "DNA":
            if "U" in naSeq:
                naSeq = naSeq.replace("T", "N")
                naType[j] = "RNA"
            dnaChain2Seq[j] = naSeq
        fullInfoList.append([pdb_id, i, j, naType[j], pSeq, naSeq, len(pSeq), len(naSeq)])
    # print(fullInfoList)

    seq2dnaChain = {}
    for j in dnaChain2Seq:
        if dnaChain2Seq[j] not in seq2dnaChain and reverse_complement(dnaChain2Seq[j]) not in seq2dnaChain:
            seq2dnaChain[dnaChain2Seq[j]] = [j]
        else:
            if dnaChain2Seq[j] in seq2dnaChain:
                seq2dnaChain[dnaChain2Seq[j]].append(j)
            elif reverse_complement(dnaChain2Seq[j]) in seq2dnaChain:
                seq2dnaChain[reverse_complement(dnaChain2Seq[j])].append(j)

    newDnaChain2Seq = {}
    for s in seq2dnaChain:
        for c in seq2dnaChain[s]:
            newDnaChain2Seq[c] = s

    newFullInfoList = []
    for i in fullInfoList:
        if i[2] in newDnaChain2Seq:
            newFullInfoList.append([i[0], i[1], i[2], i[3], i[4], newDnaChain2Seq[i[2]], i[6], i[7]])
        else:
            newFullInfoList.append(i)
    return newFullInfoList
    # return pd.DataFrame(fullInfoList, columns=["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen"])


def getPdbHeaderInfo(pdbFile):
    useHeader = ["name", "head", "idcode", "deposition_date", "release_date", "structure_method", "resolution"]
    tmpDict = parse_pdb_header(pdbFile)
    headerList = [tmpDict[i] for i in useHeader]
    # headerDf = pd.DataFrame(headerDict, index=[0])
    return [headerList]


# def getPdbChainFullInfo_old(pdbIds, pdbDir):
#     rnaLetters = ["A", "C", "G", "X", "U"]
#     dnaLetters = ["DA", "DT", "DC", "DG", "X"]
#     pid2chainSeq = getPdbChainSeq(pdbIds, pdbDir)
#     fullInfoList = []
#     for pdb_id in pdbIds:
#         chains = pid2chainSeq[pdb_id].keys()
#         pChains = []
#         naChains = []
#         naType = {}
#         for c in chains:
#             sequence = pid2chainSeq[pdb_id][c]
#             if set(sequence).issubset(set(rnaLetters)):
#                 naType[c] = "RNA"
#                 naChains.append(c)
#             elif set(sequence).issubset(set(dnaLetters)):
#                 naType[c] = "DNA"
#                 naChains.append(c)
#             else:
#                 pChains.append(c)
#
#         for i, j in itertools.product(pChains, naChains):
#             pSeq = pid2chainSeq[pdb_id][i]
#             pSeq = [protein_letters_3to1[x] for x in pSeq]
#             pSeq = "".join(pSeq)
#             naSeq = pid2chainSeq[pdb_id][j]
#             naSeq = "".join(naSeq)
#             fullInfoList.append([pdb_id, i, j, naType[j], pSeq, naSeq, len(pSeq), len(naSeq)])
#     return pd.DataFrame(fullInfoList, columns=["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen"])


# def getPdbChainLength(pdbIds, pdbDir):
#     pdb2chainLenList = []
#     naLetters = ["A", "T", "C", "G", "X", "U"]
#     for pdb_id in pdbIds:
#         pdbFile = os.path.join(pdbDir, pdb_id + ".pdb")
#         if not os.path.exists(pdbFile): continue
#         naChain2len = {}
#         pChain2len = {}
#         for r in SeqIO.parse(pdbFile, "pdb-seqres"):
#             chainId = r.annotations["chain"]
#             sequence = r.seq
#             chainType = "naChain" if set(sequence).issubset(set(naLetters)) else "pChain"
#             if chainType == "naChain":
#                 naChain2len.update({chainId: len(sequence)})
#             else:
#                 pChain2len.update({chainId: len(sequence)})
#
#         for i, j in itertools.product(pChain2len.keys(), naChain2len.keys()):
#             pdb2chainLenList.append([pdb_id, i, j, pChain2len[i], naChain2len[j]])
#     return pd.DataFrame(pdb2chainLenList, columns=["PDB_id", "pChain", "naChain", "pChainLen", "naChainLen"])


def filterPdbChains(pdbChainFullInfo, pdbHeaderInfo, pChainLen=30, naChainLen=4, resolution=3, structureMethod="x-ray"):
    mergedDf = pd.merge(pdbChainFullInfo, pdbHeaderInfo, how="inner", left_on="PDB_id", right_on="idcode")
    mergedDf = mergedDf[(mergedDf.pChainLen >= pChainLen) &
                        (mergedDf.naChainLen >= naChainLen) &
                        (mergedDf.resolution <= resolution)]
    selectColumns = ["PDB_id", "pChain", "naChain", "naType", "pChainSeq", "naChainSeq", "pChainLen", "naChainLen", "resolution"]
    return mergedDf.loc[:, selectColumns]


def filterByUserFields(pdbChainFullInfo, pdbId2field):
    oneFields = []
    twoFields = []
    threeFields = []
    for i in pdbId2field:
        if len(pdbId2field[i]) == 2:
            for x in pdbId2field[i][1]:
                twoFields.append([pdbId2field[i][0], x])
        elif len(pdbId2field[i]) == 3:
            for x, y in itertools.product(pdbId2field[i][1], pdbId2field[i][2]):
                threeFields.append([pdbId2field[i][0], x, y])
        else:
            oneFields.append(pdbId2field[i])
    oneFields = pd.DataFrame(oneFields, columns=["PDB_id"])
    twoFields = pd.DataFrame(twoFields, columns=["PDB_id", "pChain"])
    threeFields = pd.DataFrame(threeFields, columns=["PDB_id", "pChain", "naChain"])

    key1 = ["PDB_id"]
    key2 = ["PDB_id", "pChain"]
    key3 = ["PDB_id", "pChain", "naChain"]

    myDict = {"oneFields": {"fields": oneFields, "key": key1},
              "twoFields": {"fields": twoFields, "key": key2},
              "threeFields": {"fields": threeFields, "key": key3}}

    myList = []
    for i in myDict:
        key = myDict[i]["key"]
        fields = myDict[i]["fields"]
        i1 = pdbChainFullInfo.set_index(key).index
        i2 = fields.set_index(key).index
        myList.append(pdbChainFullInfo[i1.isin(i2)])

    return pd.concat(myList)


# def findProteinChainBoundNA(pdbFile, pChainId=None, naChainId=None, radius=5.0):
#     pdbId = os.path.basename(pdbFile).split(".")[0]
#     pdbId = pdbId.split("_")[0]
#
#     ppdb = PandasPdb()
#     pdbStruc = ppdb.read_pdb(pdbFile)
#     atomDf = pdbStruc.df["ATOM"]
#
#     DNAChain = atomDf[atomDf["residue_name"].isin(["DA", "DT", "DC", "DG", "DU"])]["chain_id"].tolist()
#     DNAChain = sorted(set(DNAChain))
#     if not DNAChain:
#         DNAChain = []
#
#     RNAChain = atomDf[atomDf["residue_name"].isin(["A", "T", "C", "G", "U"])]["chain_id"].tolist()
#     RNAChain = sorted(set(RNAChain))
#     if not RNAChain:
#         RNAChain = []
#
#     NAChain = RNAChain + DNAChain
#
#     if pChainId:
#         proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#         atomTree = spatial.cKDTree(proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values)
#     else:
#         atomTree = spatial.cKDTree(atomDf[['x_coord', 'y_coord', 'z_coord']].values)
#
#     boundGroup = []
#     # BoundTuple = namedtuple("BoundTuple", ["PDB_id", "pChain", "naChain", "naType"])
#
#     if naChainId:
#         NAxyz = atomDf.loc[atomDf["chain_id"].isin([naChainId])][['x_coord', 'y_coord', 'z_coord']].values
#         atomsNearNA = atomTree.query_ball_point(NAxyz, radius, p=2., eps=0)
#         nearNAIndex = sorted(set(itertools.chain(*atomsNearNA)))
#
#         if nearNAIndex:
#             if pChainId:
#                 proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#                 proteinChainBoundNA = sorted(set(proteinAtomDf.iloc[nearNAIndex]['chain_id'].unique()) - set(NAChain))[0]
#                 naChainType = "RNA" if naChainId in RNAChain else "DNA"
#                 if proteinChainBoundNA == pChainId:
#                     boundGroup.append(BoundTuple(pdbId, pChainId, naChainId, naChainType))
#             else:
#                 proteinChainBoundNA = sorted(set(atomDf.iloc[nearNAIndex]['chain_id'].unique()) - set(NAChain))
#                 naChainType = "RNA" if naChainId in RNAChain else "DNA"
#                 if proteinChainBoundNA:
#                     for pChain in proteinChainBoundNA:
#                         boundGroup.append(BoundTuple(pdbId, pChain, naChainId, naChainType))
#     else:
#         RNAxyz = atomDf.loc[atomDf["chain_id"].isin(RNAChain)][['x_coord', 'y_coord', 'z_coord']].values
#         if RNAxyz.any():
#             atomsNearRNA = atomTree.query_ball_point(RNAxyz, radius, p=2., eps=0)
#             nearRNAIndex = sorted(set(itertools.chain(*atomsNearRNA)))
#             if nearRNAIndex:
#                 proteinChainBoundRNA = sorted(set(atomDf.iloc[nearRNAIndex]['chain_id'].unique()) - set(NAChain))
#                 if proteinChainBoundRNA:
#                     for pChain in proteinChainBoundRNA:
#                         for i in RNAChain:
#                             boundGroup.append(BoundTuple(pdbId, pChain, i, "RNA"))
#
#         DNAxyz = atomDf.loc[atomDf["chain_id"].isin(DNAChain)][['x_coord', 'y_coord', 'z_coord']].values
#         if DNAxyz.any():
#             atomsNearDNA = atomTree.query_ball_point(DNAxyz, radius, p=2., eps=0)
#             nearDNAIndex = sorted(set(itertools.chain(*atomsNearDNA)))
#             if nearDNAIndex:
#                 proteinChainBoundDNA = sorted(set(atomDf.iloc[nearDNAIndex]['chain_id'].unique()) - set(NAChain))
#                 if proteinChainBoundDNA:
#                     for pChain in proteinChainBoundDNA:
#                         for i in DNAChain:
#                             boundGroup.append(BoundTuple(pdbId, pChain, i, "DNA"))
#
#     return boundGroup


def findProteinChainBoundNA2(pdbId, pdbFile, pChainIds=None, naChainIds=None, radius=5, interactFraq=None):
    ppdb = PandasPdb()
    pdbStruc = ppdb.read_pdb(pdbFile)
    atomDf = pdbStruc.df["ATOM"]
    hetatmDf = pdbStruc.df["HETATM"]
    atomDf = pd.concat([atomDf, hetatmDf])

    allProteinAtomDf = atomDf[atomDf["residue_name"].isin(PROTEIN_LETTERS)]
    allProteinChains = list(allProteinAtomDf.chain_id.unique())
    validNaStrPattern = "A|T|C|G|U"
    allNaAtomDf = atomDf[~atomDf["residue_name"].isin(PROTEIN_LETTERS)]
    allNaAtomDf = allNaAtomDf[allNaAtomDf["residue_name"].str.contains(validNaStrPattern)]
    dnaChain = allNaAtomDf[allNaAtomDf["residue_name"].str.contains("T")]["chain_id"].unique().tolist()
    dnaChain = list(set(dnaChain) - set(allProteinChains))
    rnaChain = allNaAtomDf[allNaAtomDf["residue_name"].str.contains("U")]["chain_id"].unique().tolist()
    rnaChain = list(set(rnaChain) - set(allProteinChains))

    filteredAtomDf = pd.concat([allProteinAtomDf, allNaAtomDf])

    if set(dnaChain) & set(rnaChain):
        dnaNum = allNaAtomDf[allNaAtomDf["residue_name"].str.contains("T")]["chain_id"].shape[0]
        rnaNum = allNaAtomDf[allNaAtomDf["residue_name"].str.contains("U")]["chain_id"].shape[0]
        ambiChains = set(dnaChain) & set(rnaChain)
        if rnaNum > dnaNum:
            dnaChain = list(set(dnaChain) - ambiChains)
        else:
            rnaChain = list(set(rnaChain) - ambiChains)

    allNaChains = rnaChain + dnaChain

    if pChainIds:
        pChainIds = list(pChainIds)
        proteinAtomDf = filteredAtomDf[filteredAtomDf["chain_id"].isin(pChainIds)]
        if proteinAtomDf.empty: return []
        atomTree = spatial.cKDTree(proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values)
    else:
        if filteredAtomDf[['x_coord', 'y_coord', 'z_coord']].empty: return []
        atomTree = spatial.cKDTree(filteredAtomDf[['x_coord', 'y_coord', 'z_coord']].values)

    boundGroup = []

    if naChainIds:
        naChainIds = list(naChainIds)
        naCoord = filteredAtomDf.loc[filteredAtomDf["chain_id"].isin(naChainIds)][['x_coord', 'y_coord', 'z_coord']].values
        if not naCoord.any(): return []
        atomsNearNa = atomTree.query_ball_point(naCoord, radius, p=2., eps=0)
        nearNaIndex = sorted(set(itertools.chain(*atomsNearNa)))

        if interactFraq and len(nearNaIndex) / float(len(atomsNearNa)) < interactFraq: return []

        if nearNaIndex:
            if pChainIds:
                proteinAtomDf = filteredAtomDf[filteredAtomDf["chain_id"].isin(pChainIds)]
                proteinChainBoundNA = sorted(set(proteinAtomDf.iloc[nearNaIndex]['chain_id'].unique()) - set(allNaChains))
            else:
                proteinChainBoundNA = sorted(set(filteredAtomDf.iloc[nearNaIndex]['chain_id'].unique()) - set(allNaChains))
            if proteinChainBoundNA:
                for pChain in proteinChainBoundNA:
                    boundGroup.append(BoundTuple(pdbId, pChain, "".join(naChainIds), ""))

    else:
        naCoord = filteredAtomDf.loc[filteredAtomDf["chain_id"].isin(allNaChains)][['x_coord', 'y_coord', 'z_coord']].values
        if not naCoord.any(): return []
        atomsNearNa = atomTree.query_ball_point(naCoord, radius, p=2., eps=0)
        nearNaIndex = sorted(set(itertools.chain(*atomsNearNa)))

        if interactFraq and len(nearNaIndex) / float(len(atomsNearNa)) < interactFraq: return []

        if nearNaIndex:
            proteinChainBoundNA = sorted(set(filteredAtomDf.iloc[nearNaIndex]['chain_id'].unique()) - set(allNaChains))
            naChainBoundProtein = sorted(set(filteredAtomDf.iloc[nearNaIndex]['chain_id'].unique()) & set(allNaChains))
            if proteinChainBoundNA and naChainBoundProtein:
                for pChain, naChain in itertools.product(proteinChainBoundNA, naChainBoundProtein):
                    boundGroup.append(BoundTuple(pdbId, pChain, naChain, ""))
    return boundGroup


# def findProteinChainBoundNA1(pdbFile, pChainId=None, naChainId=None, radius=5.0, interactFraq=None):
#     pdbId = os.path.basename(pdbFile).split(".")[0]
#     pdbId = pdbId.split("_")[0]
#
#     ppdb = PandasPdb()
#     pdbStruc = ppdb.read_pdb(pdbFile)
#     atomDf = pdbStruc.df["ATOM"]
#
#     DNAChain = atomDf[atomDf["residue_name"].isin(["DA", "DT", "DC", "DG", "DU"])]["chain_id"].tolist()
#     DNAChain = sorted(set(DNAChain))
#     if not DNAChain:
#         DNAChain = []
#
#     RNAChain = atomDf[atomDf["residue_name"].isin(["A", "T", "C", "G", "U"])]["chain_id"].tolist()
#     RNAChain = sorted(set(RNAChain))
#     if not RNAChain:
#         RNAChain = []
#
#     NAChain = RNAChain + DNAChain
#
#     if pChainId:
#         proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#         atomTree = spatial.cKDTree(proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values)
#     else:
#         atomTree = spatial.cKDTree(atomDf[['x_coord', 'y_coord', 'z_coord']].values)
#
#     boundGroup = []
#     # BoundTuple = namedtuple("BoundTuple", ["PDB_id", "pChain", "naChain", "naType"])
#
#     if naChainId:
#         for chain in [naChainId]:
#             NAxyz = atomDf.loc[atomDf["chain_id"].isin([chain])][['x_coord', 'y_coord', 'z_coord']].values
#             atomsNearNA = atomTree.query_ball_point(NAxyz, radius, p=2., eps=0)
#             nearNAIndex = sorted(set(itertools.chain(*atomsNearNA)))
#
#             if nearNAIndex:
#                 if pChainId:
#                     proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#                     proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                     naTree = spatial.cKDTree(NAxyz)
#                     naNearProtein = naTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                     naNearProteinLen = len(sorted(set(itertools.chain(*naNearProtein))))
#                     if interactFraq and naNearProteinLen / float(len(NAxyz)) < interactFraq: continue
#
#                     proteinChainBoundNA = sorted(set(proteinAtomDf.iloc[nearNAIndex]['chain_id'].unique()) - set(NAChain))[0]
#                     naChainType = "RNA" if chain in RNAChain else "DNA"
#                     if proteinChainBoundNA == pChainId:
#                         boundGroup.append(BoundTuple(pdbId, pChainId, chain, naChainType))
#                 else:
#                     proteinChainBoundNA = sorted(set(atomDf.iloc[nearNAIndex]['chain_id'].unique()) - set(NAChain))
#                     naChainType = "RNA" if naChainId in RNAChain else "DNA"
#                     if proteinChainBoundNA:
#                         for pChain in proteinChainBoundNA:
#                             proteinAtomDf = atomDf[atomDf["chain_id"] == pChain]
#                             proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                             naTree = spatial.cKDTree(NAxyz)
#                             naNearProtein = naTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                             naNearProteinLen = len(sorted(set(itertools.chain(*naNearProtein))))
#                             if interactFraq and naNearProteinLen / float(len(NAxyz)) < interactFraq: continue
#
#                             boundGroup.append(BoundTuple(pdbId, pChain, chain, naChainType))
#     else:
#         for chain in RNAChain:
#             RNAxyz = atomDf.loc[atomDf["chain_id"].isin([chain])][['x_coord', 'y_coord', 'z_coord']].values
#             if RNAxyz.any():
#                 atomsNearRNA = atomTree.query_ball_point(RNAxyz, radius, p=2., eps=0)
#                 nearRNAIndex = sorted(set(itertools.chain(*atomsNearRNA)))
#                 if nearRNAIndex:
#                     proteinChainBoundRNA = sorted(set(atomDf.iloc[nearRNAIndex]['chain_id'].unique()) - set(NAChain))
#                     if proteinChainBoundRNA:
#                         if pChainId and pChainId in proteinChainBoundRNA:
#                             proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#                             proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                             rnaTree = spatial.cKDTree(RNAxyz)
#                             rnaNearProtein = rnaTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                             rnaNearProteinLen = len(sorted(set(itertools.chain(*rnaNearProtein))))
#                             if interactFraq and rnaNearProteinLen / float(len(RNAxyz)) < interactFraq: continue
#                             boundGroup.append(BoundTuple(pdbId, pChainId, chain, "RNA"))
#                         else:
#                             for pChain in proteinChainBoundRNA:
#                                 proteinAtomDf = atomDf[atomDf["chain_id"] == pChain]
#                                 proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                                 rnaTree = spatial.cKDTree(RNAxyz)
#                                 rnaNearProtein = rnaTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                                 rnaNearProteinLen = len(sorted(set(itertools.chain(*rnaNearProtein))))
#                                 if interactFraq and rnaNearProteinLen / float(len(RNAxyz)) < interactFraq: continue
#                                 boundGroup.append(BoundTuple(pdbId, pChain, chain, "RNA"))
#         for chain in DNAChain:
#             DNAxyz = atomDf.loc[atomDf["chain_id"].isin([chain])][['x_coord', 'y_coord', 'z_coord']].values
#             if DNAxyz.any():
#                 atomsNearDNA = atomTree.query_ball_point(DNAxyz, radius, p=2., eps=0)
#                 nearDNAIndex = sorted(set(itertools.chain(*atomsNearDNA)))
#                 if nearDNAIndex:
#                     proteinChainBoundDNA = sorted(set(atomDf.iloc[nearDNAIndex]['chain_id'].unique()) - set(NAChain))
#                     if proteinChainBoundDNA:
#                         if pChainId and pChainId in proteinChainBoundDNA:
#                             proteinAtomDf = atomDf[atomDf["chain_id"] == pChainId]
#                             proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                             dnaTree = spatial.cKDTree(DNAxyz)
#                             dnaNearProtein = dnaTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                             dnaNearProteinLen = len(sorted(set(itertools.chain(*dnaNearProtein))))
#                             if interactFraq and dnaNearProteinLen / float(len(DNAxyz)) < interactFraq: continue
#                             boundGroup.append(BoundTuple(pdbId, pChainId, chain, "DNA"))
#                         else:
#                             for pChain in proteinChainBoundDNA:
#                                 proteinAtomDf = atomDf[atomDf["chain_id"] == pChain]
#                                 proteinxyz = proteinAtomDf[['x_coord', 'y_coord', 'z_coord']].values
#                                 dnaTree = spatial.cKDTree(DNAxyz)
#                                 dnaNearProtein = dnaTree.query_ball_point(proteinxyz, radius, p=2., eps=0)
#                                 dnaNearProteinLen = len(sorted(set(itertools.chain(*dnaNearProtein))))
#                                 if interactFraq and dnaNearProteinLen / float(len(DNAxyz)) < interactFraq: continue
#                                 boundGroup.append(BoundTuple(pdbId, pChain, chain, "DNA"))
#
#     return boundGroup


# def filterIfaceLabel(pPlyFile, naPlyFile, pid, pChain, naChain, outDir):
#     if os.path.exists(pPlyFile) and os.path.exists(naPlyFile):
#         pPlyMesh = pymesh.load_mesh(pPlyFile)
#         naPlyMesh = pymesh.load_mesh(naPlyFile)
#         pPlyLabelVertices = pPlyMesh.vertices[np.where(pPlyMesh.get_attribute("vertex_iface") == 1)[0]]
#         naPlyLabelVertices = naPlyMesh.vertices[np.where(naPlyMesh.get_attribute("vertex_iface") == 1)[0]]
#         pPlyMeshIface = np.zeros(len(pPlyMesh.vertices))
#         naPlyMeshIface = np.zeros(len(naPlyMesh.vertices))
#
#         naKDT = spatial.cKDTree(naPlyLabelVertices)
#         naNearProtIndex = naKDT.query_ball_point(pPlyLabelVertices, r=3)
#
#         filteredPPlyLabelVertices = pPlyLabelVertices[naNearProtIndex.astype(bool)]
#         filteredNaPlyLabelVertices = naPlyLabelVertices[sorted(set(itertools.chain(*naNearProtIndex)))]
#
#         pPlyNewLabelIndex = np.where((pPlyMesh.vertices == filteredPPlyLabelVertices[:, None]).all(-1))[1]
#         pPlyMeshIface[pPlyNewLabelIndex] = 1.0
#
#         naPlyNewLabelIndex = np.where((naPlyMesh.vertices == filteredNaPlyLabelVertices[:, None]).all(-1))[1]
#         naPlyMeshIface[naPlyNewLabelIndex] = 1.0
#
#         pPlyMesh.set_attribute("vertex_iface", pPlyMeshIface)
#         naPlyMesh.set_attribute("vertex_iface", naPlyMeshIface)
#
#         filterProtPlyFile = os.path.join(outDir, "{}.{}_{}.{}.ply".format(pid, pChain, naChain, pChain))
#         filterNaPlyFile = os.path.join(outDir, "{}.{}_{}.{}.ply".format(pid, pChain, naChain, naChain))
#
#         pymesh.save_mesh(filterProtPlyFile, pPlyMesh, *pPlyMesh.get_attribute_names()[1:-3], use_float=True, ascii=True)
#         pymesh.save_mesh(filterNaPlyFile, naPlyMesh, *naPlyMesh.get_attribute_names()[1:-3], use_float=True, ascii=True)
#
#         pVertIntWithNaVert = np.array([list(np.where((naPlyMesh.vertices == naPlyLabelVertices[item][:, None]).all(-1))[1]) for item in naNearProtIndex])
#         pVertIntWithNaVert = pVertIntWithNaVert[pVertIntWithNaVert.astype(bool)]
#         pVert2NaVertFile = os.path.join(outDir, "{}.{}_{}.ivert_mapping.npy".format(pid, pChain, naChain))
#         np.save(pVert2NaVertFile, pVertIntWithNaVert)


# def extractPniPDB(pdbFile, outDir, chainIds=None):
#     pniChainPairs = findProteinChainBoundNA(pdbFile)
#     pdbId = os.path.basename(pdbFile).split(".")[0]
#     outTuple = []
#     if pniChainPairs:
#         for pair in pniChainPairs:
#             if chainIds:
#                 if pair.pChain in chainIds:
#                     outTuple.append(("protein", pair.pChain))
#                 if pair.naChain in chainIds:
#                     outTuple.append((pair.naType, pair.naChain))
#             else:
#                 outTuple.append(("protein", pair.pChain))
#                 outTuple.append((pair.naType, pair.naChain))
#
#     for t in set(outTuple):
#         chainType, chain = t[0], t[1]
#         tmpOutDir = os.path.join(outDir, chainType)
#         resolveDir(tmpOutDir, chdir=False)
#         outFile = os.path.join(tmpOutDir, str(pdbId) + "_" + chain + ".pdb")
#         extractPDB(pdbFile, outFile, chainIds=[chain])
#
#     return pniChainPairs

def extractPDB(infilename, outfilename, chainIds=None):
    """
    extractPDB: Extract selected chains from a PDB and save the extracted chains to an output file.
    Pablo Gainza - LPDI STI EPFL 2019
    Released under an Apache License 2.0
    """
    # extract the chain_ids from infilename and save in outfilename.
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure(infilename, infilename)
    model = Selection.unfold_entities(struct, "M")[0]
    chains = Selection.unfold_entities(struct, "C")
    # Select residues to extract and build new structure
    structBuild = StructureBuilder.StructureBuilder()
    structBuild.init_structure("output")
    structBuild.init_seg(" ")
    structBuild.init_model(0)
    outputStruct = structBuild.get_structure()

    # Load a list of non-standard amino acid names -- these are
    # typically listed under HETATM, so they would be typically
    # ignored by the orginal algorithm
    modified_amino_acids = find_modified_amino_acids(infilename)

    for chain in model:
        if chainIds == None or chain.get_id() in chainIds:
            structBuild.init_chain(chain.get_id())
            for residue in chain:
                het = residue.get_id()
                if het[0] == " ":
                    outputStruct[0][chain.get_id()].add(residue)
                elif het[0][-3:] in modified_amino_acids:
                    outputStruct[0][chain.get_id()].add(residue)

    # Output the selected residues
    pdbio = PDBIO()
    pdbio.set_structure(outputStruct)
    pdbio.save(outfilename, select=NotDisordered(), preserve_atom_numbering=True)


"""
read_ply: Read a ply file from disk using pymesh and load the attributes used by MaSIF. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""
# def read_ply(filename):
#     # Read a ply file from disk using pymesh and load the attributes used by MaSIF.
#     # filename: the input ply file.
#     # returns data as tuple.
#     mesh = pymesh.load_mesh(filename)
#
#     attributes = mesh.get_attribute_names()
#     if "vertex_nx" in attributes:
#         nx = mesh.get_attribute("vertex_nx")
#         ny = mesh.get_attribute("vertex_ny")
#         nz = mesh.get_attribute("vertex_nz")
#
#         normals = np.column_stack((nx, ny, nz))
#     else:
#         normals = None
#     if "vertex_charge" in attributes:
#         charge = mesh.get_attribute("vertex_charge")
#     else:
#         charge = np.array([0.0] * len(mesh.vertices))
#
#     if "vertex_cb" in attributes:
#         vertex_cb = mesh.get_attribute("vertex_cb")
#     else:
#         vertex_cb = np.array([0.0] * len(mesh.vertices))
#
#     if "vertex_hbond" in attributes:
#         vertex_hbond = mesh.get_attribute("vertex_hbond")
#     else:
#         vertex_hbond = np.array([0.0] * len(mesh.vertices))
#
#     if "vertex_hphob" in attributes:
#         vertex_hphob = mesh.get_attribute("vertex_hphob")
#     else:
#         vertex_hphob = np.array([0.0] * len(mesh.vertices))
#
#     return (
#         mesh.vertices,
#         mesh.faces,
#         normals,
#         charge,
#         vertex_cb,
#         vertex_hbond,
#         vertex_hphob,
#     )


"""
save_ply: Save a ply file to disk using pymesh and load the attributes used by MaSIF. 
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""
def save_ply(filename, vertices, faces=[], normals=None, charges=None, vertex_cb=None, hbond=None, hphob=None,
             iface=None, normalize_charges=False,):
    """
        Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbond is not None:
        mesh.add_attribute("hbond")
        mesh.set_attribute("hbond", hbond)
    if vertex_cb is not None:
        mesh.add_attribute("vertex_cb")
        mesh.set_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)

    pymesh.save_mesh(filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True)


"""
protonate: Wrapper method for the reduce program: protonate (i.e., add hydrogens) a pdb using reduce 
                and save to an output file.
Pablo Gainza - LPDI STI EPFL 2019
Released under an Apache License 2.0
"""
def protonate(in_pdb_file, out_pdb_file):
    # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
    # in_pdb_file: file to protonate.
    # out_pdb_file: output file where to save the protonated pdb file.

    # Remove protons first, in case the structure is already protonated
    globalVars = GlobalVars()
    globalVars.initation()

    args = [globalVars.reduce_bin, "-Trim", in_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8').rstrip())
    outfile.close()
    # Now add them again.
    args = [globalVars.reduce_bin, "-HIS", out_pdb_file]
    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()
    outfile = open(out_pdb_file, "w")
    outfile.write(stdout.decode('utf-8'))
    outfile.close()

