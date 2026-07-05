"""Simulateur de Rubik's cube (representation Kociemba) partage entre la generation
du dataset et l'evaluation solve-rate du modele.

Un cube = 4 tableaux : cp (corner permutation), co (corner orientation, mod 3),
ep/eo (edge permutation / orientation, mod 2). Un coup permute ces 4 tableaux.
"""

import random


def cube_resolu():
    """Retourne un cube neuf, parfaitement resolu."""
    return {
        "cp": [0, 1, 2, 3, 4, 5, 6, 7],
        "co": [0, 0, 0, 0, 0, 0, 0, 0],
        "ep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "eo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }


MOVES = {
    "U": {
        "cp": [
            3,
            0,
            1,
            2,
            4,
            5,
            6,
            7,
        ],  # tourner la face haute d'un quart de tour permute les 4 corners du haut mais pas leur orientation
        "co": [0, 0, 0, 0, 0, 0, 0, 0],
        "ep": [3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],  # Idem pour les 4 premiers edges
        "eo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "R": {
        "cp": [4, 1, 2, 0, 7, 5, 6, 3],
        "co": [2, 0, 0, 1, 1, 0, 0, 2],
        "ep": [8, 1, 2, 3, 11, 5, 6, 7, 4, 9, 10, 0],
        "eo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "F": {
        "cp": [1, 5, 2, 3, 0, 4, 6, 7],
        "co": [1, 2, 0, 0, 2, 1, 0, 0],
        "ep": [0, 9, 2, 3, 4, 8, 6, 7, 1, 5, 10, 11],
        "eo": [0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
    },
    "D": {
        "cp": [0, 1, 2, 3, 5, 6, 7, 4],
        "co": [0, 0, 0, 0, 0, 0, 0, 0],
        "ep": [0, 1, 2, 3, 5, 6, 7, 4, 8, 9, 10, 11],
        "eo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "L": {
        "cp": [0, 2, 6, 3, 4, 1, 5, 7],
        "co": [0, 1, 2, 0, 0, 2, 1, 0],
        "ep": [0, 1, 10, 3, 4, 5, 9, 7, 8, 2, 6, 11],
        "eo": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    "B": {
        "cp": [0, 1, 3, 7, 4, 5, 2, 6],
        "co": [0, 0, 1, 2, 0, 0, 2, 1],
        "ep": [0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7],
        "eo": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    },
}


def turn(cube, face):
    """Applique un quart de tour horaire a la face donneee"""
    m = MOVES[face]
    nouveau = cube_resolu()  # pour avoir des listes de la bonne dimension

    for i in range(8):  # les 8 coins
        origine = m["cp"][i]
        nouveau["cp"][i] = cube["cp"][origine]
        nouveau["co"][i] = (cube["co"][origine] + m["co"][i]) % 3

    for i in range(12):
        origine = m["ep"][i]
        nouveau["ep"][i] = cube["ep"][origine]
        nouveau["eo"][i] = (cube["eo"][origine] + m["eo"][i]) % 2

    return nouveau


def appliquer(cube, coup):
    """Applique le coup en notation standard "R" quart de tour "R'" quart de tour dans l'autre sens "R2" 2 tours"""
    face = coup[0]
    if len(coup) == 1:
        nb_tours = 1
    elif coup[1] == "2":
        nb_tours = 2
    else:
        nb_tours = 3
    for _ in range(nb_tours):
        cube = turn(cube, face)
    return cube


def jouer_sequence(cube, liste_coup):
    for coup in liste_coup:
        cube = appliquer(cube, coup)
    return cube


def est_resolu(cube):
    """True si le cube est dans l'etat resolu."""
    return cube == cube_resolu()


# kociemba attend une chaine de 54 lettres (une par sticker) dans l'ordre URFDLB, les tables ci dessous permettent la conversion

COIN_STICKERS = [
    [8, 9, 20],
    [6, 18, 38],
    [0, 36, 47],
    [2, 45, 11],
    [29, 26, 15],
    [27, 44, 24],
    [33, 53, 42],
    [35, 17, 51],
]
COIN_COULEURS = [
    ["U", "R", "F"],
    ["U", "F", "L"],
    ["U", "L", "B"],
    ["U", "B", "R"],
    ["D", "F", "R"],
    ["D", "L", "F"],
    ["D", "B", "L"],
    ["D", "R", "B"],
]

ARETE_STICKERS = [
    [5, 10],
    [7, 19],
    [3, 37],
    [1, 46],
    [32, 16],
    [28, 25],
    [30, 43],
    [34, 52],
    [23, 12],
    [21, 41],
    [50, 39],
    [48, 14],
]
ARETE_COULEURS = [
    ["U", "R"],
    ["U", "F"],
    ["U", "L"],
    ["U", "B"],
    ["D", "R"],
    ["D", "F"],
    ["D", "L"],
    ["D", "B"],
    ["F", "R"],
    ["F", "L"],
    ["B", "L"],
    ["B", "R"],
]


def en_facelets(cube):
    stickers = [""] * 54
    # centres
    for lettre, case in zip("URFDLB", (4, 13, 22, 31, 40, 49)):
        stickers[case] = lettre
    # coins
    for i in range(8):
        for n in range(3):
            case = COIN_STICKERS[i][(n + cube["co"][i]) % 3]
            stickers[case] = COIN_COULEURS[cube["cp"][i]][n]

    # aretes
    for i in range(12):
        for n in range(2):
            case = ARETE_STICKERS[i][(n + cube["eo"][i]) % 2]
            stickers[case] = ARETE_COULEURS[cube["ep"][i]][n]

    return "".join(stickers)


CUBE_RESOLU_FACELETS = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def melanger(nb_coups):
    coups = []
    etats = []
    cube = cube_resolu()
    derniere_face = ""

    for _ in range(nb_coups):
        face_possibles = [f for f in "URFDLB" if f != derniere_face]
        face = random.choice(face_possibles)
        derniere_face = face

        coup = face + random.choice(["", "'", "2"])
        cube = appliquer(cube, coup)

        coups.append(coup)
        etats.append(cube)

    return coups, etats, cube


def compacter(cube):
    return [cube["cp"], cube["co"], cube["ep"], cube["eo"]]
