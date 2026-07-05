import kociemba
import random
import json

from cube import (
    cube_resolu,
    appliquer,
    jouer_sequence,
    en_facelets,
    CUBE_RESOLU_FACELETS,
    melanger,
    compacter,
)

SEED = 67
random.seed(SEED)
dataset = []


def construire_sequence(coups_melange, etats_melange, cube_melange, solution):
    tokens = []
    etats = []

    for coup, etat in zip(coups_melange, etats_melange):
        tokens.append(coup)
        etats.append(etat)

    tokens.append("<SEP>")
    etats.append(cube_melange)

    cube = cube_melange
    for coup in solution:
        cube = appliquer(cube, coup)
        tokens.append(coup)
        etats.append(cube)

    tokens.append("<E>")
    etats.append(cube_resolu())

    return tokens, etats


# Creation du dataset par tranches de nombre de coups pour avoir des difficultés variées

TRANCHES = [(1, 3), (4, 7), (8, 12), (13, 16), (17, 20)]
PAR_TRANCHE = 20  # 5 tranches * 20 = 100 exemples au total
VERIFIER = True  # verifie que le simulateur est correct (a laisser True)

dataset = []

for mini, maxi in TRANCHES:
    fait = 0
    while fait < PAR_TRANCHE:
        nb = random.randint(mini, maxi)
        coups, etats, cube = melanger(nb)

        facelets = en_facelets(cube)
        if facelets == CUBE_RESOLU_FACELETS:
            continue  # le melange s'est annule tout seul : on le reture

        solution = kociemba.solve(facelets).split()
        if not solution:
            continue

        if VERIFIER:
            # (a) rejouer les coups de melange doit redonner le meme etat
            assert jouer_sequence(cube_resolu(), coups) == cube
            # (b) appliquer la solution doit resoudre le cube
            assert jouer_sequence(cube, solution) == cube_resolu()

        tokens, etats_alignes = construire_sequence(coups, etats, cube, solution)
        assert len(tokens) == len(etats_alignes)

        dataset.append(
            {
                "scramble": coups,
                "solution": solution,
                "tokens": tokens,
                "states": [compacter(e) for e in etats_alignes],
                "difficulty": len(solution),
            }
        )
        fait += 1


def ecrire_txt(chemin, exemples):
    with open(chemin, "w") as f:
        for d in exemples:
            ligne = (
                " ".join(d["scramble"]) + " <SEP> " + " ".join(d["solution"]) + " <E>"
            )
            f.write(ligne + "\n")


random.shuffle(dataset)
n_val = max(1, len(dataset) // 10)
val, train = dataset[:n_val], dataset[n_val:]
ecrire_txt("train.txt", train)
ecrire_txt("val.txt", val)

# dataset.json : tout (coups + etats) pour l'analyse et la sonde
with open("dataset.json", "w") as f:
    json.dump(dataset, f)

from collections import Counter

distribution = Counter(d["difficulty"] for d in dataset)

print(
    "Dataset genere :",
    len(dataset),
    "exemples",
    f"(train {len(train)} / val {len(val)})",
)
print("Verification :", "OK" if VERIFIER else "desactivee")
print("Distribution par difficulte :", dict(sorted(distribution.items())))
