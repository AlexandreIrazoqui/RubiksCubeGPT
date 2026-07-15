from collections import defaultdict

from rubikscubegpt.cube import cube_resolu, jouer_sequence, est_resolu
from rubikscubegpt.model import charger_modele, load_pairs

MOVES_VALIDES = set(f + s for f in "URFDLB" for s in ("", "'", "2"))


def evaluer(modele, pairs):
    """Retourne {profondeur: (resolus, total)}."""
    stats = defaultdict(lambda: [0, 0])
    for scramble, _solution_kociemba in pairs:
        profondeur = len(scramble)
        stats[profondeur][1] += 1

        prediction = modele.solve(scramble)  # coups predits, sans <SEP> ni <E>
        if all(coup in MOVES_VALIDES for coup in prediction):
            cube_melange = jouer_sequence(cube_resolu(), scramble)
            cube_final = jouer_sequence(cube_melange, prediction)
            if est_resolu(cube_final):
                stats[profondeur][0] += 1
    return stats


if __name__ == "__main__":
    modele = charger_modele()
    pairs = load_pairs("val.txt")
    stats = evaluer(modele, pairs)

    print("Solve-rate par profondeur de scramble (jeu de validation) :")
    total_ok = total = 0
    for prof in sorted(stats):
        ok, n = stats[prof]
        total_ok += ok
        total += n
        print(f"  profondeur {prof:2d} : {ok:3d}/{n:<3d}  ({100 * ok / n:5.1f} %)")
    print(
        f"  --------\n  global      : {total_ok}/{total}  ({100 * total_ok / total:.1f} %)"
    )
