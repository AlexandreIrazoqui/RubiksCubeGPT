import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from RubiksCubeGPT import charger_modele, encode, device

LAYER = 4  # bloc dont on lit la sortie (0..n_layer-1)

# chaque composante de l'etat : (nom, nb de slots, nb de classes par slot)
COMPOSANTES = [
    ("cp", 0, 8),  # permutation des coins
    (
        "co",
        1,
        3,
    ),  # orientation des coins (mod 3 car depend de l'orientation des faces du coin)
    ("ep", 2, 12),  # permutation des aretes
    (
        "eo",
        3,
        2,
    ),  # orientation des aretes (mod 2 car depend de l'orientatoin de l'arrete)
]


def collecter_activations(modele, exemples):
    """On passe les exemples dans le GPT (gelé) et on récupère ce qui se passe dedans.
    A (N, C) = les activations de la couche LAYER pour chaque token.
    Pour chaque composante, on récupère aussi la cible (N, nb_slots) qu'on veut prédire.
    """
    captured = {}
    hook = modele.blocks[LAYER].register_forward_hook(
        lambda mod, entree, sortie: captured.__setitem__("h", sortie)
    )

    A = []
    Y = {nom: [] for nom, _, _ in COMPOSANTES}
    with torch.no_grad():
        for ex in exemples:
            ids = torch.tensor([encode(ex["tokens"])], device=device)
            modele(ids)  # declenche le hook
            h = captured["h"][0]  # (T, C)
            for t, etat in enumerate(ex["states"]):
                A.append(h[t])
                for nom, idx, _ in COMPOSANTES:
                    Y[nom].append(etat[idx])

    hook.remove()
    A = torch.stack(A)  # (N, C)
    Y = {
        nom: torch.tensor(Y[nom], dtype=torch.long, device=device)
        for nom, _, _ in COMPOSANTES
    }
    return A, Y


def entrainer_sonde(A, Y, pas=300):
    """Sonde lineaire : une seule couche C -> nb_slots*nb_classes."""
    nb_slots = Y.shape[1]
    sonde = nn.Linear(A.shape[1], nb_slots * nb_classes).to(device)
    opt = torch.optim.AdamW(sonde.parameters(), lr=1e-2)
    for _ in range(pas):
        logits = sonde(A).view(-1, nb_slots, nb_classes)
        loss = F.cross_entropy(logits, Y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return sonde


def precision(sonde, A, Y, nb_classes):
    """Pourcentage de slots correctements predits"""
    with torch.no_grad():
        pred = sonde(A).view(-1, Y.shape[1], nb_classes).argmax(-1)
    return (pred == Y).float().mean().item()


def baseline(Y_tr, Y_val):
    """Classe majoritaire par slot, apprise sur train, evaluee sur val."""
    maj = Y_tr.cpu().mode(dim=0).values.to(Y_val.device)  # (nb_slots,)
    return (Y_val == maj).float().mean().item()


if __name__ == "__main__":
    modele = charger_modele()
    modele.eval()

    exemples = json.load(open("dataset.json"))

    n_val = max(1, len(exemples) // 5)
    ex_val, ex_tr = exemples[:n_val], exemples[n_val:]

    A_tr, Y_tr = collecter_activations(modele, ex_tr)
    A_val, Y_val = collecter_activations(modele, ex_val)
    print(f"Bloc {LAYER} | {len(A_tr)} positions train, {len(A_val)} val\n")

    print(f"{'composante':<12}{'sonde (val)':>14}{'baseline':>12}")
    for nom, _, nb_classes in COMPOSANTES:
        sonde = entrainer_sonde(A_tr, Y_tr[nom], nb_classes)
        acc = precision(sonde, A_val, Y_val[nom], nb_classes)
        base = baseline(Y_tr[nom], Y_val[nom])
        print(f"{nom:<12}{100 * acc:>12.1f} %{100 * base:>10.1f} %")
