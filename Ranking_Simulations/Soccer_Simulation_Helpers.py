import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from NPI import calculate_npi
import numpy.linalg as LA
from sklearn.cluster import KMeans
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import random
import math
import numpy as np
from collections import defaultdict




@dataclass(frozen=True)
class Team:
    tid: int
    conference: int
    strength: float


Game = Tuple[int, int, bool]  # (team_i, team_j, is_conference_game)


def generate_teams(
    n_conferences: int,
    teams_per_conference: int,
    strength_sampler=None,
    seed: Optional[int] = None,
) -> List[Team]:
    rng = random.Random(seed)

    if strength_sampler is None:
        # default: N(0,1) independent of conference
        def strength_sampler(conf_id: int) -> float:
            return rng.gauss(0.0, 1.0)

    teams: List[Team] = []
    tid = 0
    for c in range(n_conferences):
        for _ in range(teams_per_conference):
            teams.append(Team(tid=tid, conference=c, strength=float(strength_sampler(c))))
            tid += 1
    return teams


def fixed_conference_baselines(
    baselines: List[float],
    within_sigma: float = 1.0,
    seed: Optional[int] = None,
):
    """
    baselines[c] is the mean for conference c.
    Team strength = baseline[c] + Normal(0, within_sigma).
    """
    rng = random.Random(seed)

    def sampler(conf_id: int) -> float:
        return baselines[conf_id] + rng.gauss(0.0, within_sigma)

    return sampler

def random_conference_baselines(
    n_conferences: int,
    between_sigma: float = 0.8,
    within_sigma: float = 1.0,
    seed: Optional[int] = None,
):
    """
    Conference baseline means are drawn as Normal(0, between_sigma).
    Team strength = baseline[conf] + Normal(0, within_sigma).
    Returns (sampler, baselines) so you can log the true conference means.
    """
    rng = random.Random(seed)
    baselines = [rng.gauss(0.0, between_sigma) for _ in range(n_conferences)]

    def sampler(conf_id: int) -> float:
        return baselines[conf_id] + rng.gauss(0.0, within_sigma)

    return sampler, baselines


def tiered_conference_baselines(
    n_conferences: int,
    n_strong: int,
    strong_mean: float = 1.0,
    weak_mean: float = -1.0,
    within_sigma: float = 1.0,
    shuffle: bool = True,
    seed: Optional[int] = None,
):
    """
    Make n_strong conferences have baseline strong_mean, others weak_mean.
    Optionally shuffle which conference ids are strong.
    """
    rng = random.Random(seed)
    labels = [1] * n_strong + [0] * (n_conferences - n_strong)
    if shuffle:
        rng.shuffle(labels)

    baselines = [strong_mean if labels[c] == 1 else weak_mean for c in range(n_conferences)]

    def sampler(conf_id: int) -> float:
        return baselines[conf_id] + rng.gauss(0.0, within_sigma)

    return sampler, baselines



def conference_round_robin(teams: List[Team]) -> List[Game]:
    games: List[Game] = []
    by_conf: Dict[int, List[int]] = {}
    for t in teams:
        by_conf.setdefault(t.conference, []).append(t.tid)

    for conf, ids in by_conf.items():
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                games.append((ids[i], ids[j], True))
    return games


def _build_conference_multiplicities(
    n_conferences: int,
    k: int,
    max_mult: Optional[int],
    seed: Optional[int],
    max_tries: int = 2000,
) -> List[List[int]]:
    """
    Build symmetric multiplicity matrix M with M[a][a]=0 and sum_b M[a][b] = k.
    This is a k-regular loopless multigraph on n_conferences vertices.
    """
    if n_conferences < 2:
        raise ValueError("Need at least 2 conferences for non-conference games.")
    if k < 0:
        raise ValueError("k must be >= 0.")
    if (n_conferences * k) % 2 != 0:
        raise ValueError(
            f"Infeasible: n_conferences*k must be even for a loopless k-regular multigraph. "
            f"Got n_conferences={n_conferences}, k={k}."
        )

    rng = random.Random(seed)

    stubs = []
    for c in range(n_conferences):
        stubs.extend([c] * k)

    for _ in range(max_tries):
        rng.shuffle(stubs)
        M = [[0] * n_conferences for _ in range(n_conferences)]
        ok = True

        i = 0
        while i < len(stubs):
            a = stubs[i]
            b = stubs[i + 1]

            # invalid if loop or too many multiedges between a and b
            if a == b or (max_mult is not None and M[a][b] >= max_mult):
                # try swapping stubs[i+1] with a later stub to fix
                swapped = False
                for j in range(i + 2, len(stubs)):
                    b2 = stubs[j]
                    if b2 != a and (max_mult is None or M[a][b2] < max_mult):
                        stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                        b = stubs[i + 1]
                        swapped = True
                        break
                if not swapped:
                    ok = False
                    break

            # add edge (a,b)
            M[a][b] += 1
            M[b][a] += 1
            i += 2

        if ok:
            # final sanity: row sums should all equal k
            if all(sum(M[r]) == k for r in range(n_conferences)):
                return M

    raise RuntimeError(
        "Failed to construct a valid conference multigraph. "
        "Try a different seed, reduce k, or allow higher max_mult."
    )


def _conference_pair_counts(
    n_conferences: int,
    stubs_per_conf: int,
    rng: random.Random,
    max_tries: int = 5000,
) -> List[List[int]]:
    """
    Build symmetric matrix M with:
      - M[a][a] = 0
      - sum_b M[a][b] = stubs_per_conf  for all a
    Interpretation: M[a][b] = number of cross-games between conference a and b (unordered).
    This is a loopless multigraph degree-sequence realization via a stub pairing at the conference level.
    """
    if n_conferences < 2:
        raise ValueError("Need at least 2 conferences.")
    if stubs_per_conf < 0:
        raise ValueError("stubs_per_conf must be >= 0.")
    total_stubs = n_conferences * stubs_per_conf
    if total_stubs % 2 != 0:
        raise ValueError(
            f"Infeasible: total conference stubs must be even. "
            f"Got n_conferences={n_conferences}, stubs_per_conf={stubs_per_conf}."
        )

    stubs = []
    for c in range(n_conferences):
        stubs.extend([c] * stubs_per_conf)

    for _ in range(max_tries):
        rng.shuffle(stubs)
        M = [[0] * n_conferences for _ in range(n_conferences)]
        ok = True

        i = 0
        while i < len(stubs):
            a = stubs[i]
            b = stubs[i + 1]
            if a == b:
                # try to fix by swapping stubs[i+1] with a later stub
                swapped = False
                for j in range(i + 2, len(stubs)):
                    if stubs[j] != a:
                        stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                        b = stubs[i + 1]
                        swapped = True
                        break
                if not swapped:
                    ok = False
                    break

            M[a][b] += 1
            M[b][a] += 1
            i += 2

        if ok and all(sum(M[r]) == stubs_per_conf for r in range(n_conferences)):
            return M

    raise RuntimeError("Failed to construct conference-pair counts; try a different seed.")


def _pair_lists_no_duplicate_edges(
    left: List[int],
    right: List[int],
    rng: random.Random,
    seen_edges: set,
    max_tries: int = 2000,
) -> List[Tuple[int, int]]:
    """
    Pair left[i] with a permuted right[j] to avoid:
      - duplicate team-vs-team edges (global seen_edges)
    Assumes left and right are from different conferences, so no self-edge is possible.
    """
    if len(left) != len(right):
        raise ValueError("Left/right lists must have same length.")
    m = len(left)
    if m == 0:
        return []

    for _ in range(max_tries):
        perm = list(range(m))
        rng.shuffle(perm)

        edges = []
        local_seen = set()
        ok = True

        for i in range(m):
            a = left[i]
            b = right[perm[i]]
            e = (a, b) if a < b else (b, a)
            if e in seen_edges or e in local_seen:
                ok = False
                break
            local_seen.add(e)
            edges.append(e)

        if ok:
            seen_edges |= local_seen
            return edges

    raise RuntimeError(
        "Failed to pair teams without duplicates. "
        "Try a different seed, reduce k, or allow repeats."
    )


def cross_conference_schedule_team_mixed(
    teams_by_conf: Dict[int, List[int]],
    k: int,
    seed: Optional[int] = None,
    balance_across_conferences: bool = True,
) -> List[Game]:
    """
    More-mixing cross-conference schedule:
      - Each team plays exactly k nonconf games.
      - Does NOT enforce 'conference-round' structure (i.e., conf A can play multiple other confs even when k=1).
      - Optionally balances aggregate counts between conference pairs.

    Input:
      teams_by_conf: dict conf_id -> list of team ids (equal sizes assumed)
      k: nonconf games per team

    Output:
      list of games (tid_i, tid_j, False) with no duplicates.
    """
    rng = random.Random(seed)
    confs = sorted(teams_by_conf.keys())
    n_conferences = len(confs)
    if n_conferences < 2:
        raise ValueError("Need at least 2 conferences.")
    if k < 0:
        raise ValueError("k must be >= 0.")

    sizes = {c: len(teams_by_conf[c]) for c in confs}
    if len(set(sizes.values())) != 1:
        raise ValueError("Conferences must have equal sizes for this helper.")
    N = sizes[confs[0]]

    # Hard feasibility if you want all opponents distinct:
    max_distinct = (n_conferences - 1) * N
    if k > max_distinct:
        raise ValueError(
            f"Infeasible without repeats: k={k} exceeds available distinct nonconf opponents {max_distinct}."
        )

    stubs_per_conf = N * k
    total_stubs = n_conferences * stubs_per_conf
    if total_stubs % 2 != 0:
        raise ValueError(
            f"Infeasible: total team stubs must be even. "
            f"(n_conferences * N * k) must be even; got {n_conferences}*{N}*{k}={total_stubs}."
        )

    # Step 1: decide how many games occur between each pair of conferences (aggregate),
    # so that each conference contributes exactly N*k team-stubs.
    if balance_across_conferences:
        M = _conference_pair_counts(n_conferences, stubs_per_conf, rng)
    else:
        # If not balancing, just randomly pair team stubs later (still cross-conf).
        M = None

    # Step 2: create per-conference team stubs (each team appears exactly k times)
    conf_team_stubs: Dict[int, List[int]] = {}
    for idx_c, c in enumerate(confs):
        stubs = []
        for tid in teams_by_conf[c]:
            stubs.extend([tid] * k)
        rng.shuffle(stubs)
        conf_team_stubs[c] = stubs

    games_set = set()
    games: List[Game] = []

    if balance_across_conferences:
        # Step 3: allocate stubs from conf a to each opponent conf b according to M
        # and then pair those lists for each (a,b).
        alloc: Dict[Tuple[int, int], List[int]] = {}
        for ai, a in enumerate(confs):
            # Build a randomized list of opponent-conference labels for this conference’s stubs
            labels = []
            for bi, b in enumerate(confs):
                if a == b:
                    continue
                labels.extend([b] * M[ai][bi])
            rng.shuffle(labels)

            stubs = conf_team_stubs[a]
            if len(labels) != len(stubs):
                raise RuntimeError("Internal mismatch in stub allocation.")

            for lab, tid in zip(labels, stubs):
                alloc.setdefault((a, lab), []).append(tid)

        # For each unordered conference pair (a<b), pair alloc(a->b) with alloc(b->a)
        for i in range(len(confs)):
            for j in range(i + 1, len(confs)):
                a, b = confs[i], confs[j]
                left = alloc.get((a, b), [])
                right = alloc.get((b, a), [])
                if len(left) != len(right):
                    raise RuntimeError(f"Internal imbalance between conferences {a} and {b}.")
                edges = _pair_lists_no_duplicate_edges(left, right, rng, games_set)
                games.extend([(u, v, False) for (u, v) in edges])

    else:
        # Unbalanced fallback: just create team stubs globally and pair stubs while forcing cross-conf.
        # (Still ensures each team has k games; mixing tends to be good but conf-pair counts may drift.)
        all_team_stubs = []
        for c in confs:
            all_team_stubs.extend(conf_team_stubs[c])
        rng.shuffle(all_team_stubs)

        conf_of = {}
        for c in confs:
            for tid in teams_by_conf[c]:
                conf_of[tid] = c

        # Greedy pairing with retries
        i = 0
        max_swaps = 200000
        swaps = 0
        while i < len(all_team_stubs):
            a = all_team_stubs[i]
            b = all_team_stubs[i + 1]
            if conf_of[a] == conf_of[b]:
                # swap with later stub to get cross-conf
                swapped = False
                for j in range(i + 2, len(all_team_stubs)):
                    if conf_of[all_team_stubs[j]] != conf_of[a]:
                        all_team_stubs[i + 1], all_team_stubs[j] = all_team_stubs[j], all_team_stubs[i + 1]
                        b = all_team_stubs[i + 1]
                        swapped = True
                        break
                if not swapped:
                    raise RuntimeError("Could not find cross-conf partner; try different seed.")
            e = (a, b) if a < b else (b, a)
            if e in games_set:
                # try swap b with later to avoid duplicate
                swapped = False
                for j in range(i + 2, len(all_team_stubs)):
                    bb = all_team_stubs[j]
                    if conf_of[bb] != conf_of[a]:
                        ee = (a, bb) if a < bb else (bb, a)
                        if ee not in games_set:
                            all_team_stubs[i + 1], all_team_stubs[j] = all_team_stubs[j], all_team_stubs[i + 1]
                            b = all_team_stubs[i + 1]
                            e = ee
                            swapped = True
                            break
                if not swapped:
                    # reshuffle tail and retry
                    tail = all_team_stubs[i + 1 :]
                    rng.shuffle(tail)
                    all_team_stubs[i + 1 :] = tail
                    swaps += 1
                    if swaps > max_swaps:
                        raise RuntimeError("Too many swaps; try different seed or reduce k.")
                    continue

            games_set.add(e)
            games.append((e[0], e[1], False))
            i += 2

    return games

def build_teams_by_conf(teams):
    by_conf = {}
    for t in teams:
        by_conf.setdefault(t.conference, []).append(t.tid)
    return by_conf

def teams_by_conference(teams):
    by_conf = {}
    for t in teams:
        by_conf.setdefault(t.conference, []).append(t.tid)
    # (Optional) sort for stability
    for c in by_conf:
        by_conf[c].sort()
    return by_conf

def build_schedule_with_conferences_mixed_nonconf(
    n_conferences: int,
    teams_per_conference: int,
    k_nonconf: int,
    seed: int = 0,
    strength_sampler=None,
    nonconf_seed: Optional[int] = None,
    balance_across_conferences: bool = True,
):
    """
    Same as build_schedule_with_conferences, but uses the team-mixed nonconference scheduler.
    """
    teams = generate_teams(
        n_conferences=n_conferences,
        teams_per_conference=teams_per_conference,
        strength_sampler=strength_sampler,
        seed=seed,
    )

    in_conf = conference_round_robin(teams)

    by_conf = teams_by_conference(teams)

    cross = cross_conference_schedule_team_mixed(
        teams_by_conf=by_conf,
        k=k_nonconf,
        seed=nonconf_seed if nonconf_seed is not None else seed,
        balance_across_conferences=balance_across_conferences,
    )

    games = in_conf + cross
    rng = random.Random(seed)
    rng.shuffle(games)
    return teams, games



def cross_conference_schedule(
    teams: List[Team],
    n_conferences: int,
    teams_per_conference: int,
    k: int,
    seed: Optional[int] = None,
) -> List[Game]:
    """
    Cross-conference schedule so each team plays exactly k nonconference games.

    Construction:
      - Build a loopless k-regular multigraph on conferences (multiplicity matrix M).
      - For each conference pair (a,b) with multiplicity m=M[a][b],
        create m matching rounds between the N teams in a and N teams in b.
      - For round r, we use cyclic shift pairing, which avoids repeats within that pair
        as long as m <= N. If m > N, repeats are unavoidable; we still schedule valid games.
    """
    rng = random.Random(seed)

    # group team ids by conference
    by_conf: Dict[int, List[int]] = {c: [] for c in range(n_conferences)}
    for t in teams:
        by_conf[t.conference].append(t.tid)

    for c in range(n_conferences):
        if len(by_conf[c]) != teams_per_conference:
            raise ValueError("Teams are not evenly divided as expected.")

    # Prefer to avoid repeat opponents within a conf-pair when possible
    # max_mult=N helps ensure m<=N if feasible; but for n_conferences=2, m=k is forced.
    if n_conferences == 2:
        max_mult = None
    else:
        max_mult = teams_per_conference

    try:
        M = _build_conference_multiplicities(
            n_conferences=n_conferences, k=k, max_mult=max_mult, seed=seed
        )
    except RuntimeError:
        # If strict max_mult fails (rare), relax and allow higher multiplicities.
        M = _build_conference_multiplicities(
            n_conferences=n_conferences, k=k, max_mult=None, seed=seed
        )

    games_set = set()

    for a in range(n_conferences):
        for b in range(a + 1, n_conferences):
            m = M[a][b]
            if m == 0:
                continue

            A = by_conf[a][:]
            B = by_conf[b][:]
            N = teams_per_conference

            # We'll reshuffle base order every N rounds to reduce repeats if m > N.
            rng.shuffle(A)
            rng.shuffle(B)

            for r in range(m):
                if r > 0 and r % N == 0:
                    rng.shuffle(A)
                    rng.shuffle(B)
                shift = r % N
                for i in range(N):
                    ta = A[i]
                    tb = B[(i + shift) % N]
                    pair = (min(ta, tb), max(ta, tb))
                    # Duplicates can only happen when m > N; keep schedule valid anyway.
                    games_set.add(pair)

    return [(i, j, False) for (i, j) in games_set]


def build_schedule_with_conferences(
    n_conferences: int,
    teams_per_conference: int,
    k_nonconf: int,
    strength_sampler=None,
    seed: Optional[int] = None,
) -> Tuple[List[Team], List[Game]]:
    teams = generate_teams(
        n_conferences=n_conferences,
        teams_per_conference=teams_per_conference,
        seed=seed,
        strength_sampler = strength_sampler
    )
    in_conf = conference_round_robin(teams)
    cross = cross_conference_schedule(
        teams=teams,
        n_conferences=n_conferences,
        teams_per_conference=teams_per_conference,
        k=k_nonconf,
        seed=seed,
    )
    games = in_conf + cross
    rng = random.Random(seed)
    rng.shuffle(games)
    return teams, games


def sanity_check_schedule(
    teams: List[Team],
    games: List[Game],
    k_nonconf: int,
) -> None:
    n = len(teams)
    conf = {t.tid: t.conference for t in teams}

    seen = set()
    nonconf_count = [0] * n
    inconf_pairs = set()

    for a, b, is_conf in games:
        if a == b:
            raise ValueError("Self-game detected.")
        pair = (min(a, b), max(a, b))
        if pair in seen:
            raise ValueError(f"Duplicate game detected: {pair}")
        seen.add(pair)

        same_conf = (conf[a] == conf[b])
        if is_conf != same_conf:
            raise ValueError(f"is_conference_game flag mismatch for game {(a, b, is_conf)}")

        if same_conf:
            inconf_pairs.add(pair)
        else:
            nonconf_count[a] += 1
            nonconf_count[b] += 1

    for tid, cnt in enumerate(nonconf_count):
        if cnt != k_nonconf:
            raise ValueError(f"Team {tid} has {cnt} nonconf games, expected {k_nonconf}")

    by_conf: Dict[int, List[int]] = {}
    for t in teams:
        by_conf.setdefault(t.conference, []).append(t.tid)

    for c, ids in by_conf.items():
        ids = sorted(ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pair = (ids[i], ids[j])
                if pair not in inconf_pairs:
                    raise ValueError(f"Missing in-conference game: conf {c}, pair {pair}")


def simulate_results(games, teams, stochastic):

    home_team = []
    away_team = []
    home_score = []
    away_score = []

    for game in games:
        home_team.append(game[0])
        away_team.append(game[1])

        if stochastic: 
            if random.random() < (math.exp(teams[game[0]].strength - teams[game[1]].strength))/(1+math.exp(teams[game[0]].strength - teams[game[1]].strength)):
                home_score.append(1)
                away_score.append(0)
            else:
                home_score.append(0)
                away_score.append(1)

        else:
            if teams[game[0]].strength > teams[game[1]].strength:
                home_score.append(1)
                away_score.append(0)
            else:
                home_score.append(0)
                away_score.append(1)


    data = {'home_team': home_team,
    'away_team': away_team,
    'home_score': home_score,
    'away_score': away_score}



    return pd.DataFrame(data)



def mean_strength_per_conference(teams):
    """
    Returns dict: conf_id -> mean strength
    """
    sums = defaultdict(float)
    counts = defaultdict(int)
    for t in teams:
        sums[t.conference] += float(t.strength)
        counts[t.conference] += 1
    return {c: (sums[c] / counts[c]) for c in sorted(counts)}

def strength_summary_per_conference(teams):
    """
    Returns dict: conf_id -> {'n':..., 'mean':..., 'std':...}
    """
    vals = defaultdict(list)
    for t in teams:
        vals[t.conference].append(float(t.strength))

    out = {}
    for c in sorted(vals):
        xs = vals[c]
        n = len(xs)
        mu = sum(xs) / n
        # population std (divide by n); switch to n-1 if you prefer sample std
        var = sum((x - mu) ** 2 for x in xs) / n if n else float("nan")
        out[c] = {"n": n, "mean": mu, "std": math.sqrt(var)}
    return out

def print_strength_summary_per_conference(teams, digits=3):
    summ = strength_summary_per_conference(teams)
    for c, s in summ.items():
        print(f"Conf {c}: n={s['n']:>3d}  mean={s['mean']:.{digits}f}  std={s['std']:.{digits}f}")




def _ranks(x: np.ndarray) -> np.ndarray:
    # average ranks for ties
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # tie handling (average ranks)
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j+1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (i + j) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks

def spearman_corr(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ra = _ranks(a)
    rb = _ranks(b)
    # Pearson on ranks
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")

def topk_capture(strength, npi, k: int) -> float:
    strength = np.asarray(strength)
    npi = np.asarray(npi)
    true_top = set(np.argsort(-strength)[:k])
    npi_top = set(np.argsort(-npi)[:k])
    return len(true_top & npi_top) / k

def inversion_rate(strength, npi) -> float:
    s = np.asarray(strength)
    r = np.asarray(npi)
    n = len(s)
    inv = 0
    tot = 0
    for i in range(n):
        for j in range(i+1, n):
            tot += 1
            ds = s[i] - s[j]
            dr = r[i] - r[j]
            # count as inversion if they disagree in sign (ignore exact ties)
            if ds == 0 or dr == 0:
                continue
            if ds * dr < 0:
                inv += 1
    return inv / tot if tot else float("nan")




def within_conference_spearman(strength, npi, conf):
    strength = np.asarray(strength)
    npi = np.asarray(npi)
    conf = np.asarray(conf)
    vals = []
    for c in np.unique(conf):
        idx = np.where(conf == c)[0]
        if len(idx) >= 5:
            vals.append(spearman_corr(strength[idx], npi[idx]))
    return float(np.mean(vals)) if vals else float("nan")


def conference_residual_offsets(strength, npi, conf):
    """
    Fit npi ≈ a + b*strength, compute residuals, then average residual by conference.
    Returns dict conf -> mean_resid, plus overall residual std.
    """
    s = np.asarray(strength, dtype=float)
    r = np.asarray(npi, dtype=float)
    conf = np.asarray(conf)

    X = np.vstack([np.ones_like(s), s]).T
    # least squares
    beta, *_ = np.linalg.lstsq(X, r, rcond=None)
    r_hat = X @ beta
    resid = r - r_hat

    offsets = {}
    for c in np.unique(conf):
        idx = conf == c
        offsets[int(c)] = float(resid[idx].mean())
    return offsets, float(resid.std())

def bubble_offset_summary(offsets: dict) -> dict:
    vals = np.array(list(offsets.values()), dtype=float)
    return {
        "rms_conf_offset": float(np.sqrt(np.mean(vals**2))),
        "range_conf_offset": float(vals.max() - vals.min()) if len(vals) else float("nan"),
        "max_abs_conf_offset": float(np.max(np.abs(vals))) if len(vals) else float("nan"),
    }

def cross_conference_inversion_rate(strength, npi, conf) -> float:
    s = np.asarray(strength)
    r = np.asarray(npi)
    conf = np.asarray(conf)
    n = len(s)
    inv = 0
    tot = 0
    for i in range(n):
        for j in range(i+1, n):
            if conf[i] == conf[j]:
                continue
            tot += 1
            ds = s[i] - s[j]
            dr = r[i] - r[j]
            if ds == 0 or dr == 0:
                continue
            if ds * dr < 0:
                inv += 1
    return inv / tot if tot else float("nan")

def npi_calibration_report(strength, npi, wins, losses, conf, topk_list=(5,10,20)):
    strength = np.asarray(strength, dtype=float)
    npi = np.asarray(npi, dtype=float)
    wins = np.asarray(wins, dtype=float)
    losses = np.asarray(losses, dtype=float)
    conf = np.asarray(conf)

    games = wins + losses
    winpct = np.divide(wins, games, out=np.zeros_like(wins), where=games>0)

    report = {}
    report["spearman_strength_npi"] = spearman_corr(strength, npi)
    report["spearman_strength_npi_within_conf_avg"] = within_conference_spearman(strength, npi, conf)
    report["inversion_rate_global"] = inversion_rate(strength, npi)
    report["inversion_rate_cross_conf"] = cross_conference_inversion_rate(strength, npi, conf)

    for k in topk_list:
        if k <= len(strength):
            report[f"top{k}_capture"] = topk_capture(strength, npi, k)

    offsets, resid_std = conference_residual_offsets(strength, npi, conf)
    report["conference_offsets"] = offsets
    report["resid_std_overall"] = resid_std
    report.update(bubble_offset_summary(offsets))

    # Simple sanity: NPI vs win%
    report["spearman_winpct_npi"] = spearman_corr(winpct, npi)
    report["spearman_strength_winpct"] = spearman_corr(strength, winpct)

    return report
