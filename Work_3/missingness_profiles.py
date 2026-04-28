"""
Paper note:
Suppose there are S sources. For participant j, the paper defines a binary
availability indicator I_j = [I_{j1}, ..., I_{jS}], where I_{ji} = 1 if
source i is observed and I_{ji} = 0 if source i is missing. For example,
when S = 3, the pattern [1, 1, 0] means source 1 and source 2 are observed
and source 3 is missing.

The paper then converts this binary vector into one profile integer:
    p_j = sum_{i=1}^S I_{ji} * 2^(S-i).
So for S = 3:
    [1, 1, 0] -> 1*2^2 + 1*2^1 + 0*2^0 = 6,
    [1, 0, 1] -> 5,
    [0, 1, 0] -> 2.
This is why the paper can later test whether source i is present using a
bitwise expression such as p_j & 2^(S-i) != 0.

Section 3 uses two related but different objects. The first is the exact
participant profile p_j itself. The second is a source combination m used to
form the paper's matrices X_m and labels y_m. For a given combination m, the
paper builds the optimization group
    G_m = { j : every source required by m is present for participant j }.
In integer form this is
    G_m = { j : (p_j & m) = m }.
So if m = 110 (sources 1 and 2), then every participant with exact profile
110 or 111 belongs to G_m.

That is the reason the paper says the groups can overlap: one participant has
only one exact profile p_j, but the same participant can belong to several
paper-level groups G_m. For example, with exact profile 111, the participant
belongs to groups 111, 110, 101, 011, 100, 010, and 001. Then X_m is built
by restricting the rows to j in G_m and keeping only the sources named by m.

Code mapping:
`X_blocks[source]` is one source matrix. A source is treated as missing for
sample j only when the whole row is NaN. `source_order` fixes the left-to-right
bit meaning. For example, if source_order = ["source1", "source2", "source3"],
then ("source1", "source3") corresponds to the paper-style code 101.
`build_exact_profiles` returns the one exact profile per sample.
`build_missingness_profiles` returns the overlapping paper-style groups G_m.
"""

from itertools import combinations

import numpy as np


def _normalize_source_order(X_blocks, source_order=None):
    """
    Validate source ordering and block shapes.

    source_order fixes which source owns each bit position in the paper-style
    binary profile code. If

        source_order = ["source1", "source2", "source3"],

    then

        110 means source1 + source2 observed, source3 missing
        101 means source1 + source3 observed, source2 missing

    If source_order is None, we keep the insertion order from X_blocks.
    """
    if not X_blocks:
        raise ValueError("X_blocks must contain at least one data source.")

    if source_order is None:
        source_order = list(X_blocks.keys())
    else:
        source_order = list(source_order)

    missing_sources = [source for source in source_order if source not in X_blocks]
    if missing_sources:
        raise ValueError(
            f"source_order contains unknown sources: {missing_sources}"
        )

    if set(source_order) != set(X_blocks.keys()):
        extra_sources = [source for source in X_blocks if source not in source_order]
        raise ValueError(
            "source_order must contain every source exactly once. "
            f"Missing or extra sources detected: {extra_sources}"
        )

    n_samples = None
    normalized_blocks = {}

    for source in source_order:
        X = np.asarray(X_blocks[source], dtype=float)

        if X.ndim != 2:
            raise ValueError(
                f"Source {source} must be a 2D matrix, got shape {X.shape}."
            )

        if n_samples is None:
            n_samples = X.shape[0]
        elif X.shape[0] != n_samples:
            raise ValueError(
                "All sources must have the same number of samples. "
                f"Expected {n_samples}, got {X.shape[0]} for source {source}."
            )

        normalized_blocks[source] = X

    return normalized_blocks, source_order, n_samples


def _observed_mask(X):
    """
    A source block is considered observed for a sample if the row is not all NaN.
    """
    return ~np.all(np.isnan(X), axis=1)


def get_profile_code(profile, source_order):
    """
    Convert a tuple of source names into a binary availability code.
    """
    profile_sources = set(profile)
    return "".join("1" if source in profile_sources else "0" for source in source_order)


def get_profile_codes(profiles, source_order):
    """
    Return a dict mapping tuple-based profile keys to binary availability codes.
    """
    return {
        profile: get_profile_code(profile, source_order)
        for profile in profiles
    }


def build_exact_profiles(X_blocks, source_order=None, allow_all_missing=False):
    """
    Build disjoint participant-level missingness profiles.

    Each sample j belongs to exactly one exact profile p_j, determined by the
    set of sources whose rows are not entirely NaN.

    In paper language, this corresponds to the participant's binary indicator
    vector I_j, or equivalently its single encoded profile integer p_j.
    In code we keep the profile as a tuple of source names because it is easier
    to read than the decimal encoding.
    """
    X_blocks, source_order, n_samples = _normalize_source_order(
        X_blocks,
        source_order=source_order,
    )

    observed_masks = {
        source: _observed_mask(X_blocks[source])
        for source in source_order
    }

    profiles = {}

    for sample_idx in range(n_samples):
        available_sources = tuple(
            source
            for source in source_order
            if observed_masks[source][sample_idx]
        )

        if not available_sources:
            if allow_all_missing:
                continue
            raise ValueError(
                "Every sample must have at least one observed source. "
                f"Sample index {sample_idx} has all sources missing."
            )

        profiles.setdefault(available_sources, []).append(sample_idx)

    return {
        profile: np.asarray(indices, dtype=int)
        for profile, indices in profiles.items()
    }


def build_missingness_profiles(X_blocks, source_order=None, allow_all_missing=False):
    """
    Build the overlapping iSFS optimization groups from section 3 of the paper.

    For a source combination m, the paper forms the group

        G_m = { j : (p_j & m) = m }.

    In words: every 1-bit required by m must also be 1 in participant j's
    exact profile p_j.

    In the tuple-based code used here, this means the group for combination m
    contains every sample whose exact observed-source tuple is a superset of m.
    Therefore these groups are overlapping by design.
    """
    exact_profiles = build_exact_profiles(
        X_blocks,
        source_order=source_order,
        allow_all_missing=allow_all_missing,
    )

    if source_order is None:
        source_order = list(X_blocks.keys())
    else:
        source_order = list(source_order)

    groups = {}

    for exact_profile, indices in exact_profiles.items():
        for group_size in range(1, len(exact_profile) + 1):
            for group_profile in combinations(exact_profile, group_size):
                groups.setdefault(group_profile, []).extend(indices.tolist())

    return {
        profile: np.asarray(indices, dtype=int)
        for profile, indices in groups.items()
    }


def print_profiles(profiles, source_order=None, title="Missingness profiles"):
    """
    Print profile counts with both tuple keys and binary availability codes.
    """
    if source_order is None:
        source_order = []
        for profile in profiles:
            for source in profile:
                if source not in source_order:
                    source_order.append(source)

    ordered_profiles = sorted(
        profiles,
        key=lambda profile: (len(profile), get_profile_code(profile, source_order), profile),
    )

    print(f"\n{title}:")
    for profile in ordered_profiles:
        code = get_profile_code(profile, source_order)
        print(f"{code} {profile}: {len(profiles[profile])} samples")


def print_profile_diagnostics(X_blocks, source_order=None, allow_all_missing=False):
    """
    Print both exact disjoint profiles and overlapping iSFS optimization groups.
    """
    exact_profiles = build_exact_profiles(
        X_blocks,
        source_order=source_order,
        allow_all_missing=allow_all_missing,
    )
    optimization_groups = build_missingness_profiles(
        X_blocks,
        source_order=source_order,
        allow_all_missing=allow_all_missing,
    )

    if source_order is None:
        source_order = list(X_blocks.keys())
    else:
        source_order = list(source_order)

    print_profiles(
        exact_profiles,
        source_order=source_order,
        title="Exact Missingness Profiles",
    )
    print_profiles(
        optimization_groups,
        source_order=source_order,
        title="iSFS Optimization Groups",
    )
