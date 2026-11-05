"""Pruebas de agrupación de duplicados y partición sin fuga de información."""

import numpy as np
import pandas as pd

from fruitq.data import burst_ids, check_no_leakage, connected_groups, group_split


def test_burst_ids_groups_consecutive_seconds():
    paths = pd.Series([
        "x/Apple_Bad/IMG_20190910_175712.jpg",
        "x/Apple_Bad/IMG_20190910_175713.jpg",   # +1 s -> misma ráfaga
        "x/Apple_Bad/IMG_20190910_175720.jpg",   # +7 s -> ráfaga nueva
        "x/Apple_Good/IMG_20190910_175713.jpg",  # otra carpeta -> ráfaga nueva
        "x/Apple_Bad/IMG_8134.jpg",              # sin hora -> -1
    ])
    b = burst_ids(paths)
    assert b[0] == b[1]
    assert b[2] != b[0]
    assert b[3] not in (b[0], b[2])
    assert b[4] == -1


def test_connected_groups_is_transitive():
    hash_key = np.array([10, 10, -1, 20, -1])
    burst = np.array([-1, 5, 5, -1, -1])
    g = connected_groups(hash_key, burst)
    assert g[0] == g[1] == g[2]      # 0~1 por hash, 1~2 por ráfaga
    assert len({g[0], g[3], g[4]}) == 3


def test_group_split_has_no_leakage():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame({
        "grupo": rng.integers(0, 150, n),
        "estrato": rng.choice(["a_buena", "a_mala", "b_buena"], n),
    })
    df["split"] = group_split(df, strat_col="estrato")
    check_no_leakage(df)
    frac = df.split.value_counts(normalize=True)
    assert 0.6 < frac["train"] < 0.8
