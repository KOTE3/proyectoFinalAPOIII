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


def test_read_annotations_excel_variants(tmp_path):
    """El lector acepta el CSV original y el que guarda Excel en español."""
    from fruitq.data import read_annotations

    original = "archivo,autor,medida_cm\nbanano_buena_01.jpg,Sebastián Mejía,17.5\n"
    excel_utf8 = "\ufeffarchivo;autor;medida_cm\r\nbanano_buena_01.jpg;Sebastián Mejía;17,5\r\n"
    excel_ansi = "archivo;autor;medida_cm\r\nbanano_buena_01.jpg;Sebastián Mejía;17,5\r\n"
    for i, (text, enc) in enumerate([(original, "utf-8"), (excel_utf8, "utf-8"), (excel_ansi, "cp1252")]):
        p = tmp_path / f"a{i}.csv"
        p.write_bytes(text.encode(enc))
        df = read_annotations(p)
        assert list(df.columns) == ["archivo", "autor", "medida_cm", "conjunto"]
        assert df.loc[0, "conjunto"] == "principal"
        assert df.loc[0, "autor"] == "Sebastián Mejía"
        assert df.loc[0, "medida_cm"] == 17.5
