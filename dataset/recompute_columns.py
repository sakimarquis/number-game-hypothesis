from math import log
from pathlib import Path

import pandas as pd


def recompute_columns(df):
    """Compute hits, p, H, typicality for each target, for each set.

    hits: number of data points for the target
    p: mean rating
    H: entropy as a function of p
    typicality: how typical is this data point, relative to the target's mean rating?
                Calculated as log(p) if rating is 1, log(1-p) if rating is 0.

    Note
    ----
    This may take a long time to process (12+ hours).
    """
    groups = []

    for _, g in df.groupby(['set', 'target'], as_index=False):
        g = g.copy()
        g['hits'] = len(g['rating'])
        p = g['rating'].mean()
        if p == 1.0:
            p = .99999
        if p == 0.0:
            p = .00001
        g['p'] = p
        g['H'] = -((p*log(p)) + ((1-p)*log(1-p)))

        g.loc[g.rating == 1, 'typicality'] = log(p)
        g.loc[g.rating == 0, 'typicality'] = log(1-p)
        groups.append(g)

    return pd.concat(groups, ignore_index=True)


if __name__ == "__main__":
    dataset_dir = Path(__file__).parent
    output_path = dataset_dir / "numbergame_data_recomputed.csv"
    recompute_columns(pd.read_csv(dataset_dir / "numbergame_data.csv")).to_csv(output_path, index=False)
    print(f"Wrote recomputed columns to {output_path}")
