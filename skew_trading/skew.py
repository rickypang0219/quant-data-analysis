import polars as pl
import matplotlib.pyplot as plt
from scipy.stats import shapiro

UNIVERSE = ["BTCUSDT", "ETCUSDT", "BCHUSDT"]


def prepare_universe_names() -> list[str]:
    return [f"{i}#close" for i in UNIVERSE]


def prepare_columns() -> list[str]:
    col = [f"{i}#close" for i in UNIVERSE]
    col.append("timestamp")
    return col


def compute_log_change(df: pl.DataFrame) -> pl.DataFrame:
    df_clone = df.clone()
    df_clone = df_clone.with_columns(
        [
            (pl.col(i).log().diff(1)).alias(f"r_{i.replace('#close','')}")
            for i in prepare_universe_names()
        ]
    )
    return df_clone


def compute_skew(df: pl.DataFrame, window: int) -> pl.DataFrame:
    names = [f"r_{i}" for i in UNIVERSE]
    for i in names:
        ret = df[i]
        mv_avg = ret.fill_null(0).rolling_mean(window)
        mv_std = ret.fill_null(0).rolling_std(window)
        mv_std = mv_std.fill_null(1e-10)
        skew = (
            ((ret - mv_avg) ** 3 / mv_std**3)
            .fill_null(0)
            .rolling_mean(window)
            .fill_null(0)
        )
        df = df.with_columns(skew.alias(f"{i.replace('r_','')}_skew"))
    return df


def plot_skews(df: pl.DataFrame) -> None:
    plt.figure()
    for i in [f"{_}_skew" for _ in UNIVERSE]:
        plt.plot(
            df.select("timestamp").to_numpy(), df.select(i).to_numpy(), label=f"{i}"
        )
    plt.legend()
    plt.show()


def plot_skews_histo(df: pl.DataFrame) -> None:
    plt.figure()
    for i in [f"{_}_skew" for _ in UNIVERSE]:
        data = df.select(i).fill_null(0).to_numpy()
        stat, p_value = shapiro(data)
        print(f"Symbol:{i}, stat:{stat}, p-value:{p_value}")
        plt.hist(data, bins=500, alpha=0.5, label=f"{i}")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("price.csv")
    selected_df = df.select(prepare_columns())
    print(compute_log_change(selected_df))
    skew_df = compute_skew(compute_log_change(selected_df), 200)
    print(skew_df)
    plot_skews(skew_df)
    plot_skews_histo(skew_df)
