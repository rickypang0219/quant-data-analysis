import polars as pl
import matplotlib.pyplot as plt

UNIVERSE = ["BTCUSDT", "ETCUSDT", "BCHUSDT"]


def prepare_universe_names() -> list[str]:
    return [f"{i}#close" for i in UNIVERSE]


def prepare_columns() -> list[str]:
    col = [f"{i}#close" for i in UNIVERSE]
    col.append("timestamp")
    return col


def compute_log_change(df: pl.DataFrame) -> pl.DataFrame:
    df_returns = df.select(
        [
            (pl.col(i).log().diff(1)).alias(f"r_{i.replace('#close','')}")
            for i in prepare_universe_names()
        ]
    )
    return df_returns.with_columns(df["timestamp"].alias("timestamp"))


def compute_skew(df: pl.DataFrame, window: int) -> pl.DataFrame:
    names = [f"r_{i}" for i in UNIVERSE]
    for i in names:
        ret = df[i]
        df = df.with_columns(
            ret.rolling_skew(window).alias(f"{i.replace('r_','')}_mv_skew")
        )
    return df.drop(names)


def plot_skews(df: pl.DataFrame) -> None:
    plt.figure()
    for i in UNIVERSE:
        plt.plot(
            df.select("timestamp").to_numpy(),
            df.select(f"{i}_mv_skew").to_numpy(),
            label=f"{i}",
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    df = pl.read_csv("price.csv")
    selected_df = df.select(prepare_columns())
    print(compute_log_change(selected_df))
    skew_df = compute_skew(compute_log_change(selected_df), 1000)
    # skew_stats(skew_df)
    plot_skews(skew_df)
    # plot_skew_diff(skew_df)
