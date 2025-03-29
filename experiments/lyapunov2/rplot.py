import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from pathlib import Path


def plot_r(dataframe: pd.DataFrame, output_path):
    dataframe.to_csv(output_path / "df.csv")
    patchwork = importr("patchwork")
    ggplot2 = importr("ggplot2")
    Hmisc = importr("Hmisc")
    grid = importr("grid")
    dplyr = importr("dplyr")

    path = Path(output_path)
    path.mkdir(parents=True, exist_ok=True)

    with (ro.default_converter + pandas2ri.converter).context():
        r_df = ro.conversion.get_conversion().py2rpy(dataframe)

    ro.globalenv["uncertainty"] = r_df

    ro.r("uncertainty$label <- as.character(uncertainty$pred)")

    ro.r("lambda_order <- uncertainty[order(uncertainty$lambda, decreasing=TRUE),]")
    ro.r(
        """ftle_xy <- (ggplot2::ggplot(lambda_order, aes(x=x, y=y, label=label))
     + geom_text(size=3, alpha=0.5, aes(color=lambda)) + scale_color_gradientn(colors=rainbow(3)))"""
    )
    path = Path(output_path) / "ftle_xy.pdf"
    ro.r(f'ggsave("{path.as_posix()}", ftle_xy)')

    path = Path(output_path) / "UQ_xyL.pdf"
    ro.r(
        f"""(ggplot2::ggplot(uncertainty, aes(x=x, y=y, label=label)) 
      + geom_text(size=3, alpha=0.1, aes(color=factor(label)))
      + geom_density2d(aes(color=factor(label)))
    )
    ggsave("{path.as_posix()}")"""
    )

    path = Path(output_path) / "lambda_MI.pdf"
    ro.r(
        f"""(ftle_mi_lambda <- ggplot2::ggplot(uncertainty, aes(x=lambda, y=MI))
      + geom_point(alpha=0.1, size=0.2, color="blue") 
      + stat_summary_bin(fun.data="mean_sdl", bins=60))
    ggsave("{path.as_posix()}", ftle_mi_lambda)"""
    )

    path = Path(output_path) / "lambda_x.pdf"
    ro.r(
        f"""(ftle_x <- ggplot2::ggplot(uncertainty, aes(x=x, y=lambda))
      + geom_point(alpha=0.1, size=0.2, aes(color=factor(label))) 
      + stat_summary_bin(fun.data="mean_sdl", bins=60)
      + scale_y_continuous(trans="log2"))
    ggsave("{path.as_posix()}", ftle_x)"""
    )
