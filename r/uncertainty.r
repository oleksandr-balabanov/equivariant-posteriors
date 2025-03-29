library(patchwork)
library(ggplot2)
uncertainty <-
  read.csv("~/projects/equivariant-posteriors/uncertainty_mlp.csv")
xy_MI <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     MI))
xy_H <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     H))
MIH <-
  ggplot(uncertainty, aes(x = H, y = MI)) +
  geom_density_2d_filled() +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log2")
MIH_points <- ggplot(uncertainty, aes(x = H, y = MI)) +
  geom_point() +
  scale_x_continuous(trans = "log2") +
  scale_y_continuous(trans = "log2")
xy_MI + xy_H + MIH + MIH_points + plot_layout(ncol=2)

