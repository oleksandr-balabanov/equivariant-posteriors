library(patchwork)
library(ggplot2)
library(Hmisc)
library(grid)
# library(gridExtra)
library(dplyr)

uncertainty <-
  read.csv("./experiments/lyapunov2/cifar_conv_uncertainty_mnist.csv")
uncertainty$label <- as.character(uncertainty$pred)
# binned_stats <- function(dataframe, column) {
# return(ggplot2::ggplot(dataframe, aes(x=exp(FTLE), y=.data[[column]]))
#   #+ geom_point(alpha=0.1, size=0.2, color="blue")
#   #+ geom_density2d(aes(color=factor(llabel)))
#   + stat_summary_bin(fun.data="mean_sdl", aes(color=factor(llabel)), bins=60)
#   #+ stat_summary_bin(fun.y="mean", aes(y=MI, shape="MI", color=factor(llabel)), bins=60)
#   #+ stat_summary_bin(fun.y="mean", aes(y=H, shape="H", color=factor(llabel)), bins=60)
#   + scale_y_continuous(trans="log2")
#   + facet_grid( . ~ data)
#   )
# }
# (binned_stats(uncertainty, "H") +
#     binned_stats(uncertainty, "A") +
#     binned_stats(uncertainty, "MI") + plot_layout(ncol=1))

# (ggplot2::ggplot(uncertainty, aes(x=H, y=MI, label=label)) 
#   + geom_text(size=3, alpha=0.1, aes(color=factor(label)))
#   + geom_density2d(aes(color=factor(label)))
#   + scale_x_continuous(trans="log2")
#   + scale_y_continuous(trans="log2")
# )
# ggsave("cifar_conv_uq_mnist_2.pdf")

lambda_order <- uncertainty[order(uncertainty$lambda, decreasing=TRUE),]
ftle_xy <- (ggplot2::ggplot(lambda_order, aes(x=x, y=y, label=label))
 + geom_text(size=3, alpha=0.5, aes(color=lambda)) + scale_color_gradientn(colors=rainbow(3))) 
ggsave("cifar_conv_ftle_xy.pdf", ftle_xy)

(ggplot2::ggplot(uncertainty, aes(x=x, y=y, label=label)) 
  #+ geom_point(aes(shape=data, color=data), alpha=0.1)
  #+ scale_color_gradient()
  + geom_text(size=3, alpha=0.1, aes(color=factor(label)))
  + geom_density2d(aes(color=factor(label)))
  # + scale_x_continuous(trans="log2")
  # + scale_y_continuous(trans="log2")
  # + stat_summary_bin(fun.y="mean", bins=60)
  # + stat_summary_bin(fun.y="mean", aes(shape=data), bins=60)
)
ggsave("cifar_conv_uq_mnist_xyL.pdf")

# ftle_mi_exp_lambda <- binned_stats(uncertainty_test)
(ftle_mi_lambda <- ggplot2::ggplot(uncertainty, aes(x=lambda, y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  + stat_summary_bin(fun.data="mean_sdl", bins=60)
  + scale_y_continuous(trans="log2"))
ggsave("cifar_conv_lambda_MI.pdf", ftle_mi_lambda)
  
# )
# (ftle_mi_exp_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=exp(FTLE), y=MI))
#   + geom_point(alpha=0.1, size=0.2, color="blue") 
#   + stat_summary_bin(fun.y="mean", bins=60)
#   + scale_y_continuous(trans="log2")
# )
# (ftle_mi_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=FTLE, y=MI))
#   + geom_point(alpha=0.1, size=0.2, color="blue") 
#   + stat_summary_bin(fun.y="mean", bins=60)
#   + scale_y_continuous(trans="log2")
# )
  
# (
#   (ftle_mi_lambda_log)
# + (ftle_mi_exp_lambda_log)
# + ftle_mi_lambda 
# + ftle_mi_exp_lambda 
# + plot_layout(ncol=2)
# )

# # Assume we have a dataframe uncertainty with columns FTLE and H
# #uncertainty <- data.frame(FTLE = runif(100), H = runif(100))

# # Define the number of bins you want
# num_bins <- 60

# # Define the break points 
# break_points <- seq(min(uncertainty$FTLE), max(uncertainty$FTLE), length.out = num_bins + 1)

# # Create a new variable in the data frame that represents the bin
# uncertainty$FTLE_bin <- cut(uncertainty$FTLE, breaks = break_points, labels = FALSE)

# # Define a function that calculates both the mean and standard deviation
# mean_sdl_custom <- function(H) {
#   return(c(H_mean = mean(H, na.rm = TRUE), H_sd = sd(H, na.rm = TRUE)))
# }

# # Now calculate mean and standard deviation of H values by FTLE_bin
# binned_data <- tapply(uncertainty$H, uncertainty$FTLE_bin, FUN = mean_sdl_custom)

# # Compute midpoints of FTLE bins
# bin_midpoints <- (break_points[-1] + break_points[-length(break_points)]) / 2

# # Create a dataframe with the midpoints and bin index
# bin_df <- data.frame(
#   FTLE_bin = 1:num_bins,
#   FTLE_bin_midpoint = bin_midpoints
# )

# # Create a dataframe with the mean and sd per bin
# binned_df <- data.frame(
#   FTLE_bin = as.numeric(names(binned_data)),
#   H_mean = unlist(binned_data, use.names = FALSE)[c(TRUE, FALSE)],  # extract mean
#   H_sd = unlist(binned_data, use.names = FALSE)[c(FALSE, TRUE)]  # extract sd
# )

# # Merge the two dataframes
# final_df_H <- merge(bin_df, binned_df, by = "FTLE_bin", all.x = TRUE)
# (
#   ggplot2::ggplot(final_df, aes(x=FTLE_bin, y=H_mean)) + geom_point(color="red")
#   + geom_errorbar(aes(ymin=H_mean - H_sd, ymax=H_mean + H_sd))
#   + ftle_mi_lambda
# )
# write.csv(final_df, file = "binned_FTLE_H_test.csv", row.names = FALSE)

# binned_data <- tapply(uncertainty$MI, uncertainty$FTLE_bin, FUN = mean_sdl_custom)
# binned_df <- data.frame(
#   FTLE_bin = as.numeric(names(binned_data)),
#   H_mean = unlist(binned_data, use.names = FALSE)[c(TRUE, FALSE)],  # extract mean
#   H_sd = unlist(binned_data, use.names = FALSE)[c(FALSE, TRUE)]  # extract sd
# )

# # Merge the two dataframes
# final_df_MI <- merge(bin_df, binned_df, by = "FTLE_bin", all.x = TRUE)
# write.csv(final_df, file = "binned_FTLE_MI_test.csv", row.names = FALSE)
