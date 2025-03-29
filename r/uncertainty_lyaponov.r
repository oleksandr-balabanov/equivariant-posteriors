library(patchwork)
library(ggplot2)
library(Hmisc)
library(grid)
library(gridExtra)
library(dplyr)

uncertainty_train <-
  read.csv("/home/herden/projects/equivariant_posteriors/experiments/lyaponov/mnist_train_uncertainty.csv")
uncertainty_test <-
  read.csv("/home/herden/projects/equivariant_posteriors/experiments/lyaponov/mnist_test_uncertainty.csv")

uncertainty_test <- uncertainty_test %>% mutate(data="test")
uncertainty_train <- uncertainty_train %>% mutate(data="train")
uncertainty <- bind_rows(uncertainty_test, uncertainty_train)

xy_MI <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     MI),
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))
xy_H <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     H), 
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))

xy_A <-
  ggplot2::ggplot(uncertainty, aes(x = x, y = y)) + geom_point(aes(color =
                                                                     A), 
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))
xy_L <-
  ggplot2::ggplot(uncertainty, aes(x = lx, y = ly)) + geom_point(aes(color =
                                                                     FTLE), 
                                                               size=0.6,
                                                               alpha=0.5) + scale_color_gradientn(colours = rainbow(5))

#MIH <-
#  ggplot(uncertainty, aes(x = H, y = MI)) +
#  geom_density_2d_filled() +
#  scale_x_continuous(trans = "log2") +
#  scale_y_continuous(trans = "log2")
#MIH_points <- ggplot(uncertainty, aes(x = H, y = MI)) +
#  geom_point() +
#  scale_x_continuous(trans = "log2") +
#  scale_y_continuous(trans = "log2")
##xy_MI + xy_H + MIH + MIH_points + plot_layout(ncol=2)
full <- xy_MI + xy_H + xy_L# + plot_layout(ncol=3)
#lim <- xlim(xmin, xmax) + ylim(ymin, ymax)
xmin <- -2.7
xmax <- 8
ymin <- -13
ymax <- -7.0
#xmin <- -13
#xmax <- -7
#ymin <- -7.5
#ymax <- -2.5

mi_b <- xy_MI + xlim(xmin, xmax) + ylim(ymin, ymax) 
h_b <-  xy_H + xlim(xmin, xmax) + ylim(ymin, ymax)
FTLE_b <- xy_L + xlim(xmin, xmax) + ylim(ymin, ymax)

#+ plot_layout(ncol=3)
(xy_MI + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black")) + 
  xy_H  + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black") + 
  xy_L  + geom_rect(aes(xmin=xmin, xmax=xmax, ymin=ymin,ymax=ymax), fill=NA, color="black") + 
  mi_b + h_b + FTLE_b + plot_layout(ncol=3)

y_min <- min(min(uncertainty_test$MI), min(uncertainty_train$MI))
y_max <- max(max(uncertainty_test$H), max(uncertainty_train$H))

binned_stats <- function(dataframe, column) {
return(ggplot2::ggplot(dataframe, aes(x=exp(FTLE), y=.data[[column]]))
  #+ geom_point(alpha=0.1, size=0.2, color="blue")
  #+ geom_density2d(aes(color=factor(llabel)))
  + stat_summary_bin(fun.data="mean_sdl", aes(color=factor(llabel)), bins=60)
  #+ stat_summary_bin(fun.y="mean", aes(y=MI, shape="MI", color=factor(llabel)), bins=60)
  #+ stat_summary_bin(fun.y="mean", aes(y=H, shape="H", color=factor(llabel)), bins=60)
  + scale_y_continuous(trans="log2")
  + facet_grid( . ~ data)
  )
}
(binned_stats(uncertainty, "H") +
    binned_stats(uncertainty, "A") +
    binned_stats(uncertainty, "MI") + plot_layout(ncol=1))

(ggplot2::ggplot(uncertainty, aes(x=H, y=MI, label=llabel)) 
  #+ geom_point(aes(shape=data, color=data), alpha=0.1)
  #+ scale_color_gradient()
  + geom_text(size=3, alpha=0.1, aes(color=factor(llabel)))
  + geom_density2d(aes(color=factor(llabel)))
  + scale_x_continuous(trans="log2")
  + scale_y_continuous(trans="log2")
  + stat_summary_bin(fun.y="mean", aes(shape=data), bins=60)
  + stat_summary_bin(fun.y="mean", aes(shape=data), bins=60)
)

ftle_mi_exp_lambda <- binned_stats(uncertainty_test)
(ftle_mi_lambda <- ggplot2::ggplot(uncertainty, aes(x=FTLE, y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  + stat_summary_bin(fun.data="mean_sdl", bins=60)
  
)
(ftle_mi_exp_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=exp(FTLE), y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  + stat_summary_bin(fun.y="mean", bins=60)
  + scale_y_continuous(trans="log2")
)
(ftle_mi_lambda_log <- ggplot2::ggplot(uncertainty, aes(x=FTLE, y=MI))
  + geom_point(alpha=0.1, size=0.2, color="blue") 
  + stat_summary_bin(fun.y="mean", bins=60)
  + scale_y_continuous(trans="log2")
)
  
(
  (ftle_mi_lambda_log)
+ (ftle_mi_exp_lambda_log)
+ ftle_mi_lambda 
+ ftle_mi_exp_lambda 
+ plot_layout(ncol=2)
)

# Assume we have a dataframe uncertainty with columns FTLE and H
#uncertainty <- data.frame(FTLE = runif(100), H = runif(100))

# Define the number of bins you want
num_bins <- 60

# Define the break points 
break_points <- seq(min(uncertainty$FTLE), max(uncertainty$FTLE), length.out = num_bins + 1)

# Create a new variable in the data frame that represents the bin
uncertainty$FTLE_bin <- cut(uncertainty$FTLE, breaks = break_points, labels = FALSE)

# Define a function that calculates both the mean and standard deviation
mean_sdl_custom <- function(H) {
  return(c(H_mean = mean(H, na.rm = TRUE), H_sd = sd(H, na.rm = TRUE)))
}

# Now calculate mean and standard deviation of H values by FTLE_bin
binned_data <- tapply(uncertainty$H, uncertainty$FTLE_bin, FUN = mean_sdl_custom)

# Compute midpoints of FTLE bins
bin_midpoints <- (break_points[-1] + break_points[-length(break_points)]) / 2

# Create a dataframe with the midpoints and bin index
bin_df <- data.frame(
  FTLE_bin = 1:num_bins,
  FTLE_bin_midpoint = bin_midpoints
)

# Create a dataframe with the mean and sd per bin
binned_df <- data.frame(
  FTLE_bin = as.numeric(names(binned_data)),
  H_mean = unlist(binned_data, use.names = FALSE)[c(TRUE, FALSE)],  # extract mean
  H_sd = unlist(binned_data, use.names = FALSE)[c(FALSE, TRUE)]  # extract sd
)

# Merge the two dataframes
final_df_H <- merge(bin_df, binned_df, by = "FTLE_bin", all.x = TRUE)
(
  ggplot2::ggplot(final_df, aes(x=FTLE_bin, y=H_mean)) + geom_point(color="red")
  + geom_errorbar(aes(ymin=H_mean - H_sd, ymax=H_mean + H_sd))
  + ftle_mi_lambda
)
write.csv(final_df, file = "binned_FTLE_H_test.csv", row.names = FALSE)

binned_data <- tapply(uncertainty$MI, uncertainty$FTLE_bin, FUN = mean_sdl_custom)
binned_df <- data.frame(
  FTLE_bin = as.numeric(names(binned_data)),
  H_mean = unlist(binned_data, use.names = FALSE)[c(TRUE, FALSE)],  # extract mean
  H_sd = unlist(binned_data, use.names = FALSE)[c(FALSE, TRUE)]  # extract sd
)

# Merge the two dataframes
final_df_MI <- merge(bin_df, binned_df, by = "FTLE_bin", all.x = TRUE)
write.csv(final_df, file = "binned_FTLE_MI_test.csv", row.names = FALSE)
