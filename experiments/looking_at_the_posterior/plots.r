library(ggplot2)
library(ggExtra)
library(dplyr)
library(matlab)
library(patchwork)
library(ggpubr)

load_uq <- function(csv_file, model, dataset) {
  df <- read.csv(csv_file);
  df$model <- model
  df$dataset <- dataset
  return(df)
}

d1 <- load_uq("./alvis/looking_at_the_posterior/conv_latest/conv_uq_CIFAR10C_all_[1, 2, 3, 4, 5].csv", "conv", "c")
d2 <- load_uq("alvis/looking_at_the_posterior/mlp_latest/mlp_uq_CIFAR10C_all_[1, 2, 3, 4, 5].csv", "mlp", "c")
d3 <- load_uq("alvis/looking_at_the_posterior/swin_latest/swin_uq_CIFAR10C_all_[1, 2, 3, 4, 5].csv", "swin", "c")

#d4 <- load_uq("experiments/looking_at_the_posterior/conv_uq_cifar_val.csv", "conv", "val")
d4 <- load_uq("alvis/looking_at_the_posterior/conv_cifar_latest/conv_uq_cifar_val.csv", "conv", "val")
d5 <- load_uq("alvis/looking_at_the_posterior/mlp_latest/mlp_uq_CIFAR10_val.csv", "mlp", "val")
d6 <- load_uq("alvis/looking_at_the_posterior/swin_latest/swin_uq_CIFAR10_val.csv", "swin", "val")

subsets <- c(
  "brightness",
  "contrast",
  "defocus_blur",
  "elastic_transform",
  "fog",
  "frost",
  "gaussian_blur",
  "gaussian_noise",
  "glass_blur",
  "impulse_noise",
  "jpeg_compression",
  "motion_blur",
  "pixelate",
  "saturate",
  "shot_noise",
  "snow",
  "spatter",
  "speckle_noise",
  "zoom_blur"
)


df <- bind_rows(d1, d2, d3, d4, d5, d6);

df$subsetname <- factor(df$subset, levels = 0:(length(subsets)-1), labels = subsets)

make_plot <- function(model_name) {
small_df <- df %>% filter(model == "swin") #df[sample(nrow(df), 5000), ]
num_levels <- 2
p<-(ggplot(small_df, aes(x=H, y=MI, z=accuracy)) + 
       stat_summary_2d(fun="mean", bins=50, aes(fill=..value..)) + 
       geom_density2d(contour_var = "ndensity", bins=num_levels) +
       geom_tile() +
       #scale_fill_viridis_c(name="Avg. Accuracy", direction = 1) + 
    scale_fill_viridis_c(name="Avg. Accuracy", direction = 1,option = "H", alpha=1) + 
       scale_x_log10(limits=c(1e-3, 3)) +
       scale_y_log10(limits=c(1e-3, 3)) +
       theme_minimal() +
       labs(x="H", y="MI"))
   #facet_grid(model ~ subsetname));

densx <- (ggplot(small_df, aes(x=H, y=accuracy)) + 
  stat_summary_bin(fun=mean, geom="bar", bins=50, alpha=0.5, aes(fill=..y..), color="black") +
    scale_fill_viridis_c(name="Avg. Accuracy", direction = 1,option = "H", alpha=1) +
    scale_x_log10(limits=c(1e-3, 3)) +
  theme_minimal() + 
  theme(legend.position = "none"))
densy <- (ggplot(small_df, aes(x=MI, y=accuracy)) + 
            stat_summary_bin(fun=mean, geom="bar", bins=50, alpha=0.5, fill="grey", color="black") +
            scale_x_log10(limits=c(1e-3, 3)) +
            theme_minimal() + 
            theme(legend.position = "none") +
            coord_flip() + scale_y_reverse())
            #coord_trans(x="reverse", y="reverse"))
            #geom_density(alpha = 0.4) + 
            #theme_void() + 
            #theme(legend.position = "none"))

fp<-plot_spacer() + densx + densy + p +
  plot_layout(ncol = 2, nrow = 2, widths = c(1, 6), heights = c(1, 6))
fp
}
#make_plot("mlp") + make_plot("conv") + make_plot("swin")
print(fp)
#fp + facet_grid(model ~ subsetname)
print(densx)
  
print(ggMarginal(p, type="histogram"))

(ggplot(df, aes(x=H, y=MI, z=severity)) + 
   stat_summary_2d(fun="mean", bins=50, aes(fill=..value..)) + 
   geom_density2d(contour_var = "ndensity", bins=num_levels) +
   geom_tile() +
   #scale_fill_viridis_c(name="severity", direction = 1) + 
   scale_fill_viridis_c(name="Avg. Accuracy", direction = 1,option = "H", alpha=1) + 
   scale_x_log10(limits=c(1e-3, 3)) +
   scale_y_log10(limits=c(1e-3, 3)) +
   theme_minimal() +
   labs(x="H", y="MI", title="2D Binned Plot with avg. severity") +
   facet_grid(model ~ subsetname));


num_levels <- 2 # df[df$model == "swin",]
(ggplot(small_df, aes(x=H, y=MI, z=accuracy)) + 
    stat_summary_2d(fun="mean", bins=50, aes(fill=..value..)) + 
    geom_density2d(contour_var = "ndensity", bins=num_levels) +
    geom_tile() +
    #scale_fill_gradientn(colours = rev(rainbow(n=4, start=0, end=0.75, s=1, v=1))) +
    scale_fill_viridis_c(name="Avg. Accuracy", direction = 1,option = "H", alpha=1) + 
    scale_x_log10(limits=c(1e-3, 3)) +
    scale_y_log10(limits=c(1e-3, 3)) +
    theme_minimal() +
    labs(x="H", y="MI", title="2D Binned Plot with Avg. Accuracy"));# +
    #facet_grid(severity ~ subsetname));
