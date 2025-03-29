library(ggplot2)
library(dplyr)

df1 <- read.csv("./alvis/al_f9f8d3a/conv_calibrated_uq.csv")
df2 <- read.csv("./alvis/al_83027bc/conv_mutual_information.csv")
df3 <- read.csv("./alvis/al_c411ce9/conv_calibrated_uq_tn.csv")
df4 <- read.csv("./alvis/al_f49fd3a/conv_predictive_entropy.csv")
df5 <- read.csv("./alvis/al_abd3729/mlp_calibrated_uq_tn.csv")
df6 <- read.csv("./alvis/al_cc77658/mlp_calibrated_uq.csv")
df7 <- read.csv("./alvis/al_7e2e4b5/mlp_predictive_entropy.csv")
df8 <- read.csv("./alvis/al_53d0569/mlp_mutual_information.csv")
df9 <- read.csv("./alvis/al_994fb59/conv_random.csv")
dfa <- read.csv("./alvis/al_a230171/mlp_random.csv")

#dfb <- read.csv("./alvis/al_9b93bd7/swin_mutual_information.csv")
#dfc <- read.csv("./alvis/al_ea31a52/swin_predictive_entropy.csv")
#dfd <- read.csv("./alvis/al_bf1828d/swin_calibrated_uq.csv")
#dfe <- read.csv("./alvis/al_bf2a2c0/swin_random.csv")
#dff <- read.csv("./alvis/al_00a98fc/swin_calibrated_uq_tn.csv")

dfb <- read.csv("./alvis/al_1df6dbd/swin_predictive_entropy.csv")
dfc <- read.csv("./alvis/al_e064552/swin_random.csv")
dfd <- read.csv("./alvis/al_36fe252/swin_mutual_information.csv")
dfe <- read.csv("./alvis/al_0d0d751/swin_calibrated_uq.csv")
dff <- read.csv("./alvis/al_169c367/swin_calibrated_uq_tn.csv")

df <- rbind(df1,df2,df3,df4,df5,df6,df7,df8,df9,dfa,dfb,dfc,dfd,dfe,dff)
p<-(ggplot(df, aes(x=n_aquired, y=value, group=interaction(model, aquisition), color=aquisition)) + geom_line(aes(linetype=model)) + 
      facet_grid(rows=vars(metric),scales="free_y") +
  #    xlim(0, 100) +
      theme_minimal(base_size = 20))
ggsave("./experiments/looking_at_the_posterior/al_cifar_unbalanced.pdf", p)
print(p)

df_cal<-read.csv("./alvis/al_cc77658/uq_calibration_step_018_calibrated_uq.csv")
df_aq <-read.csv("./alvis/al_cc77658/uq_aquired_step_018_calibrated_uq.csv")
df_pool <-read.csv("./alvis/al_cc77658/uq_pool_step_018_calibrated_uq.csv")
min_samples_mean <- function(z) {
  if (length(z) >= 10) {
    return(mean(z))
  } else {
    return(NA)  # return NA for bins with less than 10 samples
  }
}

(ggplot() +
    stat_summary_2d(data=df_cal, aes(x=H, y=MI, z=accuracy, fill=..value..), fun=min_samples_mean, bins=15) +
    geom_point(data=df_aq, aes(x=H, y=MI), color="red", fill="green", shape=19, size=5) +
    geom_point(data=df_pool %>% filter(target < 5), aes(x=H, y=MI), color="red", shape=2) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40) #+ 
  #scale_x_continuous(trans = log_trans()) +
  #scale_y_continuous(trans = log_trans())
)
