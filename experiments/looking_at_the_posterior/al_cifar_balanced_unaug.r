library(ggplot2)
library(dplyr)

df1 <- read.csv("./alvis/al_a9257be/conv_calibrated_uq_tn.csv")
df2 <- read.csv("./alvis/al_8ceba8d/conv_calibrated_uq.csv")
df3 <- read.csv("./alvis/al_52feb2d/conv_predictive_entropy.csv")
df4 <- read.csv("./alvis/al_3be58db/mlp_calibrated_uq_tn.csv")
df5 <- read.csv("./alvis/al_7635944/conv_mutual_information.csv")
df6 <- read.csv("./alvis/al_87d4450/mlp_calibrated_uq.csv")
df7 <- read.csv("./alvis/al_eecb9a7/mlp_mutual_information.csv")
df8 <- read.csv("./alvis/al_db45db6/mlp_predictive_entropy.csv")
df9 <- read.csv("./alvis/al_8ac1e0b/conv_random.csv")
dfa <- read.csv("./alvis/al_dd352a5/mlp_random.csv")
df <- rbind(df1,df2,df3,df4,df5,df6,df7,df8,df9,dfa)
p<-(ggplot(df, aes(x=fraction, y=value, group=interaction(model, aquisition), color=aquisition)) + geom_line(aes(linetype=model)) + 
      facet_grid(rows=vars(metric),scales="free_y") +
      theme_minimal())
ggsave("./experiments/looking_at_the_posterior/al_cifar_balanced.pdf", p)
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
