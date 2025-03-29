library(ggplot2)
library(reticulate)
library(RPostgreSQL)
library(tidyr)
library(scales)

drv <- dbDriver("PostgreSQL")
con <-
  dbConnect(
    drv,
    dbname = "equiv",
    host = "localhost",
    port = "5432",
    user = "postgres",
    password = "postgres"
  )

# Fetch data from my_table
query <- "SELECT * FROM active_learning;"
df <- dbGetQuery(con, query)

# Close the database connection
dbDisconnect(con)
df1 <- read.csv("./alvis/al_8f9d209/conv_predictive_entropy.csv")
df2 <- read.csv("./alvis/al_3a5abfa/conv_calibrated_uq.csv")
df3 <- read.csv("./alvis/al_b8b383b/conv_mutual_information.csv")
df4 <- read.csv("./alvis/al_b974f7f/conv_random.csv")
df5 <- read.csv("./alvis/al_f5c4489/conv_calibrated_uq_tn.csv")
#df6 <- read.csv("./alvis/al_aa084ae/al_config.json")

df <- rbind(df1, df2, df3, df4, df5)
p<-(ggplot(df, aes(x=fraction, y=value, group=aquisition, color=aquisition)) + geom_line(aes(linetype=aquisition)) + 
    facet_grid(rows=vars(metric),scales="free_y") +
    theme_minimal())
ggsave("./experiments/looking_at_the_posterior/conv_al.pdf", p)
print(p)

df_cal<-read.csv("./alvis/al_3a5abfa/uq_calibration_step_005_calibrated_uq.csv")
df_aq <-read.csv("./alvis/al_3a5abfa/uq_aquired_step_005_calibrated_uq.csv")
df_pool <-read.csv("./alvis/al_3a5abfa/uq_pool_step_005_calibrated_uq.csv")
min_samples_mean <- function(z) {
  if (length(z) >= 10) {
    return(length(z))
    return(mean(z))
  } else {
    return(NA)  # return NA for bins with less than 10 samples
  }
}

(ggplot() +
    stat_summary_2d(data=df_cal, aes(x=H, y=MI, z=accuracy, fill=..value..), fun=min_samples_mean, bins=15) +
    geom_point(data=df_aq, aes(x=H, y=MI), color="red", fill="green", shape=19, size=5) +
    geom_point(data=df_pool, aes(x=H, y=MI), color="red", shape=2) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40) #+ 
  #scale_x_continuous(trans = log_trans()) +
  #scale_y_continuous(trans = log_trans())
)

#N, acc
#N_tn = N * (1 - acc)


df1 <- read.csv("./alvis/al_dace34e/calibrated_uq.csv")
df2 <- read.csv("./alvis/al_ad86f40/random.csv")
df3 <- read.csv("./alvis/al_3dec3eb/predictive_entropy.csv")
df4 <- read.csv("./alvis/al_c402cb3/mutual_information.csv")
df <- rbind(df1, df2, df3, df4)
p <- (ggplot(df, aes(x=fraction, y=value, group=model, color=model)) + geom_line(aes(linetype=model)) + 
    facet_grid(~metric,scales="free_y") +
    theme_minimal())
ggsave("./experiments/looking_at_the_posterior/conv_al.pdf", p)

df <- read.csv("./experiments/looking_at_the_posterior/al_794843e/uq_calibration.csv")
p <- (ggplot(df, aes(x = fraction, y = value, group = model, color=model)) + 
        geom_line(linetype=9) + 
        facet_wrap( ~metric, scales="free_y") + 
        theme_minimal(base_size = 44))
print(p)

df_cal<-read.csv("./alvis/al_f5c4489/uq_calibration_step_015_calibrated_uq_tn.csv")
df_aq <-read.csv("./alvis/al_f5c4489/uq_aquired_step_015_calibrated_uq_tn.csv")
df_pool <-read.csv("./alvis/al_f5c4489/uq_pool_step_015_calibrated_uq_tn.csv")
np <- import("numpy")
mean_accs <- np$load("./alvis/al_28ba7eb/uq_mean_acc_step_006_calibrated_uncertainty.npy")
mean_accs <- pivot_longer()
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
    geom_point(data=df_pool, aes(x=H, y=MI), color="red", shape=2) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40) #+ 
    #scale_x_continuous(trans = log_trans()) +
    #scale_y_continuous(trans = log_trans())
  )

(ggplot() +
    stat_density2d(data=df_cal, aes(x=H, y=MI, z=accuracy, fill=..value..), fun=min_samples_mean) +
    geom_point(data=df_aq, aes(x=H, y=MI), color="red", fill="green", shape=19, size=5) +
    geom_point(data=df_pool, aes(x=H, y=MI), color="red", shape=2) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40) #+ 
  #scale_x_continuous(trans = log_trans()) +
  #scale_y_continuous(trans = log_trans())
)

(ggplot(df, aes(x=H, y=MI, z=accuracy)) +
    stat_summary_2d(aes(fill=..value..), fun="mean", bins=20) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40)
)

df<-read.csv("./alvis/al_28ba7eb/uq_pool_step_003_calibrated_uncertainty.csv")
(ggplot(df, aes(x=H, y=MI, z=accuracy)) +
    stat_summary_2d(aes(fill=..value..), fun="mean", bins=20) +
    scale_fill_viridis_c() +
    theme_minimal(base_size=40)
)
