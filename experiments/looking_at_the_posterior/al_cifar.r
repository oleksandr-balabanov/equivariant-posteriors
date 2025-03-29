library(ggplot2)

df1 <- read.csv("./alvis/al_0e58c0a/conv_random.csv")
df2 <- read.csv("./alvis/al_ef6b78b/conv_calibrated_uq.csv")
df3 <- read.csv("./alvis/al_f5e2afc/mlp_calibrated_uq_tn.csv")
df4 <- read.csv("./alvis/al_96a9d7f/mlp_calibrated_uq.csv")
df5 <- read.csv("./alvis/al_b9a18af/conv_mutual_information.csv")
df6 <- read.csv("./alvis/al_cc65411/mlp_random.csv")
df7 <- read.csv("./alvis/al_6cfb501/conv_predictive_entropy.csv")
df8 <- read.csv("./alvis/al_6a07e38/mlp_mutual_information.csv")
df9 <- read.csv("./alvis/al_ceb41d9/conv_calibrated_uq_tn.csv")
dfa <- read.csv("./alvis/al_6f9caaf/mlp_predictive_entropy.csv")
df <- rbind(df1,df2,df3,df4,df5,df6,df7,df8,df9,dfa)
p<-(ggplot(df, aes(x=fraction, y=value, group=interaction(model, aquisition), color=aquisition)) + geom_line(aes(linetype=model)) + 
      facet_grid(rows=vars(metric),scales="free_y") +
      theme_minimal())
ggsave("./experiments/looking_at_the_posterior/al_cifar.pdf", p)
print(p)

