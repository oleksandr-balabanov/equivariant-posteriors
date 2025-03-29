library(ggplot2)
library(dplyr)
library(latex2exp)

database <-
  read.csv("database.csv")
plot <- ggplot(
  dplyr::filter(
    database,
    ensemble_id != "",
    #train_config.model.config.embed_d %in% c(5, 20, 100)
    train_config.model.config.mlp_dim %in% c(1, 5, 20)
  ),
  aes(
    x = epoch,
    y = accuracy,
    group = ensemble_id,
    fill = interaction(factor(train_config.model.config.mlp_dim), factor(train_config.model.config.embed_d)),
    color = train_config.model.config.mlp_dim,
    linetype = interaction(factor(train_config.model.config.mlp_dim), factor(train_config.model.config.embed_d)),
  )
) + stat_summary(
  geom = "ribbon",
  fun.min = function(x)
    quantile(x, 0.1),
  fun.max = function(x)
    quantile(x, 0.9),
  alpha = 0.2,
  color = NA
) + stat_summary(geom = "line", fun = median, color = "black") + scale_fill_brewer(palette="Spectral") + labs(linetype=TeX("($d_{MLP}$, $d_{embed}$)"), fill=TeX("($d_{MLP}$, $d_{embed}$)"))
ggsave("mlp_dim.pdf")
