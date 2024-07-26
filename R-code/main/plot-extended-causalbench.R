source("main/dependencies.R")

res <- read_csv("../results/try-extended/n_preds_2-n_conf_2-n_trainenv_2/20240726-093011//causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_1-n_conf_1-n_trainenv_1/20240726-092059/causalbench-res.csv")

# problems when null space is estimated wrongly
res <- read_csv("../results/try-extended/n_preds_2-n_conf_0-n_trainenv_1/20240726-100937/causalbench-res.csv") %>% 
  filter(run_id == 15)

res <- read_csv("../results/try-extended/n_preds_1-n_conf_0-n_trainenv_1/20240726-103742//causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_2-n_conf_0-n_trainenv_2/20240726-092618/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_3-n_conf_0-n_trainenv_3/20240726-102337/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-105124/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-115335/causalbench-res.csv")

# imp = 0 
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-183930/causalbench-res.csv")

sum_res <- res %>% 
  filter(algorithm %in% c("BCF", "ConstFunc", "OLS")) %>%
  group_by(algorithm, response, training_envs) %>% 
  summarise(mse = max(mse))

sum_res %>% 
  group_by(algorithm) %>% 
  summarise(mean(mse))

ggplot(sum_res, aes(x = algorithm, y = mse)) +
  geom_line(aes(group=interaction(response, training_envs)), alpha = .1) +
  geom_boxplot() +
  stat_summary(geom = "errorbar", 
               fun.min = mean, 
               fun = mean, 
               fun.max = mean, width = .75, col = "red") 
