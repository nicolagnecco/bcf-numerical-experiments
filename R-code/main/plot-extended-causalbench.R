source("main/dependencies.R")

res <- read_csv("../results/try-extended/n_preds_1-n_conf_0-n_trainenv_1/20240725-201041/causalbench-res.csv")

res %>% select(training_envs) %>% unique()

sum_res <- res %>% 
  group_by(algorithm, response) %>% 
  summarise(mse = max(mse))

res %>% 
  group_by(algorithm) %>% 
  summarise(max(mse))


ggplot(sum_res) +
  geom_boxplot(aes(x = algorithm, y = mse))

# Imp seems to be bad if we get directions wrong?

