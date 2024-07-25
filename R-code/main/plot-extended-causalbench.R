source("main/dependencies.R")

res <- read_csv("../results/try-extended/n_preds_3-n_conf_1-n_trainenv_1/20240725-175202/causalbench-res.csv")

sum_res <- res %>% 
  group_by(response, predictors, confounders, algorithm, run_id) %>% 
  summarise(mse = max(mse))

sum_res %>% 
  group_by(algorithm) %>% 
  summarise(mse = mean(mse))

ggplot(sum_res) +
  geom_boxplot(aes(x = algorithm, y = mse))


res %>% filter(test_envs=="ENSG00000108298") %>% view(
  
)
