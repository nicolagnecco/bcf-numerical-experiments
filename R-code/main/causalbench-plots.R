library(tidyverse)


# import data
res <- read_csv("../results/output_data/20240723-102826/causalbench-res.csv") %>% 
  # filter(is.na(M_0) | M_0 != "[[0.]\n [0.]\n [0.]\n [0.]\n [0.]\n [0.]\n [0.]\n [0.]\n [0.]\n [0.]]") %>% 
  filter(env_id == 0)

ggplot(res) +
  geom_boxplot(aes(x = factor(-n_env_obs), 
                   y = mse_test, 
                   col = factor(algorithm)), outlier.shape = 21) +
  theme_bw() 

ggplot(res, mapping = aes(x = factor(-n_env_obs), 
                   y = mse_test, 
                   col = factor(algorithm))) +
  geom_boxplot(outlier.shape=NA) + 
  geom_point(position=position_jitterdodge()) +
  theme_bw() +
  coord_cartesian(ylim = c(0, 1))

genes <- res$gene %>% unique()

dat2plot <- res %>% group_by(algorithm, n_env_obs) %>% 
  summarise(mse_test = mean(mse_test),
            mse_train = mean(mse_train)) %>% 
  pivot_longer(cols = c(mse_train, mse_test), names_to = "setting", values_to = "mse")

ggplot(dat2plot) +
  facet_wrap(~ setting, scales = "free_y") + 
  geom_line(aes(x = -n_env_obs, y = mse, col = algorithm)) +
  geom_point(aes(x = -n_env_obs, y = mse, col = algorithm), 
             shape = 21, fill = "white") +
  coord_cartesian(ylim = c(0, .5))
  
