library(tidyverse)


# import data
res <- read_csv("../results/check_data/20240722-125309/res.csv")

ggplot(res) +
  geom_boxplot(aes(x = factor(algorithm), 
                   y = mse_test, 
                   col = factor(algorithm)), outlier.shape = 21) +
  theme_bw() 

res %>% filter(algorithm=="BCF") %>% view
genes <- res$gene %>% unique()

dat2plot <- res %>% group_by(algorithm, n_envs) %>% 
  # filter(gene == genes[1]) %>%
  summarise(mse_test = max(mse_test),
            mse_train = max(mse_train)) %>% 
  pivot_longer(cols = c(mse_train, mse_test), names_to = "setting", values_to = "mse")

ggplot(dat2plot) +
  facet_wrap(~ setting, scales = "free_y") + 
  geom_line(aes(x = n_envs, y = mse, col = algorithm)) +
  geom_point(aes(x = n_envs, y = mse, col = algorithm), shape = 21, fill = "white") +
  theme_bw()


