library(tidyverse)


# Import data
dat <- read_csv(
  "../results/check_data/20240711-111238/data_0.csv"
)

# Look how the covariate space looks like
ggplot(dat) +
  geom_point(aes(x = X1, y = X2, col = set), alpha = .1)

# Look how predictions look like
ggplot(dat %>% filter(set %in% c("train", "test"))) +
  facet_grid(set~.) +
  geom_point(aes(x = X1, y = y, col = set), alpha = .1) 


ggplot(dat %>% filter(set %in% c("test", "train"))) +
  facet_grid(set~algorithm) +
  geom_point(aes(x = X1, y = y)) +
  geom_point(aes(x = X1, y = y_pred, col = algorithm))

# Compute mses
dat_mses <- dat %>% 
  group_by(environment, algorithm, set) %>%
  summarise(mse = mean((!!sym(GENE_B) - y_pred)^2), .groups = 'drop')

ggplot(dat_mses) +
  facet_wrap(~ set) +
  geom_line(aes(x = environment, y = mse, col = algorithm)) +
  geom_point(aes(x = environment, y = mse, col = algorithm), shape = 21, fill = "white")

# Compute correlations
dat_cor <- dat %>% 
  filter(Z == "non-targeting") %>% 
  select(1:6)

colMeans(dat_cor) 

dat_cor <- dat %>% 
  filter(set == "test") %>% 
  select(1:6)

colMeans(dat_cor)
  