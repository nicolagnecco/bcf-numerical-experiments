library(tidyverse)


# Import data
dat <- read_csv(
  "../results/output_data/20240710-190751/causalbench-data_ENSG00000083845.csv"
)

ENV <- 1

datsubsamp <- bind_rows(
  dat %>% filter(Z == "non-targeting") %>% group_by(environment, algorithm) %>% slice_sample(n = 1000),
  dat %>% filter(Z != "non-targeting")
) %>% 
  mutate(setenv = paste(Z, set, sep = "-"))

genes <- colnames(datsubsamp)[1:6]

# Constants for gene column names
GENE_A <- genes[2]
GENE_B <- genes[1]

# Look how many training samples are non-observational
dat2plot <- datsubsamp %>% 
  filter(environment == 1)

X <- dat2plot %>% ungroup() %>% filter(Z == "non-targeting") %>% select(2:3)

apply(X, MARGIN = 2, function(x){quantile(x, probs = .0)})

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(GENE_A), y = !!sym(GENE_B), col = Z), alpha = 0.1) +
  geom_point(aes(x = !!sym(GENE_A), y = !!sym(GENE_B)), col = "red",
             data = dat2plot %>% filter(Z != "non-targeting", set == "train"))

# Look train-test split
dat2plot <- datsubsamp %>% 
  filter(environment == 1)

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(GENE_A), y = !!sym(GENE_B), col = set), alpha = 0.1) +
  xlim(c(0, 6)) +
  ylim(c(0, 5))

# Look how predictions look like
ggplot(dat %>% filter(set %in% c("test", "train"))) +
  facet_grid(environment ~ algorithm) +
  geom_point(aes(x = !!sym(GENE_A), y = !!sym(GENE_B))) +
  geom_point(aes(x = !!sym(GENE_A), y = y_pred, col = algorithm))

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
  