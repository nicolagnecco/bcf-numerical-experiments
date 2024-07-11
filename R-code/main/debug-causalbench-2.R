library(tidyverse)


# Import data
dat <- read_csv(
  "../results/output_data/20240711-190321/causalbench-data_ENSG00000109475.csv"
)

genes <- colnames(dat)[1:6]

# Constants for gene column names
gene_x1 <- genes[2]
gene_y <- genes[1]

# Look at fits
fitlm <- lm(ENSG00000138326 ~ ENSG00000231500 + ENSG00000144713 + ENSG00000161970,
            data = dat %>% filter(set == "train"))
summary(fitlm)

plot(dat$ENSG00000231500, dat$ENSG00000138326)
points(dat$ENSG00000231500, predict(fitlm, newdata = dat), col = "green")

# Look how many training samples are non-observational
dat2plot <- dat %>% filter(environment == 0, n_env_top == 20)

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y), col = Z), alpha = 0.1) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y)), col = "red",
             data = dat2plot %>% filter(Z != "non-targeting", set == "train")) +
  coord_fixed()

# Look train-test split
dat2plot <- dat %>% filter(environment == 0, n_env_top == 20)

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y), col = set), alpha = 0.1) +
  coord_equal()

# Look how predictions look like
ggplot(dat %>% filter(set %in% c("train", "test"), n_env_top == 20)) +
  facet_grid(environment ~ algorithm) +
  geom_point(aes(x = !!sym(genes[3]), y = !!sym(gene_y))) +
  geom_point(aes(x = !!sym(genes[3]), y = y_pred, col = algorithm)) +
  coord_cartesian(xlim = c(0, 5))


# Look at covariate space
ggplot(dat %>% filter(environment == 2)) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(genes[3]), col = Z)) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(genes[3])), col = "red",
             data = dat %>% filter(Z != "non-targeting", set == "train")) 

X_reg <- dat %>% filter(set %in% c("train")) %>% select(genes[2:4], Z)

lm(ENSG00000149273 ~ Z, data = X_reg)

mean(X_reg$ENSG00000149273)

# Compute mses
dat_mses <- dat %>% 
  group_by(environment, algorithm, n_env_top, set) %>%
  summarise(mse = mean((!!sym(gene_y) - y_pred)^2), .groups = 'drop')

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
  