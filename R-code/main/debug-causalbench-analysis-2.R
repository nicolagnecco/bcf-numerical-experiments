source("main/dependencies.R")

# Import data
dat <- read_csv(
  "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240802-141916/_debug/debug-response_ENSG00000083845-run_id_26-iter_id_0.csv"
)

dat <- read_csv(
  "../results/causalbench-analysis/n_preds_9-n_trainenv_5-confounders_False/20240806-120957/_debug/debug_response_ENSG00000042429-run_id_50-iter_id_0.csv"
)

genes <- colnames(dat)[1:4]


# Constants for gene column names
gene_x1 <- genes[2]
gene_y <- genes[1]


# Look how many training samples are non-observational
dat2plot <- dat %>% 
  select(-y_pred) %>% 
  filter(algo == "BCF") %>% 
  filter(env == "train" | (Z == genes[3])) %>% 
  unique() 
  

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(genes[3]), y = !!sym(genes[2]), col = env), 
             alpha = 0.5) +
  coord_cartesian(xlim = c(0, 5))


# Look at predictions
dat2plot <- dat %>% 
  filter(env == "train") %>% 
  unique()

dat_train <- dat %>% 
  filter(algo == "BCF") %>% 
  filter(env == "train")

dat_test <- dat %>% 
  filter(algo == "BCF") %>% 
  filter(env == "test")


ggplot(dat) +
  facet_grid(env~ algo) +
  geom_point(aes(x = !!sym(genes[2]), y = !!sym(gene_y))) +
  geom_point(aes(x = !!sym(genes[2]), y = y_pred, col = algo)) +
  coord_cartesian(xlim = c(0, 5))
