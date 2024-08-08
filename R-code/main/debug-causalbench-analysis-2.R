source("main/dependencies.R")

# Import data
dat <- read_csv(
  "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240802-141916/_debug/debug-response_ENSG00000083845-run_id_26-iter_id_0.csv"
)

dat <- read_csv(
  "../results/causalbench-analysis-2/n_preds_3-train_mask_top-confounders_False/20240808-104520/_debug/debug-response_ENSG00000063177-run_id_23-iter_id_0.csv"
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
  

ggplot(dat2plot %>% filter(interv_strength %in% c(0, .2))) +
  geom_point(aes(x = !!sym(genes[3]), y = !!sym(genes[1]), col = Z), 
             alpha = 0.5) 


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


ggplot(dat %>% filter(interv_strength %in% c(0, 0.5))) +
  facet_grid(Z ~ algo) +
  geom_point(aes(x = !!sym(genes[4]), y = !!sym(gene_y))) +
  geom_point(aes(x = !!sym(genes[4]), y = y_pred, col = algo))
