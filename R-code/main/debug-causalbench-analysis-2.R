source("main/dependencies.R")

# Import data
dat <- read_csv(
  "../results/causalbench-analysis-2/n_preds_3-confounders_False/20240802-141916/_debug/debug-response_ENSG00000083845-run_id_26-iter_id_0.csv"
)

genes <- colnames(dat)[1:4]


# Constants for gene column names
gene_x1 <- genes[2]
gene_y <- genes[1]


# Look how many training samples are non-observational
dat2plot <- dat %>% filter(algo == "BCF") %>% 
  filter(Z %in% c("non-targeting", genes[3])) %>% 
  filter(env %in% c("train", "test"), interv_strength %in% c(0, 0.9))

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(genes[3]), y = !!sym(gene_y), col = env), 
             alpha = 0.5) +
  coord_cartesian(xlim = c(0, 5))
