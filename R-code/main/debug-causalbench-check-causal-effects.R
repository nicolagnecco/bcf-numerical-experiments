source("main/dependencies.R")

dat <- read_csv("../data/processed/genes_no_zeros_causal_effect_median.csv")

dat$Z %>% unique() %>% length()


genes <- colnames(dat)[-ncol(dat)]
n_genes <- length(genes)

M <- matrix(0, ncol = n_genes, nrow = n_genes)

for (i in seq_along(genes)){
  for (j in seq_along(genes)){
    if (i !=j) {
      dat_obs <- (dat %>% filter(Z == "non-targeting"))[, j][[1]]
      dat_int <- (dat %>% filter(Z == genes[i]))[, j][[1]]
      
      M[i, j] <- abs(median(dat_obs) - median(dat_int))
    }
    
  }
}


M %>% View()

gene_1 <- genes[10]
gene_2 <- genes[8]


dat2plot <- dat %>% filter(Z %in% c("non-targeting", gene_1))

(dat2plot %>% filter(Z == "non-targeting") %>% select(!!sym(gene_2)))[[1]] %>% 
  hist()

(dat2plot %>% filter(Z != "non-targeting") %>% select(!!sym(gene_2)))[[1]] %>% 
  hist()

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_1), y = !!sym(gene_2), col = Z))

# ?: do scale gene expr in Causalbench?
# Per experiments they normalize them
# Within each experiment, 
# Send pairs
# -> maybe not bad. Ask Matthieu Chevalley
