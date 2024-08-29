source("main/dependencies.R")

dat <- read_csv("../data/processed/genes.csv")
dim(dat)

# let x = count
# y = x / sum_{1, ..., 1000}(x)
# z = log(y + 1)
# y = exp(z) - 1
# sum_{1, ..., 28}(y) = sum_{1, ..., 28}(x) / sum_{1, ..., 1000}(x) < 1?

# normalization removes batch effect. 
# When doing PCA with raw count, the first two PCs are batch effect

row_sums <- rowSums(exp(dat[, -29]) - 1)
length(row_sums)
hist(row_sums)

dat_trans <- (exp(dat[, -29])-1) %>% as_tibble() %>% 
  mutate(Z = dat$Z)

training_rows <- sample(which(dat$Z == "non-targeting"), 1000)
test_rows <- which(dat$Z == "ENSG00000122406")

write_csv(dat_trans, "../data/processed/genes_exp.csv")

ggplot(dat[c(training_rows, test_rows), ]) +
  geom_point(aes(x = ENSG00000122406, y = ENSG00000144713, col = Z), alpha = .5)

ggplot(dat_trans[c(training_rows, test_rows), ]) +
  geom_point(aes(x = ENSG00000122406, y = ENSG00000144713, col = Z), alpha = .5)



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

gene_1 <- genes[7]
gene_2 <- genes[5]


dat2plot <- dat %>% filter(Z %in% c("non-targeting", gene_1))

(dat2plot %>% filter(Z == "non-targeting") %>% select(!!sym(gene_2)))[[1]] %>% 
  hist()

(dat2plot %>% filter(Z != "non-targeting") %>% select(!!sym(gene_2)))[[1]] %>% 
  hist()

ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_1), y = !!sym(gene_2), col = Z))


dat2analyze <- dat %>% filter(Z %in% c("non-targeting"))
summarise(dat2analyze %>% select(-Z) %>% as.matrix())




# ?: do scale gene expr in Causalbench?
# Per experiments they normalize them
# Within each experiment, 
# Send pairs
# -> maybe not bad. Ask Matthieu Chevalley
