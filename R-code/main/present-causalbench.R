source("main/dependencies.R")
library(GGally)
library(ggpubr)

set.seed(1321)

# show genes
dat <- read_csv("../data/processed/genes.csv")


selected_genes <- colnames(dat)[1:3]

dat2plot <- dat %>% 
  filter(Z %in% c("non-targeting", selected_genes)) %>% 
  select(all_of(c(selected_genes, "Z"))) %>% 
  mutate(Z = if_else(Z == "non-targeting", "Observational", Z))

dat2plot <- bind_rows(
  dat2plot %>% filter(Z != "Observational"),
  dat2plot %>% filter(Z == "Observational") %>% slice_sample(n=250)
)

gg <- ggplot(dat2plot) + 
  geom_point(aes(x = ENSG00000122406, y = ENSG00000144713, col = Z),
             alpha = .5) +
  guides(color = guide_legend(title = "Setting")); gg


save_myplot(plt = gg, plt_nm = "../results/figures/gene-pres.pdf",
            width = 2.5, height = 2.5)
  

# show interventional strength
dat <- read_csv(
  "../results/discuss-genes/confounders_False-npreds_3/20240723-180311/causalbench-data_ENSG00000063177.csv"
) %>% 
  mutate(Z = if_else(Z == "non-targeting", "Observational", Z))

genes <- colnames(dat)[1:6]
gene_x1 <- genes[2]
gene_y <- genes[1]

dat2plot <- dat %>% filter(environment == 0, n_env_top == 20, algorithm=="BCF")
gg0 <- ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y), col = Z), alpha = 0.5) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y)), col = "red",
             data = dat2plot %>% filter(Z != "Observational", set == "train")) +
  theme(legend.position = "none"); gg0

dat2plot <- dat %>% filter(environment == 0, n_env_top == 100, algorithm=="BCF")
gg1 <- ggplot(dat2plot) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y), col = Z), alpha = 0.5) +
  geom_point(aes(x = !!sym(gene_x1), y = !!sym(gene_y)), col = "red",
             data = dat2plot %>% filter(Z != "Observational", set == "train")) +
  theme(legend.position = "none"); gg1

legend <- get_legend(
  # create some space to the left of the legend
  gg0 +
    guides(color = guide_legend(title = "Setting", nrow=1)) +
    theme(legend.position = "bottom")
)

gg3 <- ggarrange(gg0, gg1,
                nrow = 1, ncol = 2, align = "hv",
                legend = "bottom", legend.grob = legend)

gg3
save_myplot(gg3, "../results/figures/genes-interv-strength.pdf", height = 3, width = 6)



# show confounder
dat <- bind_rows(
  read_csv(
    "../results/discuss-genes/confounders_False-npreds_3/20240723-180311/causalbench-data_ENSG00000063177.csv"
  ) %>% 
    mutate(Z = if_else(Z == "non-targeting", "Observational", Z), confounder = FALSE),
  read_csv(
    "../results/discuss-genes/confounders_True-npreds_3/20240723-174036/causalbench-data_ENSG00000063177.csv"
  ) %>% 
    mutate(Z = if_else(Z == "non-targeting", "Observational", Z), confounder = TRUE)
) %>% 
  mutate(confounder = texify_column(confounder, "confounder"))


gg_conf <- ggplot(dat %>% filter(environment == 0, n_env_top == 20, algorithm=="BCF")) +
  facet_grid(~confounder, labeller = label_parsed) +
  geom_point(aes(x = ENSG00000105372, y = ENSG00000063177, col = Z), alpha=.5) +
  guides(color = guide_legend(title = "Setting")) +
  theme(legend.position = "bottom")
  
save_myplot(gg_conf, "../results/figures/genes-conf.pdf", height = 2, width = 2)

