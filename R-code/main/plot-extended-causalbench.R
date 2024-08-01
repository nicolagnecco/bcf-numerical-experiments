source("main/dependencies.R")

res <- read_csv("../results/try-extended/n_preds_2-n_conf_2-n_trainenv_2/20240726-093011//causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_1-n_conf_1-n_trainenv_1/20240726-092059/causalbench-res.csv")

# problems when null space is estimated wrongly
res <- read_csv("../results/try-extended/n_preds_2-n_conf_0-n_trainenv_1/20240726-100937/causalbench-res.csv") %>% 
  filter(run_id == 15)

res <- read_csv("../results/try-extended/n_preds_1-n_conf_0-n_trainenv_1/20240726-103742//causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_2-n_conf_0-n_trainenv_2/20240726-092618/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_3-n_conf_0-n_trainenv_3/20240726-102337/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-105124/causalbench-res.csv")
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-115335/causalbench-res.csv")

# imp = 0 
res <- read_csv("../results/try-extended/n_preds_10-n_conf_0-n_trainenv_10/20240726-183930/causalbench-res.csv")


# try all_genes
res <- read_csv("../results/try-extended/n_preds_27-n_conf_0-n_trainenv_27/20240731-134803/causalbench-res.csv")


# try increasing strength
res <- read_csv("../results/try-extended/n_preds_3-n_conf_1-n_trainenv_1/20240801-160504/causalbench-res.csv") %>% 
  mutate(algorithm = refactor_methods(algorithm, rev=TRUE)) 


ggplot(res) +
  # facet_grid(~confounder) +
  geom_boxplot(aes(x = factor(interv_strength), 
                   y = mse, 
                   col = factor(algorithm)), outlier.shape = 21) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))

dat2plot <- res %>% 
  group_by(algorithm, interv_strength) %>% 
  summarise(mse = mean(mse))

dat2plot_byrun <- res %>% 
  group_by(algorithm, interv_strength, run_id) %>% 
  summarise(mse = mean(mse))

gg_test <- ggplot(dat2plot) +
  # facet_grid(setting ~ confounder, scales = "free_y",  labeller = label_parsed) + 
  # geom_line(data = dat2plot_byrun,
  #           aes(x =  interv_strength, y = mse, col = algorithm,
  #               group = interaction(algorithm, run)), alpha = .25) +
  geom_line(aes(x = interv_strength, y = mse, col = algorithm, size = algorithm, 
                linetype=algorithm), alpha = .75) +
  geom_point(aes(x = interv_strength, y = mse, col = algorithm, shape=algorithm), 
             fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  # theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  xlab("Perturbation strength") +
  ylab("MSE") +
  # theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  # coord_cartesian(ylim = c(0.05, .35)) +
  labs(colour="Methods:", shape="Methods:", size="Methods:", linetype="Methods:"); gg_test


sum_res <- res %>% 
  # filter(test_envs == "train") %>%
  filter(algorithm %in% c("BCF", "ConstFunc", "OLS")) %>%
  group_by(algorithm, response, training_envs) %>% 
  summarise(mse = max(mse))

sum_res %>% 
  group_by(algorithm) %>% 
  summarise(mean(mse))

ggplot(sum_res, aes(x = algorithm, y = mse)) +
  geom_line(aes(group=interaction(response, training_envs)), alpha = .1) +
  geom_boxplot() +
  stat_summary(geom = "errorbar", 
               fun.min = mean, 
               fun = mean, 
               fun.max = mean, width = .75, col = "red") 



# debug test_envs = "ENSG00000151849"
# training environments mixed: not really good
# try with training environments != predictors and test environments predictors
# try with training environments == predictors and test environments
# -> try with confounders
# -> try with fewer training environments