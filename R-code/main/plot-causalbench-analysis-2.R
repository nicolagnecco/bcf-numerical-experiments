source("main/dependencies.R")

res <- read_csv("../results/causalbench-analysis-2/n_preds_3-n_trainenv_1/20240802-093448///causalbench-res.csv") %>% 
  mutate(algorithm = refactor_methods(algorithm, rev=TRUE)) 


res %>% select(response, predictors, training_envs, confounders, test_envs) %>% unique() %>% view(
)

ggplot(res) +
  geom_boxplot(aes(x = factor(interv_strength), y = mse, col = algorithm)) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))


# for each response, and intervention strength, 
# take maximum over training environments, confounders, and, most importantly,
# over direction of perturbations
dat2plot <- res %>% 
  group_by(algorithm, response, interv_strength) %>% 
  summarise(mse = max(mse))

dat2plot_agg <- dat2plot %>% 
  ungroup() %>% 
  group_by(algorithm, interv_strength) %>% 
  summarise(mse = mean(mse))

ggplot(data=dat2plot_agg) +
  geom_line(mapping=aes(x = interv_strength, y = mse, 
                        col = algorithm, size = algorithm, linetype=algorithm))+
  # geom_line(data=dat2plot, mapping=aes(x = interv_strength, y = mse, col = algorithm,
  # group = interaction(algorithm, response)),
  # alpha = .2) +
  geom_point(aes(x = interv_strength, y = mse, col = algorithm, shape=algorithm), 
             fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) 

