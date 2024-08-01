source("main/dependencies.R")

res <- read_csv("../results/causalbench-analysis/n_preds_3-n_trainenv_3/20240801-230614/causalbench-res.csv") %>%
  mutate(algorithm = refactor_methods(algorithm, rev=TRUE)) 


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


ggplot(res) +
  geom_boxplot(aes(x = factor(interv_strength), y = mse, col = algorithm)) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))


ggplot() +
  geom_line(data=dat2plot_agg, mapping=aes(x = interv_strength, y = mse, col = algorithm),
            size=2) +
  # geom_line(data=dat2plot, mapping=aes(x = interv_strength, y = mse, col = algorithm,
                                       # group = interaction(algorithm, response)),
            # alpha = .2) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))
