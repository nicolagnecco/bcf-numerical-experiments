source("main/dependencies.R")

"../results/causalbench-analysis-2/n_preds_3-train_mask_top-confounders_False/20240808-104520//"
base_folder <- "../results/causalbench-analysis-2"
experiment <- "n_preds_3-train_mask_top-confounders_False"
run <- "20240808-104520"
res_name <- "causalbench-res.csv"

filename <- file.path(base_folder, experiment, run, res_name)
imagename <- file.path(base_folder, experiment, run, "res.pdf")

res <- read_csv(
  filename
) %>%
  mutate(algorithm = refactor_methods(algorithm, rev=TRUE)) 

res %>% select(response, predictors, training_envs, confounders, test_envs) %>% unique() %>% view()

ggplot(res) +
  geom_boxplot(aes(x = factor(interv_strength), y = mse, col = algorithm)) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))


# for each response, intervention strength, and iter_id,
# take maximum over training environments, confounders, and, most importantly,
# over direction of perturbations
dat2plot <- res %>% 
  group_by(algorithm, response, training_envs, iter_id, interv_strength) %>% 
  summarise(mse = max(mse))

dat2plot_agg <- dat2plot %>% 
  ungroup() %>% 
  group_by(algorithm, interv_strength) %>% 
  summarise(mse = mean(mse))


gg <- ggplot(data=dat2plot_agg %>% filter(algorithm == algorithm)) +
  geom_line(mapping=aes(x = interv_strength, y = mse, 
                        col = algorithm, size = algorithm, linetype=algorithm))+
  geom_point(aes(x = interv_strength, y = mse, col = algorithm, shape=algorithm), 
             fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)); gg

save_myplot(gg, imagename,
            width = 4, height = 4)
