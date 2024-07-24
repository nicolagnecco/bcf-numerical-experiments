source("main/dependencies.R")

# Function to read and mutate each CSV file
read_and_mutate <- function(file_path) {
  # Extract run ID from the file path
  run_id <- str_extract(file_path, "[^/]+(?=/causalbench-res.csv)")
  
  # Extract confounder and npreds from the file path
  confounder <- str_extract(file_path, "(?<=confounders_)[^_-]+")
  npreds <- str_extract(file_path, "(?<=npreds_)[^/]+")
  
  # Convert confounder to logical
  confounder <- ifelse(confounder == "True", TRUE, FALSE)
  
  # Convert npreds to integer
  npreds <- as.integer(npreds)
  
  # Read the CSV file
  df <- read_csv(file_path)
  
  # Mutate to add the run ID, confounder, and npreds
  df <- df %>%
    mutate(run = run_id,
           confounder = confounder,
           npreds = npreds)
  
  return(df)
}


# List all CSV files in the specified pattern
file_paths_confounded <- list.files(
  path = "../results/discuss-genes/confounders_True-npreds_3/",
  pattern = "causalbench-res.csv",
  full.names = TRUE,
  recursive = TRUE)

file_paths_unconfounded <- list.files(
  path = "../results/discuss-genes/confounders_False-npreds_3/",
  pattern = "causalbench-res.csv",
  full.names = TRUE,
  recursive = TRUE)

res <- bind_rows(
  map_dfr(file_paths_unconfounded, read_and_mutate),
  map_dfr(file_paths_confounded, read_and_mutate),
) %>% 
  # filter(is.na(M_0) | M_0 != "[[0.]\n [0.]\n [0.]]") %>%
  filter(env_id == 0) %>% 
  mutate(algorithm = refactor_methods(algorithm, rev=TRUE)) %>% 
  filter(algorithm != "xxx") %>% 
  mutate(confounder = texify_column(confounder, "confounder"))

ggplot(res) +
  facet_grid(~confounder) +
  geom_boxplot(aes(x = factor(-n_env_obs), 
                   y = mse_test, 
                   col = factor(algorithm)), outlier.shape = 21) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE))

dat2plot <- res %>% 
  group_by(algorithm, n_env_obs, confounder) %>% 
  summarise(test = mean(mse_test),
            train = mean(mse_train)) %>% 
  pivot_longer(cols = c(train, test), names_to = "setting", values_to = "mse") %>% 
  filter(setting == "test")

dat2plot_byrun <- res %>% 
  group_by(algorithm, n_env_obs, confounder, run) %>% 
  summarise(test = mean(mse_test),
            train = mean(mse_train)) %>% 
  pivot_longer(cols = c(train, test), names_to = "setting", values_to = "mse") %>% 
  filter(setting == "test")

gg_test <- ggplot(dat2plot) +
  facet_grid(setting ~ confounder, scales = "free_y",  labeller = label_parsed) + 
  geom_line(data = dat2plot_byrun,
            aes(x =  -n_env_obs, y = mse, col = algorithm,
                group = interaction(algorithm, run)), alpha = .25) +
  geom_line(aes(x = -n_env_obs, y = mse, col = algorithm, size = algorithm, 
                linetype=algorithm), alpha = .75) +
  geom_point(aes(x = -n_env_obs, y = mse, col = algorithm, shape=algorithm), 
             fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  # theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  xlab("Perturbation strength") +
  ylab("MSE") +
  theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  coord_cartesian(ylim = c(0.05, .35)) +
  labs(colour="Methods:", shape="Methods:", size="Methods:", linetype="Methods:"); gg_test

save_myplot(plt = gg_test, plt_nm = "../results/figures/genes.pdf", 
            width = 1.75, height = 1.75)


dat2plot <- res %>% 
  group_by(algorithm, n_env_obs, confounder) %>% 
  summarise(test = mean(mse_test),
            train = mean(mse_train)) %>% 
  pivot_longer(cols = c(train, test), names_to = "setting", values_to = "mse") %>% 
  filter(setting == "train")

dat2plot_byrun <- res %>% 
  group_by(algorithm, n_env_obs, confounder, run) %>% 
  summarise(test = mean(mse_test),
            train = mean(mse_train)) %>% 
  pivot_longer(cols = c(train, test), names_to = "setting", values_to = "mse") %>% 
  filter(setting == "train")

gg_train <- ggplot(dat2plot) +
  facet_grid(setting ~ confounder, scales = "free_y",  labeller = label_parsed) + 
  geom_line(data = dat2plot_byrun,
            aes(x =  -n_env_obs, y = mse, col = algorithm,
                group = interaction(algorithm, run)), alpha = .25) +
  geom_line(aes(x = -n_env_obs, y = mse, col = algorithm, size = algorithm, 
                linetype=algorithm), alpha = .75) +
  geom_point(aes(x = -n_env_obs, y = mse, col = algorithm, shape=algorithm), 
             fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  # theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  xlab("Perturbation strength") +
  ylab("MSE") +
  theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  coord_cartesian(ylim = c(0.0, .1)) +
  labs(colour="Methods:", shape="Methods:", size="Methods:", linetype="Methods:"); gg_train


legend <- get_legend(
  # create some space to the left of the legend
  gg_test +
    guides(color = guide_legend(title = "Methods:", nrow=1),
           shape = guide_legend(title = "Methods:", nrow=1),
           size = guide_legend(title = "Methods:", nrow=1),
           linetype = guide_legend(title = "Methods:", nrow=1)) +
    theme(legend.position = "bottom")
)


gg_mses <- ggarrange(gg_test, gg_train,
                     nrow = 2, ncol = 1, align = "hv",
                     legend = "bottom", legend.grob = legend)

save_myplot(gg_mses, "../results/figures/genes-mses.pdf", height = 6, width = 5)
