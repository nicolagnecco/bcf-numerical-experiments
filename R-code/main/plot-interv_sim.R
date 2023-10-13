source("main/dependencies.R")

# Constant definitions
P_EFF <- 5
GAMMA_NORM <- 2
CSVFILE <- "../results/output_data/experiment_1-bcf.csv"
FILENAME <- glue::glue("../results/figures/rank_vs_p_eff-gamma={GAMMA_NORM}.pdf")

# Main 
df <- read_csv(CSVFILE) %>% 
  filter(`_run_seq` == 1)

df_2 <- df %>% 
   filter(gamma_norm == GAMMA_NORM) %>% 
   filter(p_effective == 3) %>% 
   filter(q %in% c(1, 5, 7, 10)) %>% 
  filter(method_names %in% c("BCF", "OLS", "ConstFunc", "Causal","IMP"))

df_t <- df_2 %>% 
  group_by(method_names, r, q, p, p_effective, gamma_norm, interv_strength) %>% 
  summarise(MSE = mean(MSE)) %>% 
  mutate(method_names = refactor_methods(method_names, rev=TRUE),
         rank = texify_column(q, "q"),
         p = texify_column(p, "p"),
         p_eff = texify_column(p_effective, "p_{eff}"),
         gamma_norm = texify_column(gamma_norm, "c")) 

gg <- ggplot(df_t %>% arrange(method_names)) +
  facet_grid(p ~ rank, 
             scales = "free_y", labeller = label_parsed) +
  geom_line(mapping = aes(x = interv_strength, y = MSE, col=method_names,
            size = method_names, linetype = method_names),
            alpha = 0.75) +
  geom_point(mapping = aes(x = interv_strength, y = MSE, col = method_names,
             shape = method_names), fill = "white", size = 2,
             stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  theme(legend.position = c(0.5, -0.35), legend.direction = "horizontal") +
  xlab("Perturbation strength") +
  labs(colour="Methods", shape="Methods", size="Methods", linetype="Methods"); gg

 save_myplot(plt = gg, plt_nm = FILENAME, width = 1.5, height = 1.5)
