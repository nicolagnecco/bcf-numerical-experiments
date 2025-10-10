source("main/dependencies.R")

# Constant definitions
P_EFF <- 5
GAMMA_NORM <- 2
CSVFILE <- "../python/outputs/exp-regularization/20250913_164707-baseline/predictions.csv"
CSVFILE1 <- "../python/outputs/exp-regularization/20250914_124752-baseline/predictions.csv"
CSVFILE2 <- "../python/outputs/exp-regularization/20250914_134013-baseline/predictions.csv"
MSEFILE <- "../python/outputs/exp-regularization/20250913_164707-baseline/mses.csv"
MSEFILE1 <- "../python/outputs/exp-regularization/20250914_124752-baseline/mses.csv"
MSEFILE2 <- "../python/outputs/exp-regularization/20250914_134013-baseline/mses.csv"
FILENAME <- glue::glue("../results/figures/app-regularization-mse.pdf")



## 1) Define your mapping in one place
methods_map <- tibble::tibble(
  code  = c('BCF-MLP','OLS-MLP','IMP','Causal','CF-MLP','CF-MLP-2','CF-MLP-3','CF-MLP-4'),
  label = c('BCF','LS','IMP','Structural',
            TeX("$\\lambda=2.5\\times 10^{-3}$"),
            TeX("$\\lambda=2.5\\times 10^{-2}$"),
            TeX("$\\lambda=2.5\\times 10^{-1}$"),
            TeX("$\\lambda=2.5\\times 10^{0}$")),
  color = c(
    my_palette$c3,             # BCF-MLP
    my_palette$c1,             # OLS-MLP
    my_palette$darkgrey,       # IMP
    my_palette$darkgrey,       # Causal
    colorRampPalette(c("#1D976C","#0F5A40"))(4)  # the 4 lambda curves
  ),
  shape = c(21, 21, 17, 17, 21, 21, 21, 21)
)

## If you want the legend shown top-to-bottom as written above,
## keep this order. If you want reversed, do:
# methods_map <- methods_map %>% dplyr::arrange(dplyr::desc(row_number()))

## 2) Build named vectors for scales
lab_vec   <- methods_map$label; names(lab_vec)   <- methods_map$code
col_vec   <- methods_map$color; names(col_vec)   <- methods_map$code
shape_vec <- methods_map$shape; names(shape_vec) <- methods_map$code
breaks_vec <- methods_map$code                   # legend/order anchor

## 3) Prep data with factor levels matching your mapping
mses <- bind_rows(
  read_csv(MSEFILE),
  read_csv(MSEFILE1),
  read_csv(MSEFILE2)
) %>% 
  filter(rep_id == 2) %>% 
  group_by(model, int_par) %>% 
  summarise(mse = mean(test_mse))

mses2 <- mses %>%
  filter(model %in% breaks_vec) %>%
  mutate(model_ = factor(model, levels = breaks_vec))

## 4) Plot (color + shape share the same legend)
gg <- ggplot(mses2 %>% arrange(desc(model_)),
             aes(x = int_par, y = mse, color = model_, shape = model_)) +
  geom_line() +
  geom_point(fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = col_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  scale_shape_manual(values = shape_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  labs(x = "Perturbation Strength", y = "MSE")

gg

save_myplot(plt = gg, plt_nm = FILENAME, width = 2.5, height = 2.5)

# Predictions 
CSVFILE4 <- "../python/outputs/exp-regularization/20250914_182121-baseline/predictions.csv"
df <- bind_rows(
  read_csv(CSVFILE4),
  # read_csv(CSVFILE1),
  # read_csv(CSVFILE2)
) %>% 
  filter(model %in% c("BCF-RF"))

df2plot <- df %>% filter(rep_id == 3, env == env)

ggplot() +
  geom_point(data = df2plot %>% filter(model=="BCF-RF"),
             aes(x = X1, y = y), alpha = .1) +
  geom_point(data = df2plot %>% filter(model %in% c( 
    "BCF-RF"
  )),
  aes(x = X1, y = y_hat, col = model), alpha = .25)


# other mse
MSEFILE4 <- "../python/outputs/exp-regularization/20250914_182121-baseline/mses.csv"
mses <- bind_rows(
  read_csv(MSEFILE4),
  read_csv(MSEFILE) %>% filter(model != "BCF-RF")
) %>% 
  filter(rep_id == rep_id) %>% 
  group_by(model, int_par) %>% 
  summarise(mse = mean(test_mse)) %>% 
  filter(model %in% c("Causal", "BCF-MLP", "BCF-RF", "IMP", "OLS-MLP"))

ggplot(mses) +
  geom_line(aes(x = int_par, y = mse, col = model))

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
