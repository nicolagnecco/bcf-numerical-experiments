source("main/dependencies.R")

# Constant definitions
MSEFILE1 <- "../python/outputs/exp-identifiability/nonlin_g_True-inst_disc_False-n_5000/20251014_204549/mses.csv"
MSEFILE1small <- "../python/outputs/exp-identifiability/nonlin_g_True-inst_disc_False-n_1000/20251014_210846/mses.csv"
MSEFILE2 <- "../python/outputs/exp-identifiability/nonlin_g_False-inst_disc_True-n_5000/20251015_094012/mses.csv"
MSEFILE2small <- "../python/outputs/exp-identifiability/nonlin_g_False-inst_disc_True-n_1000/20251015_093636/mses.csv"
CSVFILE <- "../python/outputs/exp-identifiability/nonlin_g_True-inst_disc_False-n_5000/20251014_204549/predictions.csv"
CSVFILEsmall <- "../python/outputs/exp-identifiability/nonlin_g_True-inst_disc_False-n_1000/20251014_210846/predictions.csv"
FILENAME1 <- glue::glue("../results/figures/app-identifiability-mse-1.pdf")
FILENAME2 <- glue::glue("../results/figures/app-identifiability-mse-2.pdf")


dat <- read_csv(CSVFILEsmall)

ggplot(dat %>% filter(model == "BCF", rep_id==0)) +
  geom_point(aes(x = X1, y = y)) +
  geom_point(aes(x = X1, y = y_hat), alpha = 0.5, col = "tomato")

## 1) Define your mapping in one place
methods_map <- tibble::tibble(
  code  = c('BCF', 'CF', 'OLS','IMP','Causal'),
  label = c('BCF', 'CF', 'LS','BCF-oracle','Structural'),
  color = c(
    my_palette$c3,             # BCF
    my_palette$c2,             # CF
    my_palette$c1,             # OLS
    my_palette$blue2,       # IMP
    my_palette$darkgrey       # Causal
  ),
  shape = c(21, 21, 21, 17, 17)
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
mses_nonlinear_g <- bind_rows(
  read_csv(MSEFILE1) %>% mutate(n = 5000),
  read_csv(MSEFILE1small) %>% mutate(n = 1000),
) %>% 
  # filter(rep_id < 2) %>%
  group_by(model, int_par, n) %>% 
  summarise(mse = mean(test_mse),
            max_mse = quantile(test_mse, p=1),
            min_mse = quantile(test_mse, p=0))

mses_linear_g <- bind_rows(
  read_csv(MSEFILE2) %>% mutate(n = 5000),
  read_csv(MSEFILE2small) %>% mutate(n = 1000),
) %>% 
  # filter(rep_id == 2) %>% 
  group_by(model, int_par, n) %>% 
  summarise(mse = mean(test_mse),
            max_mse = quantile(test_mse, p=1),
            min_mse = quantile(test_mse, p=0))

mses <- bind_rows(
  mses_nonlinear_g %>% mutate(assumption = "Assumption 4 (diff.ble g and f)"),
  mses_linear_g %>% mutate(assumption = "Assumption 3 (lin. g and discr. Z)")
)

mses2 <- mses %>%
  filter(model %in% breaks_vec) %>%
  mutate(model_ = factor(model, levels = breaks_vec)) %>% 
  mutate(n = paste0("n = ", n)) %>% 
  filter(model_ != "CFx")

## 4) Plot (color + shape share the same legend)
gg <- ggplot(mses2 %>% arrange(desc(model_)),
             aes(x = int_par, y = mse, color = model_, shape = model_)) +
  facet_grid(assumption ~ n, scales = "free_y") +
  # geom_ribbon(aes(x = int_par,ymin=min_mse,ymax=max_mse, fill = model_), alpha = .25) +
  geom_line() +
  geom_point(fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = col_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  scale_fill_manual(values = col_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  scale_shape_manual(values = shape_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  labs(x = "Perturbation Strength", y = "MSE"); gg


save_myplot(plt = gg, plt_nm = FILENAME1, width = 2.5, height = 2.5)
