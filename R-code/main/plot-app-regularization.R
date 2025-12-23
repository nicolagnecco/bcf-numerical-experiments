source("main/dependencies.R")

# Constant definitions
MSEFILE <- "../results/output_data/exp-regularization/20250913_164707-baseline/mses.csv"
MSEFILE1 <- "../results/output_data/exp-regularization/20250914_124752-baseline/mses.csv"
MSEFILE2 <- "../results/output_data/exp-regularization/20250914_134013-baseline/mses.csv"
FILENAME <- glue::glue("../results/figures/app-regularization-mse.pdf")



## 1) Define your mapping in one place
methods_map <- tibble::tibble(
  code  = c('BCF-MLP','OLS-MLP', 'OLS-MLP2-wd25', 'OLS-MLP2-wd1','IMP','Causal',
            'CF-MLP','CF-MLP-2','CF-MLP-3','CF-MLP-4'),
  label = c('BCF-MLP','LS', 'LS2-wd25', 'LS2-wd1', 'IMP','Structural',
            TeX("$\\lambda=2.5\\times 10^{-3}$"),
            TeX("$\\lambda=2.5\\times 10^{-2}$"),
            TeX("$\\lambda=2.5\\times 10^{-1}$"),
            TeX("$\\lambda=2.5\\times 10^{0}$")),
  color = c(
    my_palette$c3,             # BCF-MLP
    my_palette$c1,             # OLS-MLP
    "red",             # OLS-MLP
    "green",             # OLS-MLP
    "black", #my_palette$darkgrey,       # IMP
    "black", #my_palette$darkgrey,       # Causal
    colorRampPalette(c("#1D976C","#0F5A40"))(4)  # the 4 lambda curves
  ),
  shape = c(21, 21, 21, 21, NA, NA, 21, 21, 21, 21), # NA was 17 before
  linetype = c("solid", "solid", "solid", "solid",
               "dashed", "dotted", "solid", "solid", "solid", "solid"),
  size = c(.5, .5, .5, .5, .35, .35, .5, .5, .5, .5)
)

## If you want the legend shown top-to-bottom as written above,
## keep this order. If you want reversed, do:
# methods_map <- methods_map %>% dplyr::arrange(dplyr::desc(row_number()))

## 2) Build named vectors for scales
lab_vec   <- methods_map$label; names(lab_vec)   <- methods_map$code
col_vec   <- methods_map$color; names(col_vec)   <- methods_map$code
shape_vec <- methods_map$shape; names(shape_vec) <- methods_map$code
linetype_vec <- methods_map$linetype; names(linetype_vec) <- methods_map$code
size_vec <- methods_map$size; names(size_vec) <- methods_map$code
breaks_vec <- methods_map$code                   # legend/order anchor

## 3) Prep data with factor levels matching your mapping
mses <- bind_rows(
  read_csv(MSEFILE),
  read_csv(MSEFILE1),
  read_csv(MSEFILE2)
) %>% 
  group_by(model, int_par) %>% 
  summarise(mse = mean(test_mse))

mses2 <- mses %>%
  filter(model %in% breaks_vec) %>%
  mutate(model_ = factor(model, levels = breaks_vec))

## 4) Plot (color + shape share the same legend)
gg <- ggplot(mses2 %>% arrange(desc(model_)),
             aes(x = int_par, y = mse, 
                 color = model_, 
                 shape = model_,
                 linetype = model_,
                 linewidth=model_)) +
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
  scale_linetype_manual(values = linetype_vec,
                     breaks = breaks_vec,
                     labels = lab_vec,
                     name   = "Methods") +
  scale_linewidth_manual(values = size_vec,
                        breaks = breaks_vec,
                        labels = lab_vec,
                        name   = "Methods") +
  labs(x = "Perturbation Strength", y = "MSE"); gg

save_myplot(plt = gg, plt_nm = FILENAME, width = 2.5, height = 2.5)
