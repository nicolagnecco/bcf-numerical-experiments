source("main/dependencies.R")

# Constant definitions
FILENAME1 <- glue::glue("../results/figures/app-robustness-2.pdf")

root_folder <- "../results/output_data/exp-robustness/20260129_231908/"
root_folder2 <- "../results/output_data/exp-robustness/20260129_230342/"

files_mse <- 
  c(list.files(root_folder, pattern="mses\\.csv", recursive = TRUE, 
               full.names=TRUE),
    list.files(root_folder2, pattern="mses\\.csv", recursive = TRUE, 
               full.names=TRUE))

files_mse <- 
  c(list.files(root_folder, pattern="mses\\.csv", recursive = TRUE, 
               full.names=TRUE))

import_mse_file <- function(file_name){
  my_string <- str_match(file_name, 
                         "n_(\\d+)-inst_str_([0-9.]+)-rep_id_(\\d+)")
  n_train <- as.integer(my_string[,2])
  instr_str <- as.numeric(my_string[,3])
  rep_id_ <- as.integer(my_string[, 4])
  
  read_csv(file_name) %>% 
    mutate(n = n_train, instr_str = instr_str, rep_id = rep_id_)
  
}


dat <- map_dfr(files_mse, import_mse_file)


cols_mlp <- colorRampPalette(c("#1D976C","#0F5A40"))(3) 

## 1) Define your mapping in one place
methods_map <- tibble::tibble(
  code  = c('BCF', 'BCF-MLP-large', 'OLS', 'IMP','Causal'),
  label = c('BCF-XGB', 'BCF-MLP', 'OLS', 'IMP', 'Structural'),
  color = c(
    my_palette$c3,             # BCF
    "#479FF8",             
    my_palette$yellow, # OLS
    "black", #my_palette$blue2,       # IMP
    "black" #my_palette$darkgrey       # Causal
  ),
  shape = c(21, 21, 21, NA, NA),
  linetype = c("solid", "solid", "solid", "dashed", "dotted")
)

## If you want the legend shown top-to-bottom as written above,
## keep this order. If you want reversed, do:
# methods_map <- methods_map %>% dplyr::arrange(dplyr::desc(row_number()))

## 2) Build named vectors for scales
lab_vec   <- methods_map$label; names(lab_vec)   <- methods_map$code
col_vec   <- methods_map$color; names(col_vec)   <- methods_map$code
shape_vec <- methods_map$shape; names(shape_vec) <- methods_map$code
linetype_vec <- methods_map$linetype; names(linetype_vec) <- methods_map$code
breaks_vec <- methods_map$code                   # legend/order anchor


## 3) Prep data with factor levels matching your mapping
mses <- dat %>% 
  # filter(rep_id == 1) %>%
  # filter(!(model %in% c("OLS"))) %>% 
  filter(model != "OLS") %>% 
  mutate(model = if_else(model=="OLSMLP", "OLS", model)) %>% 
  arrange(n) %>% 
  filter(n != 1500) %>% 
  mutate(n = texify_column(n, "n"))

mses1 <- mses %>% 
  group_by(model, int_par, n, instr_str) %>% 
  summarise(mse = mean(test_mse),
            max_mse = quantile(test_mse, p=1.00),
            min_mse = quantile(test_mse, p=0.00)) %>% 
  mutate(max_mse = if_else(model %in% c("Causal", "IMP"), mse, max_mse),
         min_mse = if_else(model %in% c("Causal", "IMP"), mse, min_mse))

mses2 <- mses1 %>%
  filter(model %in% breaks_vec) %>%
  mutate(model_ = factor(model, levels = breaks_vec)) 

# Optional: data to plot individual runs
mses3 <-  mses %>%
  filter(model %in% breaks_vec) %>%
  mutate(model_ = factor(model, levels = breaks_vec)) 



## 4) Plot (color + shape share the same legend)
gg <- ggplot(
  mses2 %>% arrange(desc(model_)) %>% 
    filter(model_ %in% c("BCF", "BCF-MLP-large", "OLS")) %>% 
    filter(n == (mses$n %>% unique())[3]),
  aes(x = int_par, y = mse, color = model_, fill = model_,
      shape = model_, linetype = model_)
) +
  # facet_grid(~ n, scales = "fixed", labeller = label_parsed) +
  geom_line(
    data = mses2 %>%
      arrange(desc(model_)) %>%
      filter(model_ %in% c("IMP")), size=.35
  ) +
  geom_ribbon(aes(ymin = min_mse, ymax = max_mse),
              alpha = .125, size=.2) +  # no edge line
  # geom_line(data=mses3 %>% arrange(desc(model_)) %>%
  #             filter(model_ %in% c("BCF", "BCF-MLP-large")),
  #           aes(x = int_par, y = test_mse, color = model_,
  #               group = interaction(model, rep_id)),
  #           size = 0.15) +
  geom_line() +
  geom_point(fill = "white", size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = col_vec, breaks = breaks_vec, labels = lab_vec) +
  scale_fill_manual(values = col_vec, breaks = breaks_vec, labels = lab_vec) +
  scale_shape_manual(values = shape_vec, breaks = breaks_vec, labels = lab_vec) +
  scale_linetype_manual(values = linetype_vec, breaks = breaks_vec, labels = lab_vec) +
  labs(x = "Perturbation Strength", y = "MSE", color = "Methods",
       fill = "Methods", shape = "Methods", linetype = "Methods") +
  theme(legend.position = "none") + 
  xlab("") +
  ylab("") +
  coord_cartesian(ylim = c(0.3, 4.75)); gg

save_myplot(plt = gg, plt_nm = "../results/figures/app-robustness-n-3000.pdf", 
            width = 1.75, height = 1.75)

# Slides
mses_BCF <- dat %>% 
  arrange(n) %>% 
  group_by(model, int_par, n, instr_str) %>% 
  summarise(mse = mean(test_mse)) %>% 
  filter(model %in% c("BCF-MLP-large")) %>% 
  mutate(model = paste(model, n))

gg <- ggplot(
  mses_BCF,
  aes(x = int_par, y = mse, color = model, fill = model,
      shape = model, linetype = model)
) +
  # facet_grid(~ n, scales = "fixed", labeller = label_parsed) +
  geom_line(
    data = mses2 %>%
      arrange(desc(model_)) %>%
      filter(model_ %in% c("IMP")), size=.35
  ) +
  geom_line() +
  coord_cartesian(ylim = c(1, 1.5))
gg

gg <- ggplot(
  mses_BCF %>% filter(int_par == 2),
  aes(x = n, y = mse)
) +
  # facet_grid(~ n, scales = "fixed", labeller = label_parsed) +
  geom_line() 

gg
