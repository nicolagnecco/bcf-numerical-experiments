source("main/dependencies.R")

# Constant definitions
HOUSEDATA <- "../data/processed/housing-temp.csv"
RES_TEST <- "../results/output_data/test_mse-new.csv"
RES_TRAIN <- "../results/output_data/training_mse-new.csv"
FIG_NAME <- "../results/figures/housing_methods.pdf"



# We consider the following train/test splits:
AREA_LBLS = c( "North/South",
               "South/North",
               "East/West",
               "West/East",
               "SE/rest",
               "rest/SE",
               "SW/rest",
               "rest/SW",
               "NE/rest",
               "rest/NE",
               "NW/rest",
               "rest/NW")

dat <- 
  bind_rows(
    tibble(
      set = "Testing",
      read_csv(RES_TEST)),
    tibble(
      set = "Training",
      read_csv(RES_TRAIN)
    )) %>% 
  mutate(parts = str_split(Rep, "_")) %>%          
  unnest_wider(parts, names_sep = "_") %>%         
  select(-Rep) %>% 
  rename(Split = parts_1, Rep = parts_2) %>% 
  mutate(Split = factor(Split, levels = AREA_LBLS, labels = AREA_LBLS)) %>% 
  group_by(set, Split,  Method) %>% 
  summarise(MSE = mean(MSE))

# Here are the results of BCF vs least squares and constant model (average).
# - BCF = [f_0 = XGBoost(), gamma_0 = XGBoost(), f_imp = XGBoost()]; number of iterations = 10;
# - LS = XGBoost().
dat_methods <- dat %>% filter(Method %in% c("BCF", "LS", "AVE")) %>% 
  mutate(Method = if_else(Method == "LS", "OLS",
                          if_else(Method == "AVE", "ConstFunc", Method))) %>% 
  mutate(Method = refactor_methods(Method, rev = TRUE))

gg2 <- ggplot(dat_methods %>% arrange(Method)) +
  facet_wrap(~ set) +
  # geom_ribbon(aes(x = Split, 
  #                 ymin = MSE - 2 * MSE_standard_error,
  #                 ymax = MSE + 2 * MSE_standard_error,
  #                 fill = Method, group = Method),
  #             alpha = 0.25) +
  geom_line(aes(x = Split, y = MSE, col = Method, group = Method), alpha = 0.75) +
  geom_point(aes(x = Split, y = MSE, col = Method,),
             fill = "white", shape = 21, size = 2, stroke = 0.75, alpha = 0.75) +
  scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_fill_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
  scale_shape_manual(values = my_shapes, guide = guide_legend(reverse = TRUE)) +
  scale_size_manual(values = my_sizes, guide = guide_legend(reverse = TRUE)) +
  scale_linetype_manual(values = my_linetypes, guide = guide_legend(reverse = TRUE)) +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.25, hjust=0.25)) +
  labs(colour="Methods", shape="Methods", fill="Methods")

gg2
save_myplot(plt = gg2, plt_nm = FIG_NAME, 
            width = 2, height = 2)
