source("main/dependencies.R")

# Constant definitions
HOUSEDATA <- "../data/processed/housing-temp.csv"
RES_TEST <- "../results/output_data/housing/test_mse.csv"
RES_TRAIN <- "../results/output_data/housing/training_mse.csv"
FIG_NAME <- "../results/figures/housing_methods.pdf"

# We consider the California housing dataset 
# (https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
# based on the 1990 U.S. census.
# The unit of analysis is a block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data
# We try to predict
# Y: median house value, using the following covariates
# - median income,
# - median house age,
# - average number of rooms per household
# - average number of bedrooms per household
# - total population
# - average number of household members
# - average annual temperature between 1991 -- 2020.
# The temperature data is taken from https://prism.oregonstate.edu/normals/
# and describes the average annual conditions over the most recent three full decades.
# Unfortunately they don't have 1990 year, but I think they are more reliable than taking a single year data (i.e., 1990) since
# house prices is more likely to depend on the typical weather condition rather than the one of a specific year.

# We use as instruments Longitude and Latitude.

# We include temperature data because:
# - is predictive on the training data,
# - is correlated with Lat/Lon,
# - has possibly nonlinear effect in space on median house value 
# (e.g., positive correlation between median house value and 
# temperature in one area and negative in another area).
# This helps to create shifts.

# Here is a plot of datapoints with color-coded temperature.
dat_temp <- read_csv(HOUSEDATA)

ggplot(dat_temp) +
  geom_point(aes(x = Longitude, y = Latitude, col = Temp))


# We consider the following train/test splits:
# "North/South",
# "South/North",
# "East/West",
# "West/East",
# "SE/rest",
# "rest/SE",
# "SW/rest",
# "rest/SW",
# "NE/rest",
# "rest/NE",
# "NW/rest",
# "rest/NW"
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
  mutate(Rep = factor(Rep, levels = AREA_LBLS, labels = AREA_LBLS)) %>% 
  rename(Split = Rep)

# Here are the results of BCF vs least squares and constant model (average).
# - BCF = [f_0 = XGBoost(), gamma_0 = XGBoost(), f_imp = XGBoost()]; number of iterations = 10;
# - LS = XGBoost().
dat_methods <- dat %>% filter(Method %in% c("BCF", "LS", "AVE")) %>% 
  mutate(Method = if_else(Method == "LS", "OLS",
                          if_else(Method == "AVE", "ConstFunc", Method))) %>% 
  mutate(Method = refactor_methods(Method, rev = TRUE)) %>% 
  mutate(MSE_standard_error = if_else(set == "Training", 0, MSE_standard_error))

gg2 <- ggplot(dat_methods %>% arrange(Method)) +
  facet_wrap(~ set) +
  geom_ribbon(aes(x = Split, 
                  ymin = MSE - 2 * MSE_standard_error,
                  ymax = MSE + 2 * MSE_standard_error,
                  fill = Method, group = Method),
              alpha = 0.25) +
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
