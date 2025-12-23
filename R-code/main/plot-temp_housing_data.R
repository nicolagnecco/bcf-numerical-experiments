source("main/dependencies.R")

# Constant definitions
HOUSEDATA <- "../data/processed/housing-temp.csv"
RES_TEST <- "../results/output_data/housing-data/20251013_093050/test_mse-new.csv"
RES_TRAIN <- "../results/output_data/housing-data/20251013_093050/training_mse.csv"

FIG_NAME <- "../results/figures/housing_methods.pdf"


# We consider the following train/test splits:
AREA_LVLS <- c(
    "North/South",
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
    "rest/NW"
)

AREA_LBLS <- c(
  "N/S",
  "S/N",
  "E/W",
  "W/E",
  "SE/(N+SW)",
  "(N+SW)/SE",
  "SW/(N+SE)",
  "(N+SE)/SW",
  "NE/(S+NW)",
  "(S+NW)/NE",
  "NW/(S+NE)",
  "(S+NE)/NW"
)

# Read data
dat0 <- bind_rows(
  tibble(
    set = "Testing",
    read_csv(RES_TEST)
  ),
  tibble(
    set = "Training",
    read_csv(RES_TRAIN)
  )
)

dat_mse <- dat0 %>%
    mutate(parts = str_split(Rep, "_")) %>%
    unnest_wider(parts, names_sep = "_") %>%
    select(-Rep) %>%
    rename(Split = parts_1, Rep = parts_2) %>%
    mutate(Split = factor(Split, levels = AREA_LVLS, labels = AREA_LBLS))

# Aggregate over repetitions
dat <- dat_mse %>%
    group_by(set, Split, Method) %>%
    summarise(
        MSEmean = mean(MSE),
        MSEmin = min(MSE),
        MSEmax = max(MSE)
    )

# Reorder splits according to "hardness" for LS
dat_ols <- dat %>%
    filter(Method == "LS") %>%
    select(set, Split, Method, MSEmean) %>%
    pivot_wider(names_from = set, values_from = MSEmean) %>%
    mutate(diff = Testing - Training) %>%
    arrange(diff)

dat2plot <- dat %>%
    mutate(Split = factor(Split, levels = dat_ols$Split))

# Plot results for BCF, least squares, and other models.
dat_methods <- dat2plot %>% 
  filter(Method %in% c(
    "LS",
    "AnchorBooster-small",
    "CF-small", 
    "GroupDRO", 
    "BCF", "AVE"
  )) %>% 
  mutate(Method = refactor_methods(Method, rev = TRUE))

gg <- ggplot(dat_methods %>% arrange(Method)) +
    facet_wrap(~set) +
    geom_ribbon(
        aes(
            x = Split, ymin = MSEmin, ymax = MSEmax, fill = Method,
            group = Method
        ),
        alpha = 0.25
    ) +
    geom_line(aes(x = Split, y = MSEmean, col = Method, group = Method),
        alpha = 0.75
    ) +
    geom_point(aes(x = Split, y = MSEmean, col = Method),
        fill = "white", shape = 21, size = 2, stroke = 0.75, alpha = 0.75
    ) +
    scale_fill_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
    scale_color_manual(values = my_colors, guide = guide_legend(reverse = TRUE)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(
        colour = "Methods", shape = "Methods", fill = "Methods",
        x = "Geographic Split",
        y = "MSE"
    ) +
  theme(
    axis.ticks.x = element_line(linewidth = 0.25)
  ) +
  scale_x_discrete(breaks = unique(dat_methods$Split))
gg

save_myplot(
    plt = gg, plt_nm = FIG_NAME,
    width = 2.25, height = 2.25
)
