library(tidyverse)

dat2plot2 <- bind_rows(
    read_csv("../bcf-numerical-experiments/results/output_data/genes-test_data_0_noLatLon.csv") %>%
        mutate(setting = "test"),
    read_csv("../bcf-numerical-experiments/results/output_data/genes-train_data_noLatLon.csv") %>%
        mutate(setting = "train")
)



ggplot(dat2plot2) +
    geom_point(aes(x = X1, y = Y, col = setting)) +
    geom_point(aes(x = X1, y = `LS`))

# try pairs with higher signal
# try more predictors/envs
# try warfarin data
