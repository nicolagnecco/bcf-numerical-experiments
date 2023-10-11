library(tidyverse)

# Constant definitions
INPUT_TEMP <- "../data/processed/temp-data-california.csv"
INPUT_HOUSING <- "../data/processed/housing-sklearn.csv"
OUTPUT_HOUSING <- "../data/processed/housing-temp.csv"

# Main
dat_temp <- read_csv(INPUT_TEMP) %>% 
  mutate(Latitude_r = round(lat, 1),
         Longitude_r = round(lon, 1)) %>% 
  group_by(Latitude_r, Longitude_r) %>% 
  summarise(temp = mean(Mean_Temp))


dat_housing <- read_csv(INPUT_HOUSING) %>% 
  mutate(Latitude_r = round(Latitude, 1),
         Longitude_r = round(Longitude, 1))


dat <- dat_housing %>% 
  left_join(dat_temp, by = c("Latitude_r", "Longitude_r")) %>% 
  drop_na() %>%   
  rename(Temp = temp) %>% 
  select(-Latitude_r, -Longitude_r)

write_csv(dat, OUTPUT_HOUSING)
