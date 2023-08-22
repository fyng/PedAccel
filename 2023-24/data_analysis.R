install.packages("adeptdata")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("reshape2")
install.packages("lubridate")

library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)

# Load the data from the RDA file in the same folder as the script
load("2023-24/acc_walking_IU.rda")

# Check the structure of the loaded data
str(acc_walking_IU)

# Your data manipulation and plotting code
plot_data <- acc_walking_IU %>%
  filter(time_s < 6, subj_id == acc_walking_IU$subj_id[1]) %>%
  mutate(loc_id = factor(
    loc_id, 
    levels = c("left_wrist", "left_hip", "left_ankle", "right_ankle"),
    labels = c("Left wrist", "Left hip", "Left ankle", "Right ankle"))) %>%
  melt(id.vars = c("subj_id", "loc_id", "time_s")) 

# Create the ggplot plot
my_plot <- ggplot(plot_data, aes(x = time_s, y = value, color = variable)) + 
  geom_line() + 
  facet_wrap(~ loc_id, ncol = 2) + 
  theme_bw(base_size = 9) + 
  labs(x = "Exercise time [s]", 
       y = "Amplitude [g]", 
       color = "Sensor\naxis",
       title = "Raw accelerometry data of walking (100 Hz)")

# Save the plot as a PNG file
ggsave("2023-24/plot1.png", plot = my_plot)








# head(acc_walking_IU)

# acc_walking_IU %>%
#   filter(time_s < 6, subj_id == acc_walking_IU$subj_id[1]) %>%
#   mutate(loc_id = factor(
#     loc_id, 
#     levels = c("left_wrist", "left_hip", "left_ankle", "right_ankle"),
#     labels = c("Left wrist", "Left hip", "Left ankle", "Right ankle"))) %>%
#   melt(id.vars = c("subj_id", "loc_id", "time_s")) %>%
#   ggplot(aes(x = time_s, y = value, color = variable)) + 
#   geom_line() + 
#   facet_wrap(~ loc_id, ncol = 2) + 
#   theme_bw(base_size = 9) + 
#   labs(x = "Exercise time [s]", 
#        y = "Amplitude [g]", 
#        color = "Sensor\naxis",
#        title = "Raw accelerometry data of walking (100 Hz)") 