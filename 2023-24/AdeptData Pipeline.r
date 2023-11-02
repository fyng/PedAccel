library(adeptdata)
library(dplyr)
library(ggplot2)
library(ggplot)
library(reshape2)
library(lubridate)

accel <- read.csv("RawAccel_Sample1.csv") %>%
  select(Time_s, X, Y, Z)

accel_plot <- ggplot(data = accel, aes(x = Time_s, color = variable)) +
  geom_line(data = accel, aes(y = X), color = "blue", linetype = "solid") +
  geom_line(data = accel, aes(y = Y), color = "red", linetype = "solid") +
  geom_line(data = accel, aes(y = Z), color = "green", linetype = "solid") +
  labs(x = "Time [s]", y = "Amplitude [g]", color = "Sensor\naxis",
       title = "Raw accelerometry data sample") +
  labs(color = "Sensor Axis") +  # Customize the legend title here
  scale_color_manual(values = c("blue", "red", "green"),
                     labels = c("X", "Y", "Z"))
ggsave("newplot3.png", plot = accel_plot)
