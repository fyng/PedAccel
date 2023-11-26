# library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)
library(read.gt3x)
library(scattermore)

gt3xfile <-
  system.file(
    "extdata", "Patient1_GT3X.gt3x",
    package = "read.gt3x")

gt3xfolders <- unzip.gt3x("Patient1_GT3X.gt3x", location = tempdir())

gt3xfolder <- gt3xfolders[1]
X <- read.gt3x(gt3xfolder)

X <- as.data.frame(X)

# X <- as.data.frame(gt3xfolders[1])
# head(X)

# X <- read.gt3x(gt3xfile)

X$Magnitude <- sqrt(X$X^2 + X$Y^2 + X$Z^2) - 1

X$time_seconds <- seq(0, by = 1 / 100, length.out = nrow(X))

ggplot(X, aes(x = time_seconds)) +
  geom_scattermore(aes(y = Magnitude),
            color = "purple", linetype = "solid") +
  labs(x = "Time [s]", y = "Magnitude [g]", color = "Magnitude",
       title = "100Hz Patient 1 ENMO Data") +
  scale_color_manual(values = c("purple"))

# Export plot to PNG
#ggsave("Patient1data1_magnitude_seconds.png", plot = accel_plot)

