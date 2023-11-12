library(adeptdata)
library(dplyr)
library(ggplot2)
library(reshape2)
library(lubridate)
library(readr)

# Replace "your_file.csv" with the path to your CSV file.
file_path <- "Patient1_Data.csv"

# Read the CSV file and output it to the terminal
data <- read_csv(file_path) #, headers = TRUE, stringsAsFactors = FALSE)

# Delete rows with junk information
new_data <- data[-(1:10), ]

print(new_data)
