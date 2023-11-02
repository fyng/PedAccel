library(gt3x2csv)

# Setting up a test directory - ignore this.
my_directory <- gt3x2csv:::local_dir_with_files()

# An example directory with some GT3X files
list.files(my_directory)
#> [1] "test_file1.gt3x" "test_file2.gt3x" "test_file3.gt3x" "test_file4.gt3x"
#> [5] "test_file5.gt3x"
gt3x_2_csv(
  gt3x_files = my_directory,
  outdir = NULL, # Save to the same place
  progress = FALSE, # Show a progress bar?
  parallel = TRUE # Process files in parallel?
  )

# Directory now has the new files.
list.files(my_directory)
#>  [1] "test_file1.gt3x"   "test_file1RAW.csv" "test_file2.gt3x"  
#>  [4] "test_file2RAW.csv" "test_file3.gt3x"   "test_file3RAW.csv"
#>  [7] "test_file4.gt3x"   "test_file4RAW.csv" "test_file5.gt3x"  
#> [10] "test_file5RAW.csv"