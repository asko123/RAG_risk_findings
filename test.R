suppressMessages(library(dplyr, quietly = TRUE))
suppressMessages(library(lubridate, quietly = TRUE))
suppressMessages(library(tidyverse, quietly = TRUE))
suppressMessages(library(stringr, quietly = TRUE))
suppressMessages(library(sqldf, quietly = TRUE))
suppressMessages(library(gc, quietly = TRUE)) # For manual garbage collection

# ============================ READ IN RAW DATA ============================

data_dir <- "/local/data/dev/data" # Sys.getenv("LOCAL_DATA_DIR")
scan_raw <- read.csv(paste0(data_dir, "/model/semgrep_scanner.csv"), stringsAsFactors=FALSE) %>% unique()

# Semgrep branches in previous scope which convey old L1/2
merge_cloud_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
merge_prem_raw_prev <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_semgrep_biweekly.csv"), stringsAsFactors = F)
merge_cloud_raw_1 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART1.csv"), stringsAsFactors = F)
merge_cloud_raw_2 <- read.csv(paste0(data_dir, "/ingest/merge_cloud_branches_result_PART2.csv"), stringsAsFactors = F)
merge_prem_raw <- read.csv(paste0(data_dir, "/ingest/merge_prem_branches_result_TP.csv"), stringsAsFactors = F)
spr_git_mapping_raw_all <- read.csv(paste0(data_dir, "/ingest/spr_mapping_all.csv"), stringsAsFactors = F)
spr_raw <- read.csv(paste0(data_dir, "/ingest/spr.csv"), stringsAsFactors=FALSE) %>% unique()
appdir_raw <- read.csv(paste0(data_dir, "/ingest/appdir.csv"), stringsAsFactors=FALSE) %>% unique()
semgrep_historical <- read.csv(paste0(data_dir, "/ingest/semgrep_scan_historical_biweekly.csv"), stringsAsFactors=FALSE) %>% unique()

# ============================ DEFINE TIME RANGE PARAMETERS ============================

# Set time range - limit to most recent X months (default to 6 months)
months_to_include <- 6  # Adjust as needed (3-6 months suggested)
current_date <- Sys.Date()
cutoff_date <- current_date - months(months_to_include)
cutoff_date_str <- as.character(cutoff_date)

print(paste0("Limiting data processing to the last ", months_to_include, " months (since ", cutoff_date_str, ")"))

# ============================ DEFINE STANDARDIZED RUNWEEK FUNCTIONS ============================

get_startWeek <- function(date_str) {
  # Find the Monday of the week containing the given date
  if (is.na(date_str) || date_str == "") {
    return(NA)
  }
  date <- as.Date(substr(date_str, 1, 10))
  monday <- date - (wday(date) - 2) %% 7
  return(paste(as.character(monday), '15:00:00'))
}

get_endweek <- function(start_week) {
  if (is.na(start_week) || start_week == "") {
    return(NA)
  }
  start_date <- as.Date(substr(start_week, 1, 10))
  end_week <- start_date + 14
  end_week <- paste(as.character(end_week), '15:00:00')
  return(end_week)
}

get_biweekly_scope <- function(startWeek, endWeek) {
  if (is.na(startWeek) || is.na(endWeek)) {
    return(NA)
  }
  output <- paste('From', startWeek, 'to', endWeek)
  return(output)
}

create_time_range_list <- function(time_range) {
  # Break early if input is invalid
  if (is.na(time_range[1]) || is.na(time_range[2])) {
    return(NA)
  }
  
  minDate <- as.Date(time_range[1]) # get the earliest commit date
  maxDate <- as.Date(time_range[2]) # get the latest date (today)
  
  # Apply the cutoff date to limit range
  minDate <- max(minDate, cutoff_date)
  
  all_dates <- seq(as.Date(minDate), as.Date(maxDate), by='day')
  mondays_only <- all_dates[weekdays(all_dates) %in% c('Monday')]
  mondays_only <- sort(mondays_only, decreasing=TRUE)
  dates <- mondays_only[!is.na(mondays_only)]
  new_dates <- as.character(as.Date(dates[1]) + 14)
  start_dates <- as.character(as.Date(dates, '1970-01-01'))
  start_dates <- sapply(start_dates, paste, '14:59:59', sep=' ')
  result <- paste(start_dates, collapse=';')
  return(result)
}

# Define scan periods using the cutoff date
scan_periods <- tibble(
  start = c(cutoff_date_str),
  end = c(as.character(as.POSIXct(Sys.time())))
) %>% mutate(time_range = paste0(start, ' - ', end),
             time_range_list = lapply(time_range, create_time_range_list),
             startWeek = strsplit(as.character(time_range_list), ";")) %>%
  ungroup %>%
  unnest(startWeek) %>% ungroup %>%
  mutate(endWeek = get_endweek(startWeek),
         runWeek = get_biweekly_scope(startWeek, endWeek)) %>%
  select(startWeek, endWeek, runWeek) %>% unique() %>% glimpse()

gc() # Garbage collection after creating scan periods

# ============================ Model Data ============================

mmodel_appdir <- appdir_raw %>% select(deploymentId,
                                      deploymentName,
                                      ProdAccessPreApproval,
                                      Payment,
                                      External3rdPartyAudit,
                                      FinancialStatement,
                                      DMZ,
                                      HRTP,
                                      SCOE,
                                      ETC,
                                      EQStack,
                                      CCAR,
                                      ProvisionalAudit,
                                      Privacy_New,
                                      OutsourcedHosted,
                                      OutsourcedBuilt,
                                      PublicCloud,
                                      HighRisk,
                                      `MNPI.Data`
                                      ) %>% unique()

model_spr <- spr_raw %>% select(guid, deploymentId, productName, productStatus) %>% unique()

spr_git_mapping_raw_all <- spr_git_mapping_raw_all %>% 
  mutate(current_scope = 0) %>%
  mutate(current_scope = ifelse(is.na(current_scope), 1, 0),
         creationDate = as.POSIXct(creationDate, format= "%a %b %d %H:%M:%S GMT %Y"))

model_spr_git <- spr_git_mapping_raw_all %>% select(guid, repository.repoGuid, stack.name, current_scope, creationDate) %>%
  mutate(git_id = str_extract(repository.repoGuid, '\\d+')) %>% unique()

# Free up memory
rm(spr_git_mapping_raw_all)
gc()

# Combine cloud and premise data with time filtering
merge_cloud_raw_prev <- merge_cloud_raw_prev %>% 
  mutate(productStack = 'CLOUDSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_prem_raw_prev <- merge_prem_raw_prev %>% 
  mutate(productStack = 'GITSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_raw_prev <- rbind(merge_cloud_raw_prev, merge_prem_raw_prev) %>% unique()

# Free up memory
rm(merge_cloud_raw_prev, merge_prem_raw_prev)
gc()

# Combine merge_cloud_raw_1 and merge_cloud_raw_2 before filtering
merge_cloud_raw <- rbind(merge_cloud_raw_1, merge_cloud_raw_2) %>% 
  mutate(productStack = 'CLOUDSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

# Free up memory
rm(merge_cloud_raw_1, merge_cloud_raw_2)
gc()

merge_prem_raw <- merge_prem_raw %>% 
  mutate(productStack = 'GITSDLC') %>%
  filter(commit.date >= cutoff_date_str)  # Apply time filter

merge_raw <- rbind(merge_cloud_raw, merge_prem_raw) %>% unique()

# Free up memory
rm(merge_cloud_raw, merge_prem_raw)
gc()

semgrep_historical <- semgrep_historical %>% 
  mutate(git_id = as.character(git_id),
         startTime = scanTime) %>% 
  select(-scanTime) %>%
  filter(startTime >= cutoff_date_str)  # Apply time filter

# semgrep_historical contains future scope. Need to compare to the polaris scan to update those past 'new scope'
new_semgrep_historical <- semgrep_historical %>% 
  select(runWeek, git_id, guid, branch, productStack, scan_id, startTime) %>% 
  unique() %>% glimpse()

rm(semgrep_historical)
gc()

# ============================ SCAN DETAILS ============================

# Give credit to a git project if it got scanned once
model_scan <- scan_raw %>% 
  filter(startTime >= cutoff_date_str) %>%  # Apply time filter
  select(git_id, gitSHA, branch, scan_id, startTime) %>% 
  mutate(git_id = as.character(git_id))

model_scan <- sqldf("select a.*, b.startWeek, b.endWeek, b.runWeek 
                    from model_scan a 
                    left join scan_periods b 
                    where a.startTime >= b.startWeek and a.startTime <= b.endWeek")

model_scan <- model_scan %>% 
  group_by(git_id, gitSHA, branch, runWeek) %>%
  summarise(scan_id = max(scan_id), startTime = max(startTime)) %>% 
  ungroup() %>%
  unique()

rm(scan_raw)
gc()

# ============================ PRODUCT STACK WITH LATEST COMMIT DATE ============================

# Pipeline data will be used to get the earliest commit date for each git project - This plays the baseline for the standard_run table
model_pipeline <- merge_raw %>% 
  filter(commit.date >= cutoff_date_str) %>%  # Apply time filter
  mutate(git_id = as.character(project_id)) %>%
  mutate(timestamp = gsub("T", " ", commit.date),
         ts = gsub("Z", "", timestamp),
         ts = as.character(as.POSIXct(ts)),
         created_time = gsub("T", " ", commit.created_at),
         created_time = as.character(as.POSIXct(created_time))) %>%
  select(git_id, ts, created_time, branch, default, productStack, gitSHA) %>% 
  unique()

model_pipeline <- model_pipeline %>%
  mutate(
    startweek = get_startWeek(ts),
    endweek = get_endweek(startweek),
    runweek = get_biweekly_scope(startweek, endweek)
  )

# Use a more efficient approach for joining with scan_periods
model_pipeline <- sqldf("select a.*, b.startWeek, b.endWeek, b.runWeek 
                        from model_pipeline a 
                        left join scan_periods b 
                        where a.ts >= b.startWeek and a.ts <= b.endWeek")

rm(merge_raw)
gc()

gitsha_info <- model_pipeline %>% select(git_id, branch, default, productStack, gitSHA) %>% unique()

model_pipeline <- model_spr_git %>% 
  select(git_id, repository.repoGuid, stack.name, current_scope, creationDate) %>% 
  left_join(model_pipeline, by = "git_id") %>% 
  select(-stack.name) %>% 
  unique()

gc()

# ============================ ESTABLISH THE RUN WEEK FOR ALL THE PRODUCTS IN SCOPE ============================

# We split into 2 dataframes: one for products that have history and one that never have previous records
recurring_products <- model_pipeline %>% 
  filter(!is.na(ts)) %>%
  select(git_id, gitSHA, productStack, branch) %>%
  unique()

# Free up memory and simplify - we're not using unrecord_products later
# unrecord_products <- model_pipeline %>% 
#   filter(is.na(ts)) %>%
#   select(git_id, gitSHA, productStack, branch) %>%
#   unique()

# For recurring products, we get the min and max dates
standard_run_recurring <- model_pipeline %>%
  filter(!is.na(ts)) %>%
  group_by(git_id, gitSHA, productStack, branch) %>%
  summarise(
    min_date = min(ts, na.rm = TRUE), 
    max_date = max(ts, na.rm = TRUE),
    runWeek = first(runWeek)
  ) %>%
  ungroup()

# Simplify and make more direct
standard_run_recurring <- standard_run_recurring %>%
  left_join(model_scan, by = c("git_id", "gitSHA", "branch", "runWeek"))

gc()

# Simplify - use the latest runWeek for processing
latest_runWeek <- scan_periods %>% 
  arrange(desc(startWeek)) %>% 
  slice(1) %>% 
  pull(runWeek)

second_latest_runWeek <- scan_periods %>% 
  arrange(desc(startWeek)) %>% 
  slice(2) %>% 
  pull(runWeek)

print(paste0("Processing primarily for runWeek: ", latest_runWeek))
print(paste0("With comparison to previous runWeek: ", second_latest_runWeek))

# Simplify to focus on just the current and previous runWeeks
program_semgrep_scan <- standard_run_recurring %>%
  filter(runWeek %in% c(latest_runWeek, second_latest_runWeek)) %>%
  select(git_id, gitSHA, branch, productStack, runWeek, scan_id, startTime) %>%
  unique()

# Missing scans - focus only on the latest runWeek
missing_scan <- program_semgrep_scan %>% 
  filter(runWeek == latest_runWeek) %>%
  left_join(gitsha_info, by = c("git_id", "gitSHA", "branch", "productStack")) %>%
  filter(is.na(scan_id)) %>%
  ungroup() %>% glimpse()

# ============================ Finalize the data with enrichment ============================

# More memory-efficient joining - only include necessary fields
program_semgrep_scan_attributes <- model_pipeline %>% 
  select(git_id, repository.repoGuid, current_scope, creationDate) %>%
  left_join(model_spr, by = c("repository.repoGuid" = "guid")) %>%
  left_join(mmodel_appdir, by = "deploymentId") %>%
  unique()

gc()

# Final join with more targeted columns
program_semgrep_scan <- program_semgrep_scan %>% 
  left_join(program_semgrep_scan_attributes, by = "git_id") %>%
  glimpse()

# ============================ Save Data ============================

write.csv(missing_scan, paste0(data_dir, "/ingest/semgrep_missing_scan_biweekly.csv"), row.names = FALSE)
write.csv(program_semgrep_scan, paste0(data_dir, "/program_semgrep_scan_biweekly.csv"), row.names = FALSE)

# Clean up memory before exiting
rm(list = ls())
gc()

print(paste0("Biweekly Semgrep Scan Program Data Ends @ ", Sys.time()))



