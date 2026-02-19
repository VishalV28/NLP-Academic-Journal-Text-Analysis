# INDIVIDUAL CW 
# Student No. 37630504


# Setting up the libraries
install.packages("dplyr","tidytext","tm","stringr","wordcloud","tidyverse","topicmodels","cluster","factoextra","tidyr")
library(dplyr)
library(tidytext)
library(tm)
library(stringr)
library(wordcloud)
library(tidyverse)
library(topicmodels)
library(cluster)  
library(factoextra)  
library(tidyr)       
library(randomForest)
library(pROC)

# Loading Data
raw_data <- read.csv("journal_data_CW4.csv", stringsAsFactors = FALSE)

# Cleaning numeric columns
raw_data$views <- as.numeric(gsub(",", "", raw_data$views))
clean_data <- distinct(raw_data)

# BAG OF WORDS and EDA

# Pre-processing
data("stop_words")
custom_stop_words <- tibble(word = c("paper", "abstract", "study", "results", 
                                     "proposed", "method", "model", "based",
                                     "data", "analysis", "using", "system", "time"))

tidy_data <- clean_data %>%
  select(doc_number, journal, year, abstract) %>%
  unnest_tokens(word, abstract) %>%
  anti_join(stop_words, by = "word") %>%
  anti_join(custom_stop_words, by = "word") %>%
  # making sure to remove numbers if any
  filter(!str_detect(word, "\\d"))

# Most Common Words (Bar Plot)
top_words <- tidy_data %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 10)

# Function to filter top words by journal and plot the bar graph
plot_top_words <- function(journal_name){
  top_words <- tidy_data[tidy_data$journal == journal_name,]%>%
    count(word, sort = TRUE) %>%
    slice_max(n, n = 10)
  ggplot(top_words, aes(x = reorder(word, n), y = n)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(x = "Word", y = "Frequency" ) +
      theme_minimal()
}

plot_top_words('Journal of the Operational Research Society')
plot_top_words('International Journal of Computer Mathematics')
plot_top_words('Journal of Interdisciplinary Mathematics')
plot_top_words('Applied Artificial Intelligence')

# Function to plot the word cloud
plot_word_cloud <- function(data_1){
  set.seed(123)
  pal <- brewer.pal(8, "Dark2") 
  data_1 %>%
    count(word) %>%
    with(wordcloud(word, n, max.words = 50, colors = pal, random.order = FALSE, scale = c(2, 0.5)))
}

# Plotting words used prior to 2015 and post 2015
words_early <- tidy_data[tidy_data$year < 2018,]
words_late <- tidy_data[tidy_data$year >= 2018,]

plot_word_cloud(tidy_data)
plot_word_cloud(words_early)
plot_word_cloud(words_late)
# No clear thematic differences found

# TOPIC MODELLING (LDA)

# Creating document term matrix using standard tidytext piping
dtm_matrix <- tidy_data %>%
  count(doc_number, word) %>%
  cast_dtm(doc_number, word, n)

# Running LDA
# Experimenting with k=3, k=4
lda_3 <- LDA(dtm_matrix, k = 3, control = list(seed = 123))
lda_4 <- LDA(dtm_matrix, k = 4, control = list(seed = 123))

print(paste("Perplexity k=3:", round(perplexity(lda_3), 2)))
print(paste("Perplexity k=4:", round(perplexity(lda_4), 2)))

lda_model <- LDA(dtm_matrix, k=3, control = list(seed = 123))

# Visualising the top words in the topics
lda_topics <- tidy(lda_model, matrix = "beta")

top_terms <- lda_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

ggplot(top_terms, aes(x = reorder(term, beta), y = beta, fill = factor(topic))) +
  geom_col(show.legend = FALSE) +
  facet_wrap(~ topic, scales = "free") +
  coord_flip() +
  labs(title = "Top Terms for Each LDA Topic", x = "Term", y = "Beta")


# TASK 3: REGRESSION (Predicting Citations)

# Using the Topic Probabilities as features for regression
# Extracting Gamma (probability of each document belonging to a topic)
doc_gammas <- tidy(lda_model, matrix = "gamma") %>%
  mutate(topic = paste0("Topic_", topic)) %>%
  pivot_wider(names_from = topic, values_from = gamma) %>%
  mutate(document = as.integer(document))

# Joining back to main data
reg_data <- raw_data %>%
  inner_join(doc_gammas, by = c("doc_number" = "document"))

# Creating log_citations
reg_data$log_citations <- log(reg_data$citations + 1)

# Building Multiple Linear Regression Model
model_reg <- lm(log_citations ~ year + pages + Topic_1 + Topic_2 + Topic_3, 
                data = reg_data)

summary(model_reg)


# CLASSIFICATION (Predicting Journal)

# Creating a binary target
reg_data$is_JORS <- ifelse(reg_data$journal == "Journal of the Operational Research Society", 1, 0)
reg_data$is_IJCM <- ifelse(reg_data$journal == "International Journal of Computer Mathematics", 1, 0)
reg_data$is_JOIM <- ifelse(reg_data$journal == "Journal of Interdisciplinary Mathematics", 1, 0)
reg_data$is_AAI <- ifelse(reg_data$journal == "Applied Artificial Intelligence", 1, 0)

# Training the Logistic Regression Model
# Using the Topics found earlier as features to predict the journal
log_reg_model <- function(journal_name, topic_1, topic_2){
  model_class <- glm(journal_name ~ topic_1 + topic_2 + pages + year, 
                     data = reg_data, 
                     family = binomial)
  summary(model_class)
}

log_reg_model(reg_data$is_JORS,reg_data$Topic_1, reg_data$Topic_3)
log_reg_model(reg_data$is_IJCM,reg_data$Topic_2, reg_data$Topic_3)
log_reg_model(reg_data$is_JOIM,reg_data$Topic_1, reg_data$Topic_3)
log_reg_model(reg_data$is_AAI,reg_data$Topic_1, reg_data$Topic_3)


# RANDOM FOREST & ROC

# Using Topic Probabilities (Topic_1, Topic_2, Topic_3) + Pages + Year
rand_forest_model <- function(journal_name){
  class_data <- reg_data %>%
    mutate(Target_Journal = as.factor(ifelse(journal == journal_name, "Yes", "No"))) %>%
    select(Target_Journal, pages, year, Topic_1, Topic_2, Topic_3) %>%
    drop_na()
  
  # Splitting the data
  set.seed(123)
  sample_index <- sample(nrow(class_data), 0.7 * nrow(class_data))
  train_data <- class_data[sample_index, ]
  test_data  <- class_data[-sample_index, ]
  
  # Training Random Forest
  rf_model <- randomForest(Target_Journal ~ ., data = train_data, ntree = 100)
  print(rf_model)
  # Predicting on Test Data
  rf_pred_prob <- predict(rf_model, test_data, type = "prob")[,2] # Probability of "Yes"
  # Generating ROC Curve
  roc_obj <- roc(test_data$Target_Journal, rf_pred_prob)
  
  # Plotting ROC
  plot(roc_obj, main = paste("ROC Curve:",journal_name), col = "blue")
  text(0.5, 0.5, paste("AUC =", round(auc(roc_obj), 3)))
}

rand_forest_model("Applied Artificial Intelligence")
rand_forest_model("Journal of the Operational Research Society")
rand_forest_model("International Journal of Computer Mathematics")
rand_forest_model("Journal of Interdisciplinary Mathematics")


# Comparison of logistic and random forest models

# Training Logistic Regression (on same split as RF for fair comparison)
log_model <- glm(Target_Journal ~ ., data = train_data, family = binomial)
log_pred  <- predict(log_model, test_data, type = "response")
# Training Random Forest
rf_pred_prob <- predict(rf_model, test_data, type = "prob")[,2] # Prob of "Yes"
# Creating ROC Objects
roc_log <- roc(test_data$Target_Journal, log_pred)
roc_rf  <- roc(test_data$Target_Journal, rf_pred_prob)
# Plotting Together
plot(roc_log, col = "red", main = "Model Comparison: Logistic vs Random Forest")
lines(roc_rf, col = "blue")
legend("right", legend = c(paste("Logistic AUC =", round(auc(roc_log), 3)),
                                 paste("Random Forest AUC =", round(auc(roc_rf), 3))),
       col = c("red", "blue"), lwd = 2)

# CLUSTERING & PCA

# Preparing data for clustering
reg_data_2 <- raw_data %>%
  inner_join(doc_gammas, by = c("doc_number" = "document"))

# Clustering based on the Topic Probabilities (Topic_1, Topic_2, Topic_3)
cluster_data <- reg_data_2 %>%
  select(Topic_1, Topic_2, Topic_3)

# Finding optimal clusters
fviz_nbclust(cluster_data, pam, method = "silhouette", k.max = 6) +
  labs(title = "Optimal Number of Clusters: Silhouette Method")

# Applying PAM Clustering
pam_res <- pam(cluster_data, k = 3)

# Visualising Clusters
fviz_cluster(pam_res, 
             data = cluster_data,
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_classic(),
             main = "Cluster Plot (PCA reduced)")

# Analysing Clusters vs Actual Journals
table(Cluster = pam_res$clustering, Actual_Journal = reg_data$journal)
