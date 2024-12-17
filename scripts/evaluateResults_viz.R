library(ggplot2)
library(dplyr)
library(data.table)

#df = read.csv('../synthData 2/holdout_results_with_log_probs.csv')
df = read.csv('./holdout_results_with_log_probs.csv')

summary_df <- df %>%
  group_by(iteration, method) %>%
  summarise(
    mean_log_prob = median(log_prob, na.rm = TRUE),
    se_log_prob = sd(log_prob, na.rm = TRUE) / sqrt(n())
  )
summary_df$method[summary_df$method == 'baseline'] = 'Baseline (Random)'
summary_df$method[summary_df$method == 'perplexitySampling'] = 'Perplexity (High)'
summary_df$method[summary_df$method == 'referenceSampling'] = 'Reference Proximity'

# Create the plot
ggplot(summary_df, aes(x = iteration, y = mean_log_prob, color = method, group = method)) +
  geom_point(position = position_dodge(width = 0.2)) +
  geom_line(position = position_dodge(width = 0.2)) +
  geom_errorbar(aes(ymin = mean_log_prob - se_log_prob, ymax = mean_log_prob + se_log_prob),
                width = 0.2, position = position_dodge(width = 0.2)) +
  labs(
    x = "Iteration",
    y = "Log Probability (OPT 3.7B)"
  ) +
  theme_classic() + 
  guides(color = guide_legend(title = "Sampling Method"))+ 
  # make iteration ticks at 1:10
  scale_x_continuous(breaks = seq(0, 10, 1)) +
  theme(
    legend.position = c(0.18, 0.8),
    legend.box.background = element_rect(color = "gray", size = 0.3), # Add a box around the legend
    legend.background = element_blank() # Make the legend background transparent
  )
ggsave("log_prob_plot.png", width = 6, height = 4)

df = read.csv('../synthData 2/dimRed_embeddings.csv')
df$name[df$name == 'baseline'] = 'Baseline'
df$name[df$name == 'highPerplexitySampling'] = 'High Perplexity Sampling'
df$name[df$name == 'lowPerplexitySampling'] = 'Low Perplexity Sampling'
df$name[df$name == 'referenceSampling'] = 'Reference Proximity Sampling'
df_centroids_pca = df %>%
  group_by(name, label) %>%
  summarise(
    pca_1 = mean(pca_1),
    pca_2 = mean(pca_2)
  )
df$label = factor(df$label, levels = c('Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5', 'Iteration 6', 'Iteration 7', 'Iteration 8', 'Iteration 9', 'Iteration 10', 'Reference'))
df_centroids_pca$label = factor(df_centroids_pca$label, levels = c('Iteration 1', 'Iteration 2', 'Iteration 3', 'Iteration 4', 'Iteration 5', 'Iteration 6', 'Iteration 7', 'Iteration 8', 'Iteration 9', 'Iteration 10', 'Reference'))
colors = c(
  "#FFFFD9","#EDF8B1","#C7E9B4", "darkgreen","#7FCDBB","#41B6C4","#1D91C0","#225EA8","#253494","#081D58",
  'red'
)
ggplot(df, aes(x = pca_1, y = pca_2, color = label)) +
  geom_point(alpha = 0.01) +
  geom_point(data = df_centroids_pca, aes(x = pca_1, y = pca_2, fill = label, color = label), 
             size = 2, shape = 23) +
  labs(
    x = "PC 1",
    y = "PC 2"
  ) +
  scale_color_manual(values = colors) +
  scale_fill_manual(values = colors) +
  facet_grid(~name) +
  guides(fill = guide_legend(override.aes = list(
      size = rep(2, 11), 
      shape = rep(23, 11), 
      color = colors
    )
  )) + 
  theme(strip.text = element_text(size = 15))
ggsave("pca_plot.png", width = 15, height = 5, dpi = 500)



df_centroids_pca = df %>%
  group_by(name, label) %>%
  summarise(
    var_pca_1 = var(pca_1),
    var_pca_2 = var(pca_2),
    pca_1 = mean(pca_1),
    pca_2 = mean(pca_2)
  )
ggplot(df_centroids_pca, aes(x = var_pca_1, y = var_pca_2, color = label, shape = name)) +
  geom_point() +
  labs(
    x = "Variance in PC 1",
    y = "Variance in PC 2"
  )
