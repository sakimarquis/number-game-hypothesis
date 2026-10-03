# Plot predictive distribution for all sets

library(ggplot2)
d <- read.csv("numbergame_data.csv", header=T, sep=",")

# replace underscores in 'set' column with commas
d$set <- gsub("_", ",", d$set)

plot_all <- function(da) {
    plt <- ggplot(da, aes(x=target, y=rating)) +
        stat_summary(fun.y="mean", geom="bar")  + 
        facet_wrap( ~ set, scale='free') +
        scale_x_continuous(breaks=seq(0, 100, by = 10)) + 
        theme(axis.text.x = element_text(size=8)) +
        scale_y_continuous(breaks=NULL)

    return(plt)
}

ggsave("predictive_all.png", plot_all(d), heigh=20, width=30)