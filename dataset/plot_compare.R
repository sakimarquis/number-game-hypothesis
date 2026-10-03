# Compare predictive for different sets

library(ggplot2)
d <- read.csv("numbergame_data.csv", header=T, sep=",")

# replace underscores in 'set' column with commas
d$set <- gsub("_", ",", d$set)

plot_compare <- function(da, plot_sets){
    a <- subset(da, is.element(da$set, plot_sets))
    plt <- ggplot(a, aes(x=target, y=rating)) +
        stat_summary(fun.y="mean", geom="bar", aes(fill=set))  + 
        facet_wrap( ~ set, scale='free', ncol=1) +
        scale_x_continuous(breaks=seq(0, 100, by = 10)) + 
        theme(axis.text.x = element_text(size=8), 
              axis.text.y = element_text(size=8), 
              panel.grid.minor.y = element_blank(), 
              panel.grid.major.y = element_line(size = .1), 
              legend.position="none") +
        scale_y_continuous(breaks=NULL)
    
    return(plt)
}

# Ends in 1 (different set lengths)
sets1 <- c("81, 71, 21, 31", "71, 31, 21", "71, 11")

# Begin w/ 64, ends in 4, all length 2
sets2 <- c("64", "64, 4", "64, 44")

# 1 small 1 large, {~5, ~65}
sets3 <- c("5, 65", "7, 67", "64, 4")

# Single numbers (all 1-15  & 4 others)
sets4 <- c("1","2","3","4","5","6","7","8","9","10","11","12","13","14","15", "72", "73", "89", "95")

# Ends in 3 or 9; ends in 9 or 7; ends in 3 or 5
sets5 <- c("9, 33, 73, 99", "29, 77, 37, 17", "23, 35, 95, 83")

# Multiples of 3 vs. ends in 3
sets6 <- c("3, 63", "7, 63", "33, 3", "93, 43, 83, 53")

plot_compare(d, sets1)