# Plot predictive distribution, coloring targets for specific 'concepts' (multiples of 3, ends in 3)

library(ggplot2)
d <- read.csv("numbergame_data.csv", header=T, sep=",")

# replace underscores in 'set' column with commas
d$set <- gsub("_", ",", d$set)

plot_focus <- function(da, plot_sets, focus_targets, fill_color){
    a <- subset(da, is.element(da$set, plot_sets))
    plt <- ggplot(a, aes(x=target, y=rating)) +
        stat_summary(data=subset(a, is.element(a$target, focus_targets)), 
                     fun.y="mean", geom="bar", aes(width=.9), fill=fill_color)  + 
        stat_summary(data=subset(a, !is.element(a$target, focus_targets)), 
                     fun.y="mean", geom="bar", aes(width=.9), fill="#c5b6aa")  + 
        facet_wrap( ~ set, scale='free', ncol=1) +
        scale_x_continuous(breaks=seq(0, 100, by = 10)) + 
        theme(axis.text.x=element_text(size=8), 
              axis.text.y=element_text(size=8), 
              panel.grid.minor.y=element_blank(), 
              panel.grid.major.y=element_line(size = .1),
              legend.position="none") +
        scale_y_continuous(limits=c(0.0,1.0), breaks=seq(0.0, 1.0, by = .2)) +
        ylab("p")

    return(plt)
}

# Sets to test
sets <- c("3, 63", "33, 3", "93, 43, 83, 53")

# Multiples of 3
targets1 = c('3', '6', '9', '12', '15', '18', '21', '24', '27', '30', '33', '36', '39', '42', '45', '48', '51', '54', '57', '60', '63', '66', '69', '72', '75', '78', '81', '84', '87', '90', '93', '96', '99')
# Ends in 3
targets2 = c('3', '13', '23', '33', '43', '53', '63', '73', '83', '93')

p1 <- plot_focus(d, sets, targets1, "#0099ff")
p2 <- plot_focus(d, sets, targets2, "#ff1919")