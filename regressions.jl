using RCall
using DataFrames

R"""
library(broom)
"""

df2tuples(df) = map(NamedTuple, eachrow(df))

function fit_regressions(df; study)
    @rput df

    R"""
    df$SubNum <- factor(df$subject)
    df$isfirstIchosen <- as.numeric(df$choice ==1)
    df$sVD <- scale(abs(df$val1 -df$val2)/10, scale=FALSE, center=TRUE)
    df$savV <- scale((df$val1+df$val2)/20, scale=FALSE, center=TRUE)
    df$RT <- (df$pt1+df$pt2)
    df$cRT <- scale(df$RT, scale=FALSE, center=TRUE)
    df$sfstItemV <- scale(df$val1, scale=FALSE, center=TRUE)
    df$ssndItemVal <- scale(df$val2, scale=FALSE, center=TRUE)
    df$fstosnd <- (df$val1 -df$val2)/10
    df$spdfirst<- (df$pt1/(df$pt1+df$pt2)) - 0.5 #scale(a1b$pdfirst, scale=FALSE, center=TRUE)
    df$cinitpres1 <- scale(df$initpresdur1, scale=FALSE, center = TRUE)
    """
    if study == 1
        R"""
        choice_formula = isfirstIchosen ~ fstosnd +  spdfirst + cRT + savV + spdfirst:savV
        rt_formula = log(RT*1000)~ sVD + savV
        """
    elseif study == 2
        R"""
        for (i in levels(df$SubNum) ) {
          df$ConfBias[df$SubNum==i]  <-  mean(c(df$conf1[df$SubNum==i], df$conf2[df$SubNum==i]))
        }

        df$cConfBias <- scale(df$ConfBias, scale=FALSE, center= TRUE)
        df$sumConfidence <- df$conf1+df$conf2
        df$totalConfidence <- (df$sumConfidence)/2 - df$ConfBias
        df$cfstConfidence <- scale(df$conf1, scale=FALSE, center = TRUE)
        df$csndConfidence <- scale(df$conf2, scale=FALSE, center = TRUE)
        df$ConfDif <- df$conf1 - df$conf2

        choice_formula = isfirstIchosen ~ fstosnd + cRT + ConfDif + spdfirst + totalConfidence + fstosnd:totalConfidence + savV + ConfDif:savV + spdfirst:savV
        rt_formula = log(RT*1000) ~ sVD + savV + ConfDif + totalConfidence
        """
    end

    R"""
    choice_fit = tidy(glm(choice_formula, data = df, family=binomial(link='logit')))
    rt_fit = tidy(lm(rt_formula, data = df))
    accuracy = with(subset(df, val1 != val2), mean((val1 > val2) == (choice == 1)))
    rt_quantiles = with(df, quantile(rt))
    """
    (;choice = df2tuples(rcopy(R"choice_fit")),
      rt = df2tuples(rcopy(R"rt_fit")),
      accuracy = rcopy(R"accuracy"),
      rt_quantiles = rcopy(R"rt_quantiles"),
      )
end

