include("base.jl")
include("regressions.jl")


# %% --------

R"""
source("base.r")
FIGS_PATH = "figs/sensitivity/"
"""

# %% --------

function load_frame(model, analysis=:choice)
    results = deserialize(results_path("summary_stats/jls/$model"))

    df = flatmap(results) do x
        map(getfield(x, analysis)) do reg
            rt = NamedTuple{(:rt0, :rt25, :rt50, :rt75, :rt100)}(x.rt_quantiles)
            (;x.prm..., x.accuracy, reg..., rt...)
        end
    end |> DataFrame
    df.model .= model[3:end]
    df
end


# %% ===== load data ==========================================================

function ensure_columns!(df)
    if "subjective_slope" ∉ names(df)
        df.subjective_slope .= 1
        df.subjective_offset .= 0
    end
    df
end

s1_model_choice = mapreduce(load_frame, vcat, ["1-main", "1-zero_mean", "1-flat_prior"])
ensure_columns!(s1_model_choice)

reg1 = fit_regressions(DataFrame(make_frame(load_human_data(1))); study=1)
s1_human_choice = DataFrame(reg1.choice)

# %% --------

reg2 = fit_regressions(DataFrame(make_frame(load_human_data(2))); study=2)
s2_human_choice = DataFrame(reg2.choice)
s2_human_rt = DataFrame(reg2.rt)

s2_model_choice, s2_model_rt = map([:choice, :rt]) do analysis
    mapreduce(vcat, ["2-main", "2-zero_mean", "2-nometa"]) do name
        ensure_columns!(load_frame(name, analysis))
    end
end


# %% ===== fitting ============================================================

param_names = [:base_precision, :attention_factor, :cost, :confidence_slope, 
               :subjective_offset, :subjective_slope, :prior_mean, :prior_precision]

model = let
    m1 = DataFrames.select(s1_model_choice, :model, param_names..., :accuracy, :rt50)
    m1.study .= 1

    m2 = DataFrames.select(s2_model_choice, :model, param_names..., :accuracy, :rt50)
    m2.study .= 2
    vcat(m1, m2)
end

human = mapreduce(vcat, enumerate((reg1, reg2))) do (i, reg)
    rt50 = reg.rt_quantiles[3]
    accuracy = reg.accuracy
    DataFrame(;study=i, h_rt50=rt50, h_accuracy=accuracy)
end

@rput model human

# %% --------

R"""
fits <- model |> 
    left_join(human) |> 
    mutate( rt_err = (rt50 - h_rt50)/5, accuracy_err = accuracy - h_accuracy ) |> 
    mutate(loss = rt_err^2 + accuracy_err^2) |> 
    group_by(across(-c(accuracy, rt50))) |> 
    summarise(loss = mean(loss), rt_loss = mean(rt_err^2), accuracy_loss = mean(accuracy_err^2)) |> 
    group_by(model, study) |> 
    slice_min(loss, n=1) |> 
    arrange(study, model) |> 
    select(-c(h_rt50, h_accuracy, rt_loss, accuracy_loss))

fits |> 
    write_csv("results/summary_fits.csv")

fits |> 
    ungroup() |> 
    summarise(
        max(abs(rt_err*5000)),
        max(abs(accuracy_err))
    )
"""

R"""
library(xtable)
fits |> 
    select(study, model, cost, confidence_slope) |> 
    xtable(digits=3) |> 
    print(file="results/parameter_table.tex")
"""

# %% ==================== study 1 ====================

@rput s1_model_choice s1_human_choice

R"""
model_pal <- scale_colour_manual(values=c(
    main = BLUE,
    flat_prior = GRAY,
    zero_mean = RED,
    biased_mean = PURPLE,
    nometa = YELLOW
), aesthetics=c("fill", "colour"), name="")

bullseye <- function(data) list(
    geom_point(data=data, size=3, color="black"),
    geom_point(data=data, size=1.7, color="white"),
    geom_point(data=data, size=.7, color="black")
)

human_wide1 <- s1_human_choice %>%
    select(term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)

s1_model_choice %>%
    filter(accuracy > .55) %>%
    filter(model != "biased_mean") %>%
    group_by(model) %>%
    select(model, confidence_slope, cost, accuracy, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate) %>%
    select(model, `spdfirst:savV`, spdfirst) %>%
    ggplot(aes(spdfirst, `spdfirst:savV`)) +
    geom_point(mapping=aes(color=model), size=.3) +
    bullseye(human_wide1) +
    theme_classic() +
    model_pal + no_legend

fig("duration_interaction", w=2.5, pdf=T)
"""

# %% ==================== study 2 ====================

@rput s2_model_choice s2_model_rt s2_human_choice s2_human_rt

R"""
human2 <- bind_rows(
    mutate(s2_human_choice, term = glue("choice-{term}")),
    mutate(s2_human_rt, term = glue("rt-{term}")),
) %>% select(term, estimate) %>% pivot_wider(names_from=term, values_from=estimate)

model2 <- bind_rows(
    mutate(s2_model_choice, term = glue("choice-{term}")),
    mutate(s2_model_rt, term = glue("rt-{term}"))
) %>%
    filter(accuracy > .55) %>%
    select(model, confidence_slope, cost, accuracy, rt50, prior_mean, term, estimate) %>%
    pivot_wider(names_from=term, values_from=estimate)
"""

R"""
model2 %>%
    filter(model %in% c("main", "nometa")) %>%
    ggplot(aes(`choice-fstosnd`, `choice-fstosnd:totalConfidence`)) +
    geom_point(mapping=aes(color=model), size=.3, alpha=1) +
    bullseye(human2) +
    # labs(x="Relative Value", y="Relative Value\nby Overall Confidence") +
    theme_classic() +
    model_pal + no_legend

fig("confidence_consistency", w=2.5, pdf=T)
"""


R"""
model2 %>%
    filter(model %in% c("main", "nometa")) %>%
    ggplot(aes(`accuracy`, `rt-totalConfidence`, )) +
    geom_point(mapping=aes(color=model), size=.3) +
    bullseye(human2) +
    geom_hline(yintercept=0) +
    theme_classic() +
    expand_limits(x=c(0.5)) +
    model_pal + no_legend

fig("confidence_rt", w=2.5, pdf=T)
"""



R"""
model2 %>%
    ggplot(aes(`choice-ConfDif`, `choice-ConfDif:savV`)) +
    geom_point(mapping=aes(color=model), size=.3) +
    bullseye(human2) +
    theme_classic() +
    expand_limits(x=0, y=0) +
    model_pal + no_legend

fig("confidence_interaction", w=2.5, pdf=T)
"""
