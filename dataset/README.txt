==========
== data ==
==========

    numbergame_data.csv 
    ===================
    The primary dataset file, including 272700 binary responses for the numbergame. 

    This comma-delimited file includes 14 columns, where each row signifies a single response:

        Primary collected data
        ----------------------
        [1]:'set'
            stimulus set used for this trial
        [2]:'id'
            unique id assigned to each individual subject
        [3]:'rating'
            binary (1 or 0) rating, corresponding to whether the response was 'yes' or 'no' 
        [4]:'rt'
            reaction time (milliseconds) associated with the response (NOTE: some values are NA)
        [5]:'target'
            target stimuli used for this trial
        [6]:'trial'
            within-subject trial number, ranging from 0 to 449

        Post-collection analysis
        ------------------------
        [7]:'hits'
            number of data points for this unique (set, target) pair
        [8]:'p'
            mean rating for this unique (set, target) pair; calculated as ( yes / (yes + no) ), 
            where 'yes' denotes the number of positive responses and 'no' denotes negative
        [9]:'H'
            entropy calculated as a function of p for this unique (set, target) pair

        Demographics
        ------------
        [10]:'age'
            subject's age (NOTE: some values are NA)
        [11]:'firstlang'
            subject's first language
        [12]:'zipcode'
            subject's zip code
        [13]:'gender'
            subject's gender
        [14]:'education'
            subject's highest education level achieved


    set_descriptions.csv
    ====================
    Set descriptions collected in post-experiment questionnaire. For each subject, 
    5 set stimuli were randomly selected from the set of 15 they were shown. Subjects
    were asked to describe, in their own words, the pattern of numbers in these sets.

    This comma-delimited file includes 3 columns, where each row signifies a single response:

        [1]:'id'
            unique id assigned to each individual subject
        [2]:'set'   
            set stimulus described
        [3]:'descr'
            description of this concept stimulus set


    set_descriptions_typed.csv
    ==========================
    The rows from `set_descriptions.csv` with a semantic type and, where the
    description can be executed, a function reference into the d100 hypothesis library.

    This comma-delimited file includes 5 columns:

        [1]:'id'
            unique id assigned to each individual subject
        [2]:'set'
            set stimulus described
        [3]:'descr'
            subject's original description
        [4]:'type'
            description category: rule, interval, dk, or other
        [5]:'library_fn'
            matching function in `library/library_d100.py`; blank when no executable match exists


    instructions_rt.csv
    ===================
    Amount of time subjects spent on each of 3 instruction pages.

    This comma-delimited file includes 3 columns, where each row signifies a single response:

        [1]:'id'
            unique id assigned to each individual subject
        [2]:'instruct'
            instruction number (there are 3 instruction pages, so this is 0, 1, or 2) 
        [3]:'rt'
            reaction time (milliseconds) associated with this instruction


    show_set_rt.csv
    ==================
    Before rating targets for a set stimulus, subjects were shown a page with only the 
    stimulus set, and asked to press the 'space' key to continue when they were ready to 
    begin rating targets. This file records the amount of time spent on this page for each 
    set, for each subject.

    This comma-delimited file includes 3 columns, where each row signifies a single response:

        [1]:'set'
            set stimulus shown
        [2]:'id'
            unique id assigned to each individual subject
        [3]:'rt'
            reaction time (milliseconds) associated with this instruction



=============
== scripts ==
=============

    plot_all.R
    ==========
    Plot predictive distribution for all stimulus sets. This is how we generated `predictive_all.pdf`, 
    included in the main directory.

    plot_compare.R
    ==============
    Plot predictive distribution for multiple specified stimuli to compare. This file also includes
    a short list of set lists, with interesting patterns to compare.

    plot_focus.R
    ============
    Same as `plot_compare.R`, but highlighting certain targets to compare how predictive distributions 
    across multiple stimuli may reflect common patterns. This is how we generated Figure 2 in our paper.

    recompute_columns.py
    ====================
    Short python script showing how the data may be loaded into pandas, and showing how we computed
    the 'p', 'H', 'typicality', and 'hits' columns. Running the script writes a separate
    `numbergame_data_recomputed.csv` file and leaves the distributed source file unchanged.



===========
== other ==
===========

    predictive_all.pdf
    ==================
    full plot of predictive distributions across all subjects for every stimulus set used in the dataset
