# learning past tense
"""
1. Generation of the verb corpus (506 verbs) split into 3 tiers (corresponding to 3 stages of learning)
2. Convert the verb corpus into a training set by encoding each verb as wickelfeature_detectors activation
    pattern.
3. Tran network using the original learning algorithm (as described by McClelland and Rumelhart)
4. Review the learning curve for regular and irregular verbs on each stage. The main purpose is to show
    that performance on irregular verbs decreases at the beginning of the stage 2.
5. Confirm that the trained network correctly predicts the past tense.

    Binding (decoding) network:
    binding (decoding) network is not actually necessary for the experiment. the accuracy of past tense production
    can be checked by comparing the actual network output (activation pattern of output wickelfeature detectors)
    against the target activation pattern of output wickelfeature detectors.

"""

from boilerplate import *
from wickelfeatures import *
from collections import Counter  # calculates the number of each unique item in the list
import plotly.express as px
import pickle


# generate corpus of verbs for the experiment
def get_corpus():
    """
    generate corpus of 506 verbs (as was used in the experiment) split into 3 tiers:
    1.  10 most popular verbs (8 - irregular, 2 regular)
    2.  410 verbs learned at the second phase (306 regular, 104 irregular)
    3.  the remaining 86 verbs (69 irregular)
    :return: dataframe of 506 verbs and their past tenses marked for regularity and split into 3 tiers
    """

    frequency_file = 'words/word_frequency_df_ready.csv'
    ft = pd.read_csv(frequency_file, sep='\t')
    ft = ft[(ft.pos_type == 'v') | (ft.pos_type == 'l')]
    ft_x = ft.groupby('L1').freq.sum().sort_values(ascending=False)
    ft_x = pd.DataFrame(ft_x)

    verbs_file = 'words/most-common-verbs-english.csv'
    vf = pd.read_csv(verbs_file)
    vf = vf.join(ft_x, on='Word', how='left', rsuffix='_r')
    vf = vf[vf['Simple Past'].notnull()]
    vf = vf[vf['freq'].notnull()]
    vf = vf.drop(columns=['3singular', 'Present Participle', 'Past Participle'])
    vf.rename(columns={'Simple Past': 'past', 'Word': 'word', 'top10': 'tier'}, inplace=True)
    vf = vf[:506]

    # add incorrect overregularized forms of irregular verbs. e.g. go -> goed
    vf['over_regular'] = vf.apply(lambda r: r.word + 'd' if r.word[-1] == 'e' else r.word + 'ed', axis=1)

    tier_update = lambda r: 1 if r.tier == 1 else (3 if r.name > 430 else 2)
    vf.tier = vf.apply(tier_update, axis=1)
    vf.set_index('word', inplace=True)

    is_regular = lambda r: True if r.name + 'd' == r.past or r.name + 'ed' == r.past else False
    vf['is_regular'] = vf.apply(is_regular, axis=1)

    return vf

    # test:

    verbs = get_corpus()
    verbs.head(100)
    vf.head(100)
    len(verbs[verbs.tier == 3])
    verbs[(verbs['is_regular'] == True) & (verbs['tier'] == 3)].shape

    verbs['test'] = verbs.apply(lambda r: r.name, axis=1)
    verbs.head()


# generate training set from the corpus of verbs (converting verbs (words) into wickelfeature activation patterns)
def build_training_set(verbs, blur_ratio=0.0, verbose=False):
    """

    :param verbs: list of verbs for the training set generation
    :param blur_ratio: additional detectors
    :param verbose: print or not progress report during execution
    :return:    X.T array of verb encodings (wickelfeature detector activation patterns)
                Y.T array of past tense of the verbs encodings (also wickelfeature detector activation patterns)
                R.T array of incorrect overregularized past tense (also wf detector activation patterns)
    """


    X = np.zeros((len(verbs), 460))  # representation of main verbs
    Y = np.zeros((len(verbs), 460))  # representation of past tense
    R = np.zeros((len(verbs), 460))  # representation of overregularized past tense


    for i, v in enumerate(verbs.index):
        if i % 50 == 0 and verbose:
            print('i: ', i)
        X[i] = word_activation_pattern(v, i_return=False, blur_ratio=blur_ratio)
        Y[i] = word_activation_pattern(verbs.loc[v, 'past'], i_return=False, blur_ratio=blur_ratio)
        R[i] = word_activation_pattern(verbs.loc[v, 'over_regular'], i_return=False, blur_ratio=blur_ratio)

    return X.T, Y.T, R.T  # features x samples

    # test:
    verbs = get_corpus()
    X, Y, R = build_training_set(verbs)


# train the network
def train_network(X, Y, R, verb_indices, epochs=1000, T=1, **kwargs):
    """
    :param X: array of input features 460 x 506 (features x samples)
    :param Y: array of output features 460 x 506 (features x samples)
    :param R: array of overregularized versions of output features 460 X 506 (features x samples)
    :param verb_indices: series of verbs indices in the training set (for log identification)
    :param epochs: number of epochs of training
    :param T: temperature
    :return: w - weights of the trained network. 460 x 460
    """

    n_features = X.shape[0]  # 460 (each feature is 1 wickelfeature detector) X.shape=(460, 506)
    n_examples = X.shape[1]

    def verdict(target, prediction):  # hit/miss feedback function
        if target == 1 and prediction == 1:
            return 'hit'
        elif target == 1 and prediction == 0:
            return 'miss'
        elif target == 0 and prediction == 1:
            return 'false_alarm'
        elif target == 0 and prediction == 0:
            return 'correct_rejection'
        else:
            return 'incorrect_data'

    if 'w' in kwargs:
        w = kwargs['w']  # for 2nd and 3rd stage learning weights 'w' is provided as an argument
    else:
        w = np.zeros((n_features, n_features))  # initialize weights for the 1st stage

    if 'theta' in kwargs:
        theta = kwargs['theta']
    else:
        theta = np.zeros(n_features)  # initialize threshold

    if 'verbose' in kwargs:  # print progress report during the execution
        verbose = kwargs['verbose']
    else:
        verbose = False

    training_log = [list(), dict(), list(), dict()]  # dictionary of training history data
    """
    accuracy (rate) - (hit+correct_rejection)/total_output_size. shows percentage of correctly activated output units
    (wickelfeature detectors) for a given epoch (1 input vector (=1 verb).
    
    training_log[0] - list of accuracy rates on target (correct past tense) at each epoch (irrespective of the verb)
    training_log[1] - dictionary of accuracy rates for each individual verb (after each epoch for that verb)
    
    training_log[2] - list of accuracy rates on overregularized past tense at each epoch (irrespective of the verb)
    training_log[3] - dictionary of accuracy rates on overregularized past tense for each individual verb
    """
    epoch = 0

    while epoch < epochs:

        if epoch % 100 == 0 and verbose:  # verbose every 100 epochs
            print('epoch #: ', epoch)

        exm = np.random.randint(0, n_examples)  # choose random verb
        x = X[:, exm]  # input activation for the chosen verb
        y = Y[:, exm]  # target output activation for the chosen verb
        r = R[:, exm]  # incorrect overregularized pattern

        z = w @ x  # activation multiplied by weights z.shape=(460,506)
        z_scaled = (z - theta)/T  # scaled by threshold and temperature
        p = 1/(1 + np.exp(-z_scaled))  # probability of activation of each of the output units

        # sampling activation. the actual activation of output units is supposed to be stochastic
        # with probability of activation p as calculated above.
        random_factor = np.random.uniform(0, 1, n_features)  # random factor
        a = [1 if r < p else 0 for r, p in zip(random_factor, p)]  # activation decision for each output unit
        # a is actual (stochastically based in each case) activation of output units. a.shape = (460,506)

        # compare individual unit (wickelphone detectors) output against the target encoding:
        compare = [verdict(target, prediction) for target, prediction in zip(y, a)]  # a list of verdicts
        compare_counter = Counter(compare)  # calculate the number of hits/misses/false_alarms/correct_rejections

        accuracy = (compare_counter['hit'] + compare_counter['correct_rejection'])/len(compare)
        training_log[1][verb_indices.index[exm]] = training_log[1].get(verb_indices.index[exm], []) + [accuracy]

        # update weights and thresholds:
        for i, c in enumerate(compare):  # for each feature (=wickelfeature_detector)
            if c == 'miss':  # if output feature is a miss (it's 0 where it shall be 1)
                for j in range(n_features):  # loop over input features.
                    if x[j] == 1:  # if input feature is 1 (so it matches the output)
                        w[i, j] += 1  # the weight is increased
                theta[i] -= 1  # and the threshold is decreased
            elif c == 'false_alarm':  # but if the output feature is a false alarm (it's 1 where it shall be0)
                for j in range(n_features):  # the loop over input features
                    if x[j] == 1:  # all input features that are 1
                        w[i, j] -= 1  # have their weight decreased
                theta[i] += 1  # and the threshold increased.

        # compare individual unit (wickelphone detectors) output against the overregularized encoding:
        compare_or = [verdict(target, prediction) for target, prediction in zip(r, a)]  # compare with over_reg
        compare_or_counter = Counter(compare_or)

        accuracy_or = (compare_or_counter['hit'] + compare_or_counter['correct_rejection']) / len(compare_or)
        training_log[3][verb_indices.index[exm]] = training_log[3].get(verb_indices.index[exm], []) + [accuracy_or]

        epoch += 1
        training_log[0] += [accuracy]  # accuracy compare to the target (correct past tense)
        training_log[2] += [accuracy_or]  # accuracy compare to the overregularized form (incorrect past tense)

    return w, theta, training_log


def run_experiment():

    corpus = get_corpus()  # create corpus of verbs
    corpus['ind'] = np.arange(len(corpus))  # verb indices column added
    X, Y, R = build_training_set(corpus, blur_ratio=0.9)  # build training set (R is used for irregular verbs only)

    phase_1_indices = corpus.ind[corpus.tier == 1]  # top 10 verbs for the first phase (8 irregular, 2 regular)
    X_1 = np.array([X[:, i] for i in phase_1_indices]).T  # input activation (base form, infinitive)
    Y_1 = np.array([Y[:, i] for i in phase_1_indices]).T  # target activation (correct past tense form)
    R_1 = np.array([R[:, i] for i in phase_1_indices]).T  # overregularized activation (take -> 'taked')

    # Stage 1 training: acquisition of top 10 words.
    w_1, theta_1, training_log_1 = train_network(X_1, Y_1, R_1, verb_indices=phase_1_indices, epochs=1000, T=1)
    # after 100 epochs the accuracy is 99.1% on unit-by-unit basis (note, that 99.1% is still wrong answer (eg 'takem')
    # human-level performance requires 100% accuracy

    plt.plot(training_log_1[0])  # plot accuracy on target
    plt.plot(training_log_1[2])  # plot accuracy on overregularized form
    plt.show()
    moving_average_plot(training_log_1[0], window=10)  # plot moving average (on target)

    # Stage 2 training: acquisition of top
    phase_2_indices = corpus.ind[(corpus.tier == 1) | (corpus.tier == 2)]  # verbs for tier 1 and 2
    X_2 = np.array([X[:, i] for i in phase_2_indices]).T  # input activation (base form, infinitive)
    Y_2 = np.array([Y[:, i] for i in phase_2_indices]).T  # target activation (correct past tense form)
    R_2 = np.array([R[:, i] for i in phase_2_indices]).T  # overregularized activation (take -> 'taked')

    # Stage 2 training: learning large number of regular verbs
    w_2, theta_2, training_log_2 = train_network(X_2, Y_2, R_2, verb_indices=phase_2_indices,
                                                 epochs=190*420, T=1, w=w_1, theta=theta_1, verbose=True)

    # 190 epochs for each verb (on average).
    # even after 13K epochs (30 repetitions per verb) performance is above 99%

    plt.plot(training_log_2[0])  # show accuracy on target activation of stage 2
    plt.show()
    moving_average_plot(training_log_2[0], window=1000)

    # save phase 2 training results:
    file = open('_simulation_results/experiment_results_2', 'wb')
    pickle.dump([corpus, X, Y, R, w_2, theta_2, training_log_1, training_log_2], file)  # save experiment data
    file.close()

    return True


def analyze_data():

    # load experiment data:
    file = open('_simulation_results/experiment_results_2', 'rb')
    [corpus, X, Y, R, w_2, theta_2, training_log_1, training_log_2] = pickle.load(file)
    file.close()

    # tier 1 verbs combined training log a) on target, b) on over-regularized version
    accuracy_on_target = {i: training_log_1[1][i] + training_log_2[1][i] for i in training_log_1[1].keys()}
    accuracy_on_over_reg = {i: training_log_1[3][i] + training_log_2[3][i] for i in training_log_1[3].keys()}

    list(corpus[corpus.tier == 1].index)  # the list of tier 1 verbs (just a reminder)

    # basic illustration. compare two regular vs irregular verbs from tier 1
    v1 = 'look'
    v2 = 'live'
    v3 = 'feel'
    plot_data = [accuracy_on_target[v1], accuracy_on_target[v2], accuracy_on_target[v3], accuracy_on_over_reg[v3]]
    plot_labels = [v1, v2, v3 + ' on target', v3 + ' on overregularized form']
    moving_average_multiline_plot(plot_data, window=10, legend=plot_labels)


def predict_past_tense(n_predictions=10, T=1):
    """
    :param T: temperature (float)
    :param n_predictions: number of predictions per verb (int)
    :return:
    """

    # load experiment data:
    file = open('_simulation_results/experiment_results_1', 'rb')
    [corpus, X, Y, R, w_2, theta_2, training_log_1, training_log_2] = pickle.load(file)
    file.close()

    # tier 3 verbs
    tier_3_indices = corpus.ind[corpus.tier == 3]  # verbs for tier 3

    def verdict(target, prediction):
        if target == 1 and prediction == 1:
            return 'hit'
        elif target == 1 and prediction == 0:
            return 'miss'
        elif target == 0 and prediction == 1:
            return 'false_alarm'
        else:
            return 'correct_rejection'

    predictions_accuracy = {}
    n_features, n_verbs = X.shape[0], len(tier_3_indices)  # number of features
    i = 0

    while i < n_verbs:  # loop over verbs

        verb = tier_3_indices.index[i]
        verb_index = tier_3_indices[i]
        x = X[:, verb_index]
        y = Y[:, verb_index]
        r = R[:, verb_index]

        z = w_2 @ x
        z_scaled = (z - theta_2) / T  # scaled by threshold and temperature
        p = np.array([1 / (1 + np.exp(-z)) if z < 100 else 0 for z in z_scaled])

        random_factor = np.random.uniform(0, 1, [n_features, n_predictions])
        a = [[1 if r < p else 0 for r, p in zip(random_factor[:, i], p)] for i in range(n_predictions)]

        compare = [[verdict(target, prediction) for target, prediction in zip(y, a[i])] for i in range(n_predictions)]
        compare_counter = [dict(Counter(compare[i])) for i in range(len(compare))]

        accuracy = ([(compare_counter[i]['hit'] +
                    compare_counter[i]['correct_rejection'])/len(compare[i]) for i in range(n_predictions)])

        predictions_accuracy[verb] = [np.mean(accuracy), np.std(accuracy)]

        comp_or = [[verdict(target, prediction) for target, prediction in zip(r, a[i])] for i in range(n_predictions)]
        compare_counter_or = [dict(Counter(comp_or[i])) for i in range(len(comp_or))]

        accuracy_or = ([(compare_counter_or[i]['hit'] +
                       compare_counter_or[i]['correct_rejection'])/len(comp_or[i]) for i in range(n_predictions)])

        predictions_accuracy[verb] += [np.mean(accuracy_or), np.std(accuracy_or)]

        print(i)
        i += 1

    return predictions_accuracy

    # test:
    prediction_accuracy = predict_past_tense()
    mean_accuracy = np.mean([prediction_accuracy[x][0] for x in prediction_accuracy.keys()])
    accuracy_std = np.std([prediction_accuracy[x][0] for x in prediction_accuracy.keys()])


def moving_average_plot(x, window=10):
    x = np.array(x)
    x_ma = [np.mean(x[i:i+window]) for i in range(x.shape[0]-window)]
    plt.plot(x_ma)
    plt.show()


def moving_average_multiline_plot(x, window=10, **kwargs):

    if 'legend' in kwargs:
        legend = kwargs['legend']
    else:
        legend = ['line_' + str(i+1) for i in range(len(x))]

    for a, lbl in zip(x, legend):
        try:
            a = np.array(a)
            a_ma = [np.mean(a[i:i+window]) for i in range(a.shape[0]-window)]
            plt.plot(a_ma, label=lbl)
        finally:
            'incorrect data'

    plt.legend(loc='right')
    plt.show()





























